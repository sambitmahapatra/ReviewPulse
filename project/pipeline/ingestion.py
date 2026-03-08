from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


CANONICAL_COLUMNS = [
    "date",
    "rating",
    "review_text",
    "likes",
    "source",
    "product_id",
    "product_name",
]
UNMAPPED_COLUMN = "__not_provided__"

COMMON_COLUMN_ALIASES: dict[str, list[str]] = {
    "date": ["date", "review_date", "created_at", "created", "timestamp", "time", "at"],
    "rating": ["rating", "score", "stars", "star_rating", "review_rating"],
    "review_text": ["review_text", "review", "text", "content", "comment", "body", "message"],
    "likes": ["likes", "thumbsupcount", "helpful", "helpful_votes", "votes"],
    "product_id": ["product_id", "app_id", "item_id", "sku", "asin", "id"],
    "product_name": ["product_name", "app_name", "name", "title", "product", "item_name"],
}


@dataclass
class IngestionResult:
    frame: pd.DataFrame
    analysis_label: str
    topic_context: str


def normalize_play_store_frame(frame: pd.DataFrame, app_id: str) -> IngestionResult:
    normalized = frame.copy()
    normalized["source"] = "google_play"
    normalized["product_id"] = app_id
    normalized["product_name"] = app_id
    normalized = ensure_canonical_columns(normalized)
    return IngestionResult(
        frame=normalized,
        analysis_label=app_id,
        topic_context=app_id,
    )


def suggest_column_mapping(columns: list[str]) -> dict[str, str]:
    normalized_lookup = {_normalize_name(column): column for column in columns}
    mapping: dict[str, str] = {}

    for target_column, aliases in COMMON_COLUMN_ALIASES.items():
        selected = UNMAPPED_COLUMN
        for alias in aliases:
            matched = normalized_lookup.get(_normalize_name(alias))
            if matched:
                selected = matched
                break
        mapping[target_column] = selected

    return mapping


def normalize_uploaded_frame(
    frame: pd.DataFrame,
    mapping: dict[str, str],
    source_name: str,
    product_name: str,
    product_id: str,
) -> IngestionResult:
    if frame.empty:
        raise ValueError("The uploaded CSV is empty.")

    working = pd.DataFrame()
    for canonical_column in CANONICAL_COLUMNS:
        mapped_column = mapping.get(canonical_column, UNMAPPED_COLUMN)
        if mapped_column != UNMAPPED_COLUMN and mapped_column in frame.columns:
            working[canonical_column] = frame[mapped_column]

    if "review_text" not in working.columns:
        raise ValueError("Map at least one column to `review_text`.")

    working["review_text"] = working["review_text"].fillna("").astype(str).str.strip()
    working = working[working["review_text"].astype(bool)].copy()
    if working.empty:
        raise ValueError("No non-empty review text was found after mapping.")

    if "date" in working.columns:
        working["date"] = pd.to_datetime(working["date"], errors="coerce").dt.normalize()
    else:
        working["date"] = pd.NaT

    if "rating" in working.columns:
        working["rating"] = pd.to_numeric(working["rating"], errors="coerce")
    else:
        working["rating"] = pd.NA

    if "likes" in working.columns:
        working["likes"] = pd.to_numeric(working["likes"], errors="coerce").fillna(0)
    else:
        working["likes"] = 0

    working["likes"] = working["likes"].astype(int)
    working["source"] = source_name.strip() or "csv_upload"
    if "product_name" in working.columns:
        working["product_name"] = working["product_name"].fillna("").astype(str)
    else:
        working["product_name"] = ""
    if "product_id" in working.columns:
        working["product_id"] = working["product_id"].fillna("").astype(str)
    else:
        working["product_id"] = ""

    if product_name.strip():
        working["product_name"] = product_name.strip()
    if product_id.strip():
        working["product_id"] = product_id.strip()

    working = ensure_canonical_columns(working)
    working = working.drop_duplicates(subset=["review_text", "date", "rating"]).reset_index(drop=True)
    if working.empty:
        raise ValueError("No reviews remained after normalizing the uploaded CSV.")

    label = product_name.strip() or product_id.strip() or source_name.strip() or "csv_upload"
    topic_context = " ".join(part for part in [product_name.strip(), product_id.strip(), source_name.strip()] if part).strip()
    return IngestionResult(frame=working, analysis_label=label, topic_context=topic_context)


def ensure_canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    for column in CANONICAL_COLUMNS:
        if column not in normalized.columns:
            if column == "likes":
                normalized[column] = 0
            elif column == "date":
                normalized[column] = pd.NaT
            else:
                normalized[column] = ""

    normalized["review_text"] = normalized["review_text"].fillna("").astype(str)
    normalized["source"] = normalized["source"].fillna("").astype(str)
    normalized["product_id"] = normalized["product_id"].fillna("").astype(str)
    normalized["product_name"] = normalized["product_name"].fillna("").astype(str)
    normalized["likes"] = pd.to_numeric(normalized["likes"], errors="coerce").fillna(0).astype(int)
    return normalized[CANONICAL_COLUMNS].copy()


def _normalize_name(value: str) -> str:
    return "".join(character.lower() for character in str(value) if character.isalnum())
