from __future__ import annotations

import pandas as pd
from google_play_scraper import Sort, reviews


MAX_ALLOWED_REVIEWS = 10000
SORT_OPTIONS: dict[str, Sort] = {"newest": Sort.NEWEST}
if hasattr(Sort, "MOST_RELEVANT"):
    SORT_OPTIONS["most_relevant"] = getattr(Sort, "MOST_RELEVANT")
if hasattr(Sort, "RATING"):
    SORT_OPTIONS["rating"] = getattr(Sort, "RATING")
if hasattr(Sort, "HELPFULNESS"):
    SORT_OPTIONS["helpfulness"] = getattr(Sort, "HELPFULNESS")


def fetch_reviews_dataframe(
    app_id: str,
    max_reviews: int = 2000,
    lang: str = "en",
    country: str = "us",
    sort_by: str = "newest",
) -> pd.DataFrame:
    capped_reviews = min(max(int(max_reviews), 1), MAX_ALLOWED_REVIEWS)
    sort = SORT_OPTIONS.get(sort_by, Sort.NEWEST)

    all_reviews: list[dict] = []
    continuation_token = None

    while len(all_reviews) < capped_reviews:
        batch_size = min(200, capped_reviews - len(all_reviews))
        batch, continuation_token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=sort,
            count=batch_size,
            continuation_token=continuation_token,
        )

        if not batch:
            break

        all_reviews.extend(batch)
        if not continuation_token:
            break

    if not all_reviews:
        raise ValueError("No reviews were returned for the selected app.")

    frame = pd.DataFrame(all_reviews)
    required_columns = {
        "at": "date",
        "score": "rating",
        "content": "review_text",
        "thumbsUpCount": "likes",
    }
    frame = frame.rename(columns=required_columns)
    frame = frame[list(required_columns.values())].copy()

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["review_text"] = frame["review_text"].fillna("").astype(str).str.strip()
    frame["rating"] = pd.to_numeric(frame["rating"], errors="coerce").fillna(0).astype(int)
    frame["likes"] = pd.to_numeric(frame["likes"], errors="coerce").fillna(0).astype(int)

    frame = frame[frame["review_text"].astype(bool)].copy()
    frame = frame.drop_duplicates(subset=["date", "rating", "review_text"]).reset_index(drop=True)

    if frame.empty:
        raise ValueError("The app reviews were empty after schema cleanup.")

    return frame
