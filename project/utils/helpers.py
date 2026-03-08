import os
import re
from typing import Iterable
from urllib.parse import parse_qs, urlparse


PLAY_STORE_HOST = "play.google.com"
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
GENERIC_REVIEW_NOISE_TERMS = {
    "amazing",
    "app",
    "apps",
    "awesome",
    "bad",
    "best",
    "better",
    "excellent",
    "experience",
    "fantastic",
    "good",
    "great",
    "like",
    "love",
    "much",
    "many",
    "nice",
    "please",
    "really",
    "super",
    "thank",
    "thanks",
    "very",
    "wow",
    "worst",
}


def extract_app_id(play_store_url: str) -> str:
    if not play_store_url or not play_store_url.strip():
        raise ValueError("Enter a Google Play Store app URL.")

    parsed = urlparse(play_store_url.strip())
    if PLAY_STORE_HOST not in parsed.netloc:
        raise ValueError("The URL must point to play.google.com.")

    query_params = parse_qs(parsed.query)
    app_ids = query_params.get("id", [])
    if not app_ids or not app_ids[0].strip():
        raise ValueError("Could not find the app id in the URL.")

    return app_ids[0].strip()


def get_secret_value(secret_name: str, default: str = "") -> str:
    return os.getenv(secret_name, default).strip()


def app_id_stopwords(app_id: str) -> set[str]:
    return text_identifier_stopwords(app_id)


def text_identifier_stopwords(*values: str) -> set[str]:
    cleaned: set[str] = set()
    for value in values:
        tokens = re.split(r"[^a-zA-Z]+", str(value).lower())
        cleaned.update(token for token in tokens if len(token) > 2)
    cleaned.update({"app", "apps", "application", "google", "play", "store", "review", "reviews"})
    return cleaned


def generic_review_noise_stopwords() -> set[str]:
    return set(GENERIC_REVIEW_NOISE_TERMS)


def collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def make_topic_fallback_label(keywords: Iterable[str]) -> str:
    top_terms = [term.replace("_", " ").strip() for term in keywords if term.strip()]
    top_terms = top_terms[:3]
    if not top_terms:
        return "Unlabeled Topic"
    return " / ".join(term.title() for term in top_terms)


def format_percentage(value: float) -> str:
    return f"{value * 100:.1f}%"


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_") or "export"
