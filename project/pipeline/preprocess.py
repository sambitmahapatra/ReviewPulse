from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import nltk
import pandas as pd
from nltk.corpus import stopwords as nltk_stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


NEGATION_TERMS = {"no", "not", "nor", "never"}
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?")
URL_PATTERN = re.compile(r"http\S+|www\S+|https\S+", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<.*?>")


@dataclass(frozen=True)
class NltkResourceStatus:
    stopwords: bool
    wordnet: bool


class ConservativeLemmatizer:
    def lemmatize(self, token: str) -> str:
        value = str(token or "").lower()
        if len(value) <= 3:
            return value
        if value.endswith("ies") and len(value) > 4:
            return value[:-3] + "y"
        if value.endswith(("sses", "shes", "ches", "xes", "zes")) and len(value) > 4:
            return value[:-2]
        if value.endswith("s") and not value.endswith(("ss", "us", "is")):
            return value[:-1]
        return value


@lru_cache(maxsize=1)
def get_nltk_resource_status() -> NltkResourceStatus:
    def has_resource(resource_path: str) -> bool:
        try:
            nltk.data.find(resource_path)
            return True
        except LookupError:
            return False

    return NltkResourceStatus(
        stopwords=has_resource("corpora/stopwords"),
        wordnet=has_resource("corpora/wordnet"),
    )


@lru_cache(maxsize=1)
def get_stop_words() -> set[str]:
    resource_status = get_nltk_resource_status()
    if resource_status.stopwords:
        stop_words = set(nltk_stopwords.words("english"))
    else:
        stop_words = set(ENGLISH_STOP_WORDS)
    return stop_words - NEGATION_TERMS


@lru_cache(maxsize=1)
def get_lemmatizer() -> WordNetLemmatizer | ConservativeLemmatizer:
    resource_status = get_nltk_resource_status()
    if resource_status.wordnet:
        return WordNetLemmatizer()
    return ConservativeLemmatizer()


def normalize_review_text(text: str) -> str:
    text = str(text or "")
    text = text.lower()
    text = URL_PATTERN.sub(" ", text)
    text = HTML_PATTERN.sub(" ", text)
    text = re.sub(r"[^a-z'\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_review_text(text: str, preserve_terms: set[str] | None = None) -> str:
    normalized = normalize_review_text(text)
    tokens = TOKEN_PATTERN.findall(normalized)
    if not tokens:
        return ""

    stop_words = get_stop_words()
    lemmatizer = get_lemmatizer()
    preserved = {str(term).lower() for term in (preserve_terms or set()) if str(term).strip()}
    cleaned_tokens: list[str] = []
    for token in tokens:
        if len(token) <= 1:
            continue
        lemma = lemmatizer.lemmatize(token)
        if token in preserved or lemma in preserved:
            cleaned_tokens.append(lemma)
            continue
        if token in stop_words:
            continue
        cleaned_tokens.append(lemma)
    return " ".join(cleaned_tokens)


def preprocess_reviews_frame(frame: pd.DataFrame, preserve_terms: set[str] | None = None) -> pd.DataFrame:
    processed = frame.copy()
    processed["clean_text"] = processed["review_text"].apply(
        lambda value: preprocess_review_text(value, preserve_terms=preserve_terms)
    )
    processed = processed[processed["clean_text"].astype(bool)].reset_index(drop=True)
    if processed.empty:
        raise ValueError("No reviews remained after preprocessing.")
    return processed
