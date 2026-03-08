from __future__ import annotations

from collections import Counter
from typing import Iterable

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


def fit_topic_model(
    frame: pd.DataFrame,
    text_column: str = "clean_text",
    topic_count: int = 10,
    top_topic_count: int = 5,
    top_words: int = 8,
    representative_samples: int = 3,
    extra_stopwords: Iterable[str] | None = None,
) -> dict:
    working = frame[frame[text_column].astype(bool)].copy()
    if len(working) < 15:
        return {
            "topics": [],
            "message": "Not enough reviews for reliable topic modeling.",
            "document_count": int(len(working)),
        }

    n_components = min(topic_count, max(2, len(working) // 20))
    min_df = 2 if len(working) < 500 else 5
    max_df = 0.9 if len(working) >= 50 else 1.0
    stop_words = list(extra_stopwords or [])

    vectorizer = CountVectorizer(
        stop_words=stop_words,
        max_features=2000,
        min_df=min_df,
        max_df=max_df,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z']+\b",
    )

    matrix = vectorizer.fit_transform(working[text_column].tolist())
    if matrix.shape[1] == 0:
        return {
            "topics": [],
            "message": "The topic vocabulary was empty after vectorization.",
            "document_count": int(len(working)),
        }

    lda = LatentDirichletAllocation(
        n_components=n_components,
        max_iter=50,
        learning_method="batch",
        random_state=42,
    )
    topic_distribution = lda.fit_transform(matrix)
    feature_names = vectorizer.get_feature_names_out()

    dominant_topics = topic_distribution.argmax(axis=1)
    topic_counts = Counter(dominant_topics)
    topic_rows: list[dict] = []

    for topic_index, component in enumerate(lda.components_):
        keyword_indices = component.argsort()[:-top_words - 1 : -1]
        keywords = [feature_names[index] for index in keyword_indices]
        topic_scores = topic_distribution[:, topic_index]
        top_document_indices = topic_scores.argsort()[-representative_samples:][::-1]
        representative_reviews = [
            working.iloc[index]["review_text"]
            for index in top_document_indices
            if topic_scores[index] > 0
        ]

        topic_rows.append(
            {
                "topic_id": int(topic_index),
                "keywords": keywords,
                "keywords_text": ", ".join(keywords),
                "prevalence": topic_counts.get(topic_index, 0) / len(working),
                "review_count": int(topic_counts.get(topic_index, 0)),
                "representative_reviews": representative_reviews,
            }
        )

    topic_rows = sorted(
        topic_rows,
        key=lambda row: (row["review_count"], row["prevalence"]),
        reverse=True,
    )[:top_topic_count]

    return {
        "topics": topic_rows,
        "message": "",
        "document_count": int(len(working)),
        "model_topic_count": int(n_components),
    }
