from __future__ import annotations

import pandas as pd

from project.pipeline.topic_model import fit_topic_model
from project.utils.helpers import text_identifier_stopwords


def split_reviews_by_sentiment(scored_reviews: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "positive": scored_reviews[scored_reviews["sentiment"] == "positive"].copy(),
        "neutral": scored_reviews[scored_reviews["sentiment"] == "neutral"].copy(),
        "negative": scored_reviews[scored_reviews["sentiment"] == "negative"].copy(),
    }


def build_analysis_bundle(
    scored_reviews: pd.DataFrame,
    sentiment_distribution: dict[str, float],
    topic_context: str,
    vocabulary_profile: dict,
    source_name: str,
    llm_analyzer,
) -> dict:
    sentiment_frames = split_reviews_by_sentiment(scored_reviews)
    extra_stopwords = set(text_identifier_stopwords(topic_context))
    extra_stopwords.update(vocabulary_profile.get("additional_stopwords", []))

    positive_topic_result = fit_topic_model(
        sentiment_frames["positive"],
        text_column="clean_text",
        topic_count=10,
        top_topic_count=5,
        extra_stopwords=extra_stopwords,
    )
    negative_topic_result = fit_topic_model(
        sentiment_frames["negative"],
        text_column="clean_text",
        topic_count=10,
        top_topic_count=5,
        extra_stopwords=extra_stopwords,
    )

    positive_topics = llm_analyzer.label_topics(
        positive_topic_result["topics"],
        topic_context=topic_context,
        source_name=source_name,
    )
    negative_topics = llm_analyzer.label_topics(
        negative_topic_result["topics"],
        topic_context=topic_context,
        source_name=source_name,
    )
    insights = llm_analyzer.generate_business_insights(
        negative_topics=negative_topics,
        positive_topics=positive_topics,
        sentiment_distribution=sentiment_distribution,
    )

    return {
        "sentiment_frames": sentiment_frames,
        "positive_topic_result": positive_topic_result,
        "negative_topic_result": negative_topic_result,
        "positive_topics": positive_topics,
        "negative_topics": negative_topics,
        "insights": insights,
        "vocabulary_profile": vocabulary_profile,
    }
