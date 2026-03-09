from __future__ import annotations

import re
from typing import Iterable

import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_SENTIMENT_MODEL_KEY = "Balanced (RoBERTa 3-class)"
DEFAULT_SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
MODEL_LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}
SENTIMENT_MODEL_PROFILES = {
    DEFAULT_SENTIMENT_MODEL_KEY: {
        "model_name": DEFAULT_SENTIMENT_MODEL,
        "label_map": MODEL_LABEL_MAP,
        "max_length": 128,
        "description": "Best semantic fidelity for positive, neutral, and negative sentiment.",
        "summary_label": "RoBERTa 3-class",
    },
    "Fast (DistilBERT binary)": {
        "model_name": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "label_map": {0: "negative", 1: "positive"},
        "max_length": 128,
        "description": "Faster CPU inference, but binary only. Neutral share will be 0 by design.",
        "summary_label": "DistilBERT binary",
    },
}
URL_PATTERN = re.compile(r"http\S+|www\S+|https\S+", re.IGNORECASE)
HTML_PATTERN = re.compile(r"<.*?>")


def get_sentiment_model_profile(profile_key: str) -> dict:
    return SENTIMENT_MODEL_PROFILES.get(profile_key, SENTIMENT_MODEL_PROFILES[DEFAULT_SENTIMENT_MODEL_KEY]).copy()


class SentimentAnalyzer:
    def __init__(
        self,
        model_name: str = DEFAULT_SENTIMENT_MODEL,
        max_length: int = 128,
        label_map: dict[int, str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.label_map = dict(label_map or MODEL_LABEL_MAP)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _prepare_text(text: str) -> str:
        prepared = str(text or "")
        prepared = URL_PATTERN.sub(" ", prepared)
        prepared = HTML_PATTERN.sub(" ", prepared)
        prepared = re.sub(r"\s+", " ", prepared).strip()
        return prepared or "empty review"

    def predict(self, texts: Iterable[str], batch_size: int = 16) -> tuple[list[str], list[float]]:
        prepared_texts = [self._prepare_text(text) for text in texts]
        labels: list[str] = []
        confidences: list[float] = []

        for start in range(0, len(prepared_texts), batch_size):
            batch = prepared_texts[start : start + batch_size]
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            probabilities = F.softmax(outputs.logits, dim=1)
            batch_indices = torch.argmax(probabilities, dim=1).tolist()
            batch_scores = torch.max(probabilities, dim=1).values.tolist()

            labels.extend(self.label_map.get(index, "neutral") for index in batch_indices)
            confidences.extend(batch_scores)

        return labels, confidences

    def score_frame(
        self,
        frame: pd.DataFrame,
        text_column: str = "review_text",
        batch_size: int = 16,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        scored = frame.copy()
        labels, confidences = self.predict(scored[text_column].tolist(), batch_size=batch_size)
        scored["sentiment"] = labels
        scored["sentiment_confidence"] = confidences

        distribution = (
            scored["sentiment"]
            .value_counts(normalize=True)
            .reindex(["positive", "neutral", "negative"], fill_value=0.0)
            .to_dict()
        )
        return scored, distribution
