import pandas as pd

from project.pipeline.sentiment import SentimentAnalyzer, get_sentiment_model_profile


def test_prepare_text_removes_urls_and_html():
    cleaned = SentimentAnalyzer._prepare_text("Great <b>app</b> https://example.com")
    assert "<b>" not in cleaned
    assert "https://" not in cleaned
    assert cleaned == "Great app"


def test_score_frame_adds_sentiment_columns_and_distribution(monkeypatch):
    frame = pd.DataFrame({"review_text": ["great", "ok", "bad"]})

    def fake_predict(self, texts, batch_size=16):
        assert texts == ["great", "ok", "bad"]
        assert batch_size == 8
        return ["positive", "neutral", "negative"], [0.9, 0.7, 0.8]

    monkeypatch.setattr(SentimentAnalyzer, "predict", fake_predict)

    analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
    scored, distribution = analyzer.score_frame(frame, batch_size=8)

    assert list(scored["sentiment"]) == ["positive", "neutral", "negative"]
    assert list(scored["sentiment_confidence"]) == [0.9, 0.7, 0.8]
    assert distribution == {"positive": 1 / 3, "neutral": 1 / 3, "negative": 1 / 3}


def test_fast_distilbert_profile_is_binary():
    profile = get_sentiment_model_profile("Fast (DistilBERT binary)")

    assert profile["model_name"] == "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    assert profile["label_map"] == {0: "negative", 1: "positive"}


def test_binary_distribution_keeps_neutral_zero(monkeypatch):
    frame = pd.DataFrame({"review_text": ["great", "bad"]})

    def fake_predict(self, texts, batch_size=16):
        return ["positive", "negative"], [0.91, 0.88]

    monkeypatch.setattr(SentimentAnalyzer, "predict", fake_predict)

    analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
    scored, distribution = analyzer.score_frame(frame, batch_size=16)

    assert list(scored["sentiment"]) == ["positive", "negative"]
    assert distribution == {"positive": 0.5, "neutral": 0.0, "negative": 0.5}
