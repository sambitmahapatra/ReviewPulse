import pandas as pd

from project.pipeline.sentiment import SentimentAnalyzer


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
