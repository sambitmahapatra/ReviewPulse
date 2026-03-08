import pandas as pd

from project.pipeline import scraper


def test_fetch_reviews_dataframe_normalizes_schema_and_deduplicates(monkeypatch):
    calls = []

    def fake_reviews(app_id, lang, country, sort, count, continuation_token):
        calls.append(
            {
                "app_id": app_id,
                "lang": lang,
                "country": country,
                "count": count,
                "continuation_token": continuation_token,
            }
        )
        batch = [
            {
                "at": "2026-03-01T10:00:00",
                "score": 5,
                "content": "Great app",
                "thumbsUpCount": 4,
            },
            {
                "at": "2026-03-01T10:00:00",
                "score": 5,
                "content": "Great app",
                "thumbsUpCount": 4,
            },
            {
                "at": "2026-03-02T12:00:00",
                "score": 1,
                "content": "Terrible support",
                "thumbsUpCount": 1,
            },
        ]
        return batch, None

    monkeypatch.setattr(scraper, "reviews", fake_reviews)

    frame = scraper.fetch_reviews_dataframe(
        app_id="demo.app",
        max_reviews=50000,
        lang="en",
        country="us",
        sort_by="newest",
    )

    assert list(frame.columns) == ["date", "rating", "review_text", "likes"]
    assert len(frame) == 2
    assert calls[0]["count"] == 200
    assert calls[0]["app_id"] == "demo.app"
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])


def test_fetch_reviews_dataframe_rejects_empty_results(monkeypatch):
    monkeypatch.setattr(scraper, "reviews", lambda *args, **kwargs: ([], None))

    try:
        scraper.fetch_reviews_dataframe("missing.app")
    except ValueError as error:
        assert "No reviews" in str(error)
    else:
        raise AssertionError("Expected ValueError when the scraper returns no reviews.")
