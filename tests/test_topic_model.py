import pandas as pd

from project.pipeline.topic_model import fit_topic_model


def test_topic_model_returns_topics_for_repeated_patterns():
    rows = []
    for _ in range(10):
        rows.append(
            {
                "review_text": "The app has too many ads and pricing is confusing.",
                "clean_text": "app many ad pricing confusing",
            }
        )
    for _ in range(10):
        rows.append(
            {
                "review_text": "Streaming quality is good and live cricket coverage is excellent.",
                "clean_text": "streaming quality good live cricket coverage excellent",
            }
        )

    frame = pd.DataFrame(rows)
    result = fit_topic_model(frame)

    assert result["document_count"] == 20
    assert len(result["topics"]) >= 1
