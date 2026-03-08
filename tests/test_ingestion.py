import pandas as pd

from project.pipeline.ingestion import UNMAPPED_COLUMN, normalize_uploaded_frame, suggest_column_mapping


def test_suggest_column_mapping_detects_common_review_columns():
    mapping = suggest_column_mapping(["Review", "Score", "Created_At", "Helpful_Votes"])
    assert mapping["review_text"] == "Review"
    assert mapping["rating"] == "Score"
    assert mapping["date"] == "Created_At"
    assert mapping["likes"] == "Helpful_Votes"


def test_normalize_uploaded_frame_creates_canonical_schema():
    frame = pd.DataFrame(
        {
            "review_body": ["Great delivery", "Not happy with packaging"],
            "stars": [5, 2],
            "created_at": ["2026-03-01", "2026-03-02"],
        }
    )
    mapping = {
        "review_text": "review_body",
        "rating": "stars",
        "date": "created_at",
        "likes": UNMAPPED_COLUMN,
        "product_name": UNMAPPED_COLUMN,
        "product_id": UNMAPPED_COLUMN,
    }

    result = normalize_uploaded_frame(
        frame,
        mapping=mapping,
        source_name="amazon_csv",
        product_name="Sample Product",
        product_id="SKU-123",
    )

    assert list(result.frame.columns) == ["date", "rating", "review_text", "likes", "source", "product_id", "product_name"]
    assert result.analysis_label == "Sample Product"
    assert result.frame["source"].iloc[0] == "amazon_csv"
    assert result.frame["product_id"].iloc[0] == "SKU-123"
    assert result.frame["likes"].iloc[0] == 0
