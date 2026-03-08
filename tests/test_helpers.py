from project.utils.helpers import app_id_stopwords, extract_app_id


def test_extract_app_id_from_google_play_url():
    url = "https://play.google.com/store/apps/details?id=in.startv.hotstar&hl=en_IN"
    assert extract_app_id(url) == "in.startv.hotstar"


def test_extract_app_id_rejects_non_play_store_urls():
    try:
        extract_app_id("https://example.com/app?id=test.app")
    except ValueError as error:
        assert "play.google.com" in str(error)
    else:
        raise AssertionError("Expected ValueError for non Play Store URL.")


def test_app_id_stopwords_contains_brand_tokens():
    tokens = app_id_stopwords("in.startv.hotstar")
    assert "startv" in tokens
    assert "hotstar" in tokens
