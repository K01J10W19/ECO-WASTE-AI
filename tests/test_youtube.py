"""
Tests for youtube_service — the Option A Action-Protocol tutorial plugin.

Hermetic like every other external integration: TestingConfig blanks
YOUTUBE_API_KEY (keyless = the verified fallback with ZERO network calls),
and the live path is exercised against a monkeypatched requests.get.
"""
import types

import pytest
import requests

from app.services import youtube_service as ys


@pytest.fixture(autouse=True)
def _clear_search_cache():
    """The search cache is process-wide; isolate every test."""
    ys._search_youtube.cache_clear()
    yield
    ys._search_youtube.cache_clear()


def test_keyless_serves_fallback_with_zero_network(app, monkeypatch):
    def explode(*a, **k):
        raise AssertionError("network must not be touched without a key")
    monkeypatch.setattr(requests, "get", explode)

    with app.app_context():                       # TestingConfig: key = ""
        out = ys.fetch_live_youtube_data("malaysia recycling bins tutorial")

    assert out == ys.FALLBACK_VIDEO
    assert out["video_provider"] == "fallback"
    assert out["video_embed_url"].startswith("https://www.youtube.com/embed/")


def test_live_search_builds_embed_url_and_caches(app, monkeypatch):
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        return types.SimpleNamespace(status_code=200, json=lambda: {
            "items": [{"id": {"videoId": "abc123XYZ_-"},
                       "snippet": {"title": "Germany Pfand guide"}}]})

    monkeypatch.setattr(requests, "get", fake_get)

    with app.app_context():
        app.config["YOUTUBE_API_KEY"] = "yt-key"
        out1 = ys.fetch_live_youtube_data("germany pfand bottle return")
        out2 = ys.fetch_live_youtube_data("germany pfand bottle return")

    assert out1["video_embed_url"] == "https://www.youtube.com/embed/abc123XYZ_-"
    assert out1["video_title"] == "Germany Pfand guide"
    assert out1["video_provider"] == "youtube_live"
    assert out2 == out1
    assert len(calls) == 1                         # cached per unique query
    # Request contract: v3 search, ONE embeddable video, keyed.
    p = calls[0]
    assert p["maxResults"] == 1 and p["type"] == "video"
    assert p["videoEmbeddable"] == "true" and p["key"] == "yt-key"
    assert p["q"] == "germany pfand bottle return"


def test_upstream_failure_never_raises_and_degrades_to_fallback(app, monkeypatch):
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **k: types.SimpleNamespace(status_code=403, json=lambda: {}))

    with app.app_context():
        app.config["YOUTUBE_API_KEY"] = "yt-key"
        out = ys.fetch_live_youtube_data("anything at all")

    assert out == ys.FALLBACK_VIDEO                # quota/auth problems degrade


def test_empty_results_and_network_errors_degrade(app, monkeypatch):
    monkeypatch.setattr(
        requests, "get",
        lambda *a, **k: types.SimpleNamespace(status_code=200,
                                              json=lambda: {"items": []}))
    with app.app_context():
        app.config["YOUTUBE_API_KEY"] = "yt-key"
        assert ys.fetch_live_youtube_data("q one two")["video_provider"] == "fallback"

    ys._search_youtube.cache_clear()

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("offline")
    monkeypatch.setattr(requests, "get", boom)
    with app.app_context():
        app.config["YOUTUBE_API_KEY"] = "yt-key"
        assert ys.fetch_live_youtube_data("q one two")["video_provider"] == "fallback"
