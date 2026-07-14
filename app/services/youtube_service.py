"""
YouTube service — the Action-Protocol tutorial plugin (Option A).

The DMM's LLM layer generates a hyper-localized ``video_search_query``
(e.g. "Malaysia coloured recycling bins tutorial"); this module resolves it
to ONE guaranteed-live tutorial via the YouTube Data API v3 search endpoint
and returns a stable ``https://www.youtube.com/embed/<id>`` URL the SPA can
iframe directly. The model never invents URLs — it only writes the query,
so a hallucinated/dead link is architecturally impossible.

RESILIENCE CONTRACT (mirrors the Climatiq/LLM pattern):
  * Blank ``YOUTUBE_API_KEY`` → the verified universal fallback video is
    served with ZERO network calls (the app must boot and pass tests with
    no key — CLAUDE.md hard rule).
  * One cached upstream search per unique query (``lru_cache``): a v3
    search costs 100 of the free tier's 10,000 daily quota units, so
    repeated scans of the same material/country are free.
  * ``fetch_live_youtube_data`` NEVER raises — any upstream problem (quota,
    network, malformed payload) logs a warning and degrades to the fallback
    dict, so recommendations can never 502 or block on YouTube.

The fallback video id was verified live via YouTube's keyless oEmbed
endpoint on 2026-07-14: "How Recycling Works" (SciShow) — a universal,
material-agnostic tutorial.
"""
import logging
from functools import lru_cache

from flask import current_app, has_app_context

logger = logging.getLogger(__name__)

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
_YT_TIMEOUT_S = 6

# Verified universal tutorial (oEmbed-checked 2026-07-14) — served whenever
# the live search is unavailable (no key, quota exhausted, network down).
FALLBACK_VIDEO = {
    "video_embed_url": "https://www.youtube.com/embed/b7GMpjx2jDQ",
    "video_title": "How Recycling Works (universal tutorial)",
    "video_provider": "fallback",
}


def _youtube_api_key() -> str:
    """The configured YouTube key, or '' outside an app context / when unset."""
    if not has_app_context():
        return ""
    return str(current_app.config.get("YOUTUBE_API_KEY", "") or "")


@lru_cache(maxsize=128)
def _search_youtube(query: str, api_key: str) -> tuple:
    """(video_id, title) for the top result — cached per unique query.

    Raises on ANY upstream problem; ``fetch_live_youtube_data`` catches and
    serves the fallback. ``lru_cache`` only stores successful lookups, so a
    transient failure can be retried by a later request.
    """
    import requests  # local import keeps module import light for tests

    resp = requests.get(
        YOUTUBE_SEARCH_URL,
        params={
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": 1,
            "videoEmbeddable": "true",   # never hand the iframe a non-embeddable id
            "safeSearch": "strict",
            "key": api_key,
        },
        timeout=_YT_TIMEOUT_S,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"YouTube search returned HTTP {resp.status_code}")
    items = resp.json().get("items") or []
    if not items:
        raise RuntimeError(f"YouTube search returned no videos for {query!r}")
    video_id = items[0]["id"]["videoId"]
    title = str(items[0].get("snippet", {}).get("title", "")).strip() or query
    return video_id, title


def fetch_live_youtube_data(search_query: str) -> dict:
    """
    Resolve ``search_query`` to ONE live, embeddable tutorial video.

    Returns ``{"video_embed_url", "video_title", "video_provider"}`` where
    provider is ``"youtube_live"`` (real v3 result) or ``"fallback"`` (the
    verified universal video). NEVER raises and makes ZERO network calls
    without a configured key — the recommendation pipeline must never block
    or fail because of YouTube.
    """
    api_key = _youtube_api_key()
    query = str(search_query or "").strip()
    if not api_key or not query:
        return dict(FALLBACK_VIDEO)
    try:
        video_id, title = _search_youtube(query, api_key)
        return {
            "video_embed_url": f"https://www.youtube.com/embed/{video_id}",
            "video_title": title,
            "video_provider": "youtube_live",
        }
    except Exception as exc:  # noqa: BLE001 - never let YouTube break the DMM
        logger.warning("YouTube tutorial lookup failed for %r (%s); serving "
                       "the universal fallback video.", query, exc)
        return dict(FALLBACK_VIDEO)
