"""
Tests for the Step-5 carbon module: carbon_service.calculate_impact and
POST /api/calculate-impact.

The Climatiq HTTP call is monkeypatched — no network, no real API key. We pin
both paths: live-provider factors when a key is configured, and the local
dummy fallback (the app must work with a blank key), plus the endpoint's
validation errors.
"""
import types

import pytest
import requests

from app.services import carbon_service as cs
from app.schemas.carbon import CalculateImpactResponse
from app.utils.errors import ApiError


def _fake_response(status_code=200, body=None):
    return types.SimpleNamespace(status_code=status_code,
                                 json=lambda: (body or {}))


@pytest.fixture(autouse=True)
def _clear_climatiq_cache():
    """Factor cache is process-wide; isolate every test."""
    cs._fetch_climatiq_factor.cache_clear()
    yield
    cs._fetch_climatiq_factor.cache_clear()


# --- service: local fallback path (no API key) ------------------------------

def test_calculate_impact_fallback_aggregates(app):
    with app.app_context():
        app.config["CLIMATIQ_API_KEY"] = ""
        out = cs.calculate_impact(
            [{"material": "plastic", "weight_kg": 0.5},
             {"material": "glass", "weight_kg": 2.0}], country="MY")

    assert out["provider"] == "local_dummy"
    assert out["country"] == "MY"
    assert out["items"][0]["co2e_kg"] == pytest.approx(1.55)   # 3.10 * 0.5
    assert out["items"][1]["co2e_kg"] == pytest.approx(1.70)   # 0.85 * 2.0
    assert out["total_co2e_kg"] == pytest.approx(3.25)
    assert all(i["source"] == "local_dummy" for i in out["items"])
    CalculateImpactResponse(**out)   # matches the documented contract


def test_calculate_impact_rejects_unknown_material(app):
    with app.app_context():
        with pytest.raises(ApiError) as exc:
            cs.calculate_impact([{"material": "unobtainium", "weight_kg": 1.0}])
    assert exc.value.status_code == 400


def test_calculate_impact_rejects_nonpositive_weight(app):
    with app.app_context():
        with pytest.raises(ApiError) as exc:
            cs.calculate_impact([{"material": "paper", "weight_kg": 0.0}])
    assert exc.value.status_code == 400


# --- service: live Climatiq path (mocked HTTP) -------------------------------

def test_calculate_impact_uses_climatiq_when_key_present(app, monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        return _fake_response(200, {"co2e": 2.5, "co2e_unit": "kg"})

    monkeypatch.setattr(requests, "post", fake_post)

    with app.app_context():
        app.config["CLIMATIQ_API_KEY"] = "test-key"
        out = cs.calculate_impact(
            [{"material": "plastic", "weight_kg": 2.0},
             {"material": "plastic", "weight_kg": 1.0}], country="my")

    # Factor fetched once per (material, country) and scaled locally.
    assert out["provider"] == "climatiq"
    assert out["items"][0]["carbon_factor_kg_per_kg"] == 2.5
    assert out["items"][0]["co2e_kg"] == pytest.approx(5.0)
    assert out["items"][1]["co2e_kg"] == pytest.approx(2.5)
    assert out["total_co2e_kg"] == pytest.approx(7.5)
    # Request shape: bearer auth, 1 kg probe, uppercased region selector.
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["parameters"] == {"weight": 1, "weight_unit": "kg"}
    assert captured["json"]["emission_factor"]["region"] == "MY"
    assert captured["json"]["emission_factor"]["activity_id"] == \
        cs.MATERIAL_TO_CLIMATIQ_ACTIVITY["plastic"]


def test_climatiq_http_error_becomes_clean_apierror(app, monkeypatch):
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _fake_response(401, {"message": "bad key"}))

    with app.app_context():
        app.config["CLIMATIQ_API_KEY"] = "bad-key"
        with pytest.raises(ApiError) as exc:
            cs.calculate_impact([{"material": "metal", "weight_kg": 1.0}])
    assert exc.value.status_code == 502
    assert "bad key" in str(exc.value.message)


def test_climatiq_region_miss_falls_back_to_global_factor(app, monkeypatch):
    # First call (region-scoped) has no published factor; the service retries
    # without the region instead of failing the whole request.
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json["emission_factor"].get("region"))
        if json["emission_factor"].get("region"):
            return _fake_response(400, {"message": "No emission factors could be found"})
        return _fake_response(200, {"co2e": 1.9})

    monkeypatch.setattr(requests, "post", fake_post)

    with app.app_context():
        app.config["CLIMATIQ_API_KEY"] = "test-key"
        out = cs.calculate_impact([{"material": "paper", "weight_kg": 1.0}],
                                  country="MY")

    assert calls == ["MY", None]              # scoped attempt, then global retry
    assert out["items"][0]["carbon_factor_kg_per_kg"] == 1.9
    assert out["provider"] == "climatiq"


def test_climatiq_timeout_becomes_clean_apierror(app, monkeypatch):
    def timeout_post(*a, **k):
        raise requests.exceptions.Timeout("upstream slow")
    monkeypatch.setattr(requests, "post", timeout_post)

    with app.app_context():
        app.config["CLIMATIQ_API_KEY"] = "test-key"
        with pytest.raises(ApiError) as exc:
            cs.calculate_impact([{"material": "glass", "weight_kg": 1.0}])
    assert exc.value.status_code == 502


def test_estimate_impact_uses_live_factor_when_available(app, monkeypatch):
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _fake_response(200, {"co2e": 4.0}))

    with app.app_context():
        app.config["CLIMATIQ_API_KEY"] = "test-key"
        assert cs.estimate_impact("cardboard", 0.5, country="MY") == pytest.approx(2.0)
    # Outside an app context the same call falls back to the dummy factor.
    assert cs.estimate_impact("cardboard", 0.5) == pytest.approx(0.47)  # 0.94 * 0.5


# --- endpoint: POST /api/calculate-impact ------------------------------------

def test_endpoint_happy_path_local_fallback(client):
    res = client.post("/api/calculate-impact", json={
        "items": [{"material": "plastic", "weight_kg": 0.5},
                  {"material": "general rubbish", "weight_kg": 1.0}],
        "country": "MY",
    })
    assert res.status_code == 200
    body = res.get_json()
    CalculateImpactResponse(**body)
    assert body["total_co2e_kg"] == pytest.approx(1.55 + 1.20)
    assert body["provider"] == "local_dummy"


def test_endpoint_rejects_missing_body(client):
    res = client.post("/api/calculate-impact")
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_endpoint_rejects_bad_payloads(client):
    cases = [
        {"items": []},                                              # empty list
        {"items": [{"material": "plastic", "weight_kg": 0}]},       # weight <= 0
        {"items": [{"material": "plastic", "weight_kg": 2000}]},    # weight > cap
        {"items": [{"material": "plastic"}]},                       # missing weight
        {"items": [{"material": "plastic", "weight_kg": 1}],
         "country": "MYS"},                                          # bad ISO code
    ]
    for payload in cases:
        res = client.post("/api/calculate-impact", json=payload)
        assert res.status_code == 400, payload

    res = client.post("/api/calculate-impact",
                      json={"items": [{"material": "vibranium", "weight_kg": 1}]})
    assert res.status_code == 400   # unknown material from the service layer
