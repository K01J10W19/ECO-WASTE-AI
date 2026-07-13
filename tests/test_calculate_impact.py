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


# --- v3.5 UX: id echo + pixel-proxy weight substitution ----------------------

def test_calculate_impact_echoes_client_item_ids(app):
    with app.app_context():
        app.config["CLIMATIQ_API_KEY"] = ""
        out = cs.calculate_impact(
            [{"id": 7, "material": "plastic", "weight_kg": 1.0},
             {"id": 3, "material": "glass", "weight_kg": 1.0},
             {"material": "paper", "weight_kg": 1.0}])          # id optional

    assert [i["id"] for i in out["items"]] == [7, 3, None]      # verbatim, in order
    CalculateImpactResponse(**out)


def test_calculate_impact_substitutes_box_area_when_weight_missing(app):
    with app.app_context():
        app.config["CLIMATIQ_API_KEY"] = ""
        out = cs.calculate_impact(
            [{"id": 0, "material": "plastic", "box_area_px": 16000.0},   # 2 kg proxy
             {"id": 1, "material": "plastic", "weight_kg": 0.5,
              "box_area_px": 999999.0}])                                 # weight wins

    proxy, audited = out["items"]
    assert proxy["weight_source"] == "box_area_proxy"
    assert proxy["weight_kg"] == pytest.approx(2.0)              # 16000 / gamma
    assert proxy["co2e_kg"] == pytest.approx(6.2)                # 3.10 * 2
    assert audited["weight_source"] == "user_weight"
    assert audited["weight_kg"] == pytest.approx(0.5)
    CalculateImpactResponse(**out)


def test_calculate_impact_requires_a_size_signal(app):
    with app.app_context():
        with pytest.raises(ApiError) as exc:
            cs.calculate_impact([{"material": "metal"}])
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


def _no_factor_body():
    """Climatiq's genuine coverage-miss shape (verified live 2026-07-14)."""
    return {"error": "bad_request",
            "error_code": "no_emission_factors_found",
            "message": "No emission factors could be found using the current "
                       "query. ... relaxing ... region."}


def test_climatiq_region_miss_falls_back_to_global_factor(app, monkeypatch):
    # No candidate id has a MY-scoped factor: the service walks the whole
    # candidate ladder region-scoped, then retries unscoped instead of
    # failing the request.
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json["emission_factor"].get("region"))
        if json["emission_factor"].get("region"):
            return _fake_response(400, _no_factor_body())
        return _fake_response(200, {"co2e": 1.9})

    monkeypatch.setattr(requests, "post", fake_post)

    with app.app_context():
        app.config["CLIMATIQ_API_KEY"] = "test-key"
        out = cs.calculate_impact([{"material": "paper", "weight_kg": 1.0}],
                                  country="MY")

    n_candidates = len(cs.CLIMATIQ_MATERIAL_MAP["paper"]["activity_ids"])
    assert calls == ["MY"] * n_candidates + [None]   # full ladder, then global
    assert out["items"][0]["carbon_factor_kg_per_kg"] == 1.9
    assert out["provider"] == "climatiq"


def test_climatiq_region_ladder_prefers_regional_alternate_id(app, monkeypatch):
    # THE US-metal scenario: the BEIS primary id has no US publication, but the
    # EPA alternate does — the user must get the genuine US-scoped factor, and
    # the region-unscoped global fallback must NOT fire.
    attempts = []

    def fake_post(url, json=None, headers=None, timeout=None):
        sel = json["emission_factor"]
        attempts.append((sel["activity_id"], sel.get("region")))
        if sel["activity_id"] == cs.CLIMATIQ_MATERIAL_MAP["metal"]["activity_ids"][0]:
            return _fake_response(400, _no_factor_body())
        return _fake_response(200, {"co2e": 0.0220,
                                    "emission_factor": {"region": "US",
                                                        "source": "EPA",
                                                        "year": 2025}})

    monkeypatch.setattr(requests, "post", fake_post)

    with app.app_context():
        app.config["CLIMATIQ_API_KEY"] = "test-key"
        out = cs.calculate_impact([{"material": "metal", "weight_kg": 2.0}],
                                  country="us")   # lower-case in, normalised

    primary, alternate = cs.CLIMATIQ_MATERIAL_MAP["metal"]["activity_ids"]
    assert attempts == [(primary, "US"), (alternate, "US")]   # never unscoped
    assert out["items"][0]["carbon_factor_kg_per_kg"] == pytest.approx(0.0220)
    assert out["items"][0]["co2e_kg"] == pytest.approx(0.0440)
    assert out["country"] == "US"


def test_climatiq_non_coverage_400_fails_loudly_not_silently(app, monkeypatch):
    # A 400 that is NOT a coverage miss (e.g. malformed selector) must raise —
    # the fallback ladder is reserved for genuine "no factor for this region".
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        calls.append(json["emission_factor"].get("region"))
        return _fake_response(400, {"error": "bad_request",
                                    "error_code": "invalid_request",
                                    "message": "data_version malformed"})

    monkeypatch.setattr(requests, "post", fake_post)

    with app.app_context():
        app.config["CLIMATIQ_API_KEY"] = "test-key"
        with pytest.raises(ApiError) as exc:
            cs.calculate_impact([{"material": "metal", "weight_kg": 1.0}],
                                country="US")
    assert exc.value.status_code == 502
    assert "data_version malformed" in str(exc.value.message)
    assert calls == ["US"]                    # first hard error stops the ladder


def test_climatiq_material_map_covers_taxonomy_and_derives_alias():
    from app.services.classification_service import MATERIAL_CLASSES

    for material in MATERIAL_CLASSES:
        cfg = cs.CLIMATIQ_MATERIAL_MAP[material]
        ids = cfg["activity_ids"]
        assert ids and all(isinstance(a, str) and a.startswith("waste") for a in ids)
        # legacy alias = the primary candidate, always
        assert cs.MATERIAL_TO_CLIMATIQ_ACTIVITY[material] == ids[0]
        # regional overrides: upper-case ISO alpha-2 keys, waste-prefixed ids
        for region, reg_ids in cfg.get("regional_activity_ids", {}).items():
            assert len(region) == 2 and region.isalpha() and region.isupper()
            assert reg_ids and all(isinstance(a, str) and a.startswith("waste")
                                   for a in reg_ids)


def test_climatiq_regional_override_id_tried_first(app, monkeypatch):
    # AU has its own DISER municipal-solid-waste factor: the region-exact
    # override must be probed FIRST (one call, no generic-ladder walk).
    attempts = []

    def fake_post(url, json=None, headers=None, timeout=None):
        sel = json["emission_factor"]
        attempts.append((sel["activity_id"], sel.get("region")))
        return _fake_response(200, {"co2e": 1.6,
                                    "emission_factor": {"region": "AU",
                                                        "source": "DISER",
                                                        "year": 2025}})

    monkeypatch.setattr(requests, "post", fake_post)

    with app.app_context():
        app.config["CLIMATIQ_API_KEY"] = "test-key"
        out = cs.calculate_impact(
            [{"material": "general rubbish", "weight_kg": 1.0}], country="AU")

    au_override = \
        cs.CLIMATIQ_MATERIAL_MAP["general rubbish"]["regional_activity_ids"]["AU"][0]
    assert attempts == [(au_override, "AU")]
    assert out["items"][0]["carbon_factor_kg_per_kg"] == pytest.approx(1.6)


def test_climatiq_global_retry_skips_regional_overrides(app, monkeypatch):
    # When every region-scoped candidate misses, the unscoped retry must walk
    # ONLY the generic ladder — a region-exact id can never serve "global".
    calls = []

    def fake_post(url, json=None, headers=None, timeout=None):
        sel = json["emission_factor"]
        calls.append((sel["activity_id"], sel.get("region")))
        if sel.get("region"):
            return _fake_response(400, _no_factor_body())
        return _fake_response(200, {"co2e": 0.6})

    monkeypatch.setattr(requests, "post", fake_post)

    with app.app_context():
        app.config["CLIMATIQ_API_KEY"] = "test-key"
        cs.calculate_impact([{"material": "glass", "weight_kg": 1.0}],
                            country="SG")

    cfg = cs.CLIMATIQ_MATERIAL_MAP["glass"]
    scoped = [c for c in calls if c[1] == "SG"]
    unscoped = [c for c in calls if c[1] is None]
    assert [a for a, _ in scoped] == \
        list(cfg["regional_activity_ids"]["SG"]) + list(cfg["activity_ids"])
    # The unscoped pass starts from the GENERIC primary (and stops on its
    # first hit) — a region-exact override id never serves "global".
    assert unscoped == [(cfg["activity_ids"][0], None)]


def test_calculate_impact_normalizes_material_and_country(app):
    # Stray whitespace/case from a caller must price like the canonical
    # strings, not 400 or fork a new factor cache slot.
    with app.app_context():
        app.config["CLIMATIQ_API_KEY"] = ""
        out = cs.calculate_impact([{"material": "  METAL ", "weight_kg": 1.0}],
                                  country="us")

    assert out["items"][0]["material"] == "metal"
    assert out["items"][0]["carbon_factor_kg_per_kg"] == pytest.approx(4.50)
    assert out["country"] == "US"
    CalculateImpactResponse(**out)


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
        {"items": [{"material": "plastic"}]},                       # no size signal
        {"items": [{"material": "plastic", "box_area_px": -5}]},    # negative area
        {"items": [{"material": "plastic", "weight_kg": 1}],
         "country": "MYS"},                                          # bad ISO code
        {"items": [{"id": 1, "material": "plastic", "weight_kg": 1},
                   {"id": 1, "material": "glass", "weight_kg": 1}]},  # dup ids
    ]
    for payload in cases:
        res = client.post("/api/calculate-impact", json=payload)
        assert res.status_code == 400, payload

    res = client.post("/api/calculate-impact",
                      json={"items": [{"material": "vibranium", "weight_kg": 1}]})
    assert res.status_code == 400   # unknown material from the service layer


def test_endpoint_grid_sync_and_proxy_flow(client):
    # The split-screen frontend posts /predict rows verbatim: id + box area,
    # user-audited weights where edited. Ids come back untouched, in order.
    res = client.post("/api/calculate-impact", json={
        "items": [{"id": 4, "material": "plastic", "box_area_px": 233712},
                  {"id": 2, "material": "paper", "weight_kg": 0.005}],
    })
    assert res.status_code == 200
    body = res.get_json()
    CalculateImpactResponse(**body)
    assert [i["id"] for i in body["items"]] == [4, 2]
    assert body["items"][0]["weight_source"] == "box_area_proxy"
    assert body["items"][0]["weight_kg"] == pytest.approx(29.214)   # 233712 / 8000
    assert body["items"][1]["weight_source"] == "user_weight"


def test_endpoint_blank_country_defaults_to_global(client):
    # An empty-string country (frontend geolocation lookup failed) must NOT
    # 400 — it coerces to None and the factors stay region-unscoped.
    res = client.post("/api/calculate-impact", json={
        "items": [{"material": "glass", "weight_kg": 1.0}],
        "country": "",
    })
    assert res.status_code == 200
    assert res.get_json()["country"] is None
