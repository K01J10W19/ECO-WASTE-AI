"""
Tests for Module 3 — the Decision Making Module (recommendation_service) and
POST /api/recommend.

The DMM's NUMERIC core is local + deterministic (no network, no app context,
no API keys), so those paths run for real: taxonomy branching, the parallel
3-path simulation, the ascending-CO2e sorting engine, knowledge-grid
coverage, weight resolution and the endpoint contract. The v3.6 LLM text
layer is exercised against a MOCKED OpenAI-compatible endpoint — no network,
and TestingConfig blanks LLM_API_KEY so everything stays hermetic.
"""
import json as jsonlib
import types

import pytest
import requests

from app.schemas.recommendation import RecommendResponse
from app.services import carbon_service as cs
from app.services import recommendation_service as rs
from app.services.classification_service import MATERIAL_CLASSES
from app.utils.errors import ApiError


# --- lockstep: taxonomy x paths x factors x knowledge -------------------------

def test_every_material_has_exactly_three_disposal_paths():
    assert set(rs.DISPOSAL_PATHS) == set(MATERIAL_CLASSES)
    for material, methods in rs.DISPOSAL_PATHS.items():
        assert len(methods) == 3, material
        assert len(set(methods)) == 3, f"duplicate methods for {material}"


def test_taxonomy_branches_route_to_the_right_method_sets():
    dry = {"recycling", "incineration", "landfill"}
    for material in ("plastic", "glass", "metal", "cardboard", "paper"):
        assert set(rs.DISPOSAL_PATHS[material]) == dry, material
    assert set(rs.DISPOSAL_PATHS["biodegradable"]) == \
        {"composting", "anaerobic_digestion", "landfill"}
    assert set(rs.DISPOSAL_PATHS["general rubbish"]) == \
        {"material_recovery", "incineration", "landfill"}


def test_factor_matrix_and_knowledge_base_cover_every_path():
    # Full 7x3 coverage: every simulated path has a factor, a display name,
    # and non-empty expert pros/cons — no path can reach the payload blank.
    assert set(cs.DISPOSAL_METHOD_FACTORS) == set(MATERIAL_CLASSES)
    for material, methods in rs.DISPOSAL_PATHS.items():
        for method in methods:
            assert isinstance(
                cs.DISPOSAL_METHOD_FACTORS[material][method], float), (material, method)
            assert method in rs.METHOD_DISPLAY_NAMES, method
            knowledge = rs.EXPERT_KNOWLEDGE[material][method]
            assert knowledge["pros"].strip(), (material, method)
            assert knowledge["cons"].strip(), (material, method)


# --- sorting engine & ranking core --------------------------------------------

def test_paths_ranked_ascending_by_co2e_for_every_material():
    for material in MATERIAL_CLASSES:
        ranked = rs.simulate_disposal_paths(material, 1.0)
        assert [p["rank"] for p in ranked] == [1, 2, 3]
        assert [p["status_tag"] for p in ranked] == \
            ["Optimal", "Acceptable", "Warning"]
        impacts = [p["carbon_impact_kg"] for p in ranked]
        assert impacts == sorted(impacts), material


def test_plastic_ranking_rewards_recycling_offset():
    ranked = rs.simulate_disposal_paths("plastic", 0.5)
    # Recycling wins with a NEGATIVE net impact (avoided virgin production).
    assert ranked[0]["method"] == "recycling"
    assert ranked[0]["carbon_impact_kg"] == pytest.approx(-0.54)  # -1.08 * 0.5
    assert ranked[0]["status_tag"] == "Optimal"
    # GHG-only nuance (documented in CLAUDE.md): inert landfilled plastic
    # out-scores incineration on pure CO2e; the cons text carries the
    # microplastic caveat instead.
    assert [p["method"] for p in ranked] == ["recycling", "landfill", "incineration"]
    assert "microplastic" in ranked[1]["environmental_cons"]


def test_organics_prefer_energy_capture_over_landfill():
    ranked = rs.simulate_disposal_paths("biodegradable", 1.0)
    assert ranked[0]["method"] == "anaerobic_digestion"
    assert ranked[0]["carbon_impact_kg"] < 0          # biogas credit
    assert ranked[-1]["method"] == "landfill"
    assert "methane" in ranked[-1]["environmental_cons"]


def test_ranked_payload_carries_the_full_commentary_contract():
    required = {"method", "method_display", "rank", "status_tag",
                "carbon_factor_kg_per_kg", "carbon_impact_kg",
                "encouraging_verdict", "environmental_pros", "environmental_cons"}
    for path in rs.simulate_disposal_paths("metal", 2.0):
        assert required <= set(path)
        assert path["encouraging_verdict"].strip()
    # Factor audit: the echoed per-kg factor matches the matrix.
    best = rs.simulate_disposal_paths("metal", 2.0)[0]
    assert best["carbon_factor_kg_per_kg"] == cs.DISPOSAL_METHOD_FACTORS["metal"]["recycling"]
    assert best["carbon_impact_kg"] == pytest.approx(-8.2)   # -4.10 * 2 kg


def test_weight_scaling_never_changes_the_ranking_order():
    light = [p["method"] for p in rs.simulate_disposal_paths("cardboard", 0.1)]
    heavy = [p["method"] for p in rs.simulate_disposal_paths("cardboard", 10.0)]
    assert light == heavy


def test_simulation_is_deterministic():
    assert rs.simulate_disposal_paths("paper", 1.5) == \
        rs.simulate_disposal_paths("paper", 1.5)


def test_simulate_rejects_unknown_material_and_bad_weight():
    with pytest.raises(ApiError) as exc:
        rs.simulate_disposal_paths("unobtainium", 1.0)
    assert exc.value.status_code == 400
    with pytest.raises(ApiError):
        rs.simulate_disposal_paths("plastic", 0.0)


# --- weight resolution (dual-stage UX) -----------------------------------------

def test_recommend_for_items_resolves_both_weight_sources():
    out = rs.recommend_for_items([
        {"material": "plastic", "weight_kg": 0.5},
        {"material": "glass", "box_area_px": 16000.0},   # 16000 / 8000 = 2 kg
    ])

    audited, blind = out["items"]
    assert audited["weight_source"] == "user_weight"
    assert audited["effective_weight_kg"] == pytest.approx(0.5)
    assert blind["weight_source"] == "box_area_proxy"
    assert blind["effective_weight_kg"] == pytest.approx(2.0)

    # Per-item summary: best method + the worst-vs-best saving headline.
    assert audited["best_method"] == "recycling"
    assert audited["max_saving_kg"] == pytest.approx(1.715)  # 1.175 - (-0.54)
    assert blind["max_saving_kg"] == pytest.approx(0.68)     # 0.06 - (-0.62)

    # Aggregate: optimal-vs-worst totals (offsets keep totals negative-capable).
    assert out["summary"]["item_count"] == 2
    assert out["summary"]["optimal_total_co2e_kg"] == pytest.approx(-1.16)
    assert out["summary"]["worst_total_co2e_kg"] == pytest.approx(1.235)
    assert out["summary"]["max_saving_kg"] == pytest.approx(2.395)
    assert out["provider"] == "local_knowledge_base"
    RecommendResponse(**out)   # matches the documented contract


def test_user_weight_wins_over_box_area_when_both_supplied():
    out = rs.recommend_for_items(
        [{"material": "metal", "weight_kg": 1.0, "box_area_px": 80000.0}])
    assert out["items"][0]["weight_source"] == "user_weight"
    assert out["items"][0]["effective_weight_kg"] == pytest.approx(1.0)


def test_recommend_for_items_requires_a_size_signal():
    with pytest.raises(ApiError) as exc:
        rs.recommend_for_items([{"material": "paper"}])
    assert exc.value.status_code == 400


# --- endpoint: POST /api/recommend ---------------------------------------------

def test_endpoint_happy_path_validates_contract(client):
    res = client.post("/api/recommend", json={
        "items": [{"material": "plastic", "weight_kg": 0.5},
                  {"material": "biodegradable", "box_area_px": 8000}],
    })
    assert res.status_code == 200
    body = res.get_json()
    RecommendResponse(**body)
    assert len(body["items"]) == 2
    for item in body["items"]:
        assert len(item["recommendations"]) == 3
        assert item["recommendations"][0]["status_tag"] == "Optimal"
    assert body["provider"] == "local_knowledge_base"


def test_endpoint_rejects_missing_body(client):
    res = client.post("/api/recommend")
    assert res.status_code == 400
    assert "error" in res.get_json()


def test_endpoint_rejects_bad_payloads(client):
    cases = [
        {"items": []},                                            # empty list
        {"items": [{"material": "plastic"}]},                     # no size signal
        {"items": [{"material": "plastic", "weight_kg": 0}]},     # weight <= 0
        {"items": [{"material": "plastic", "weight_kg": 2000}]},  # weight > cap
        {"items": [{"material": "plastic", "box_area_px": -5}]},  # negative area
        {"items": [{"material": "plastic", "weight_kg": 1}],
         "country": "MYS"},                                        # bad ISO code
    ]
    for payload in cases:
        res = client.post("/api/recommend", json=payload)
        assert res.status_code == 400, payload

    res = client.post("/api/recommend",
                      json={"items": [{"material": "vibranium", "weight_kg": 1}]})
    assert res.status_code == 400   # unknown material from the service layer


def test_endpoint_blank_country_defaults_to_global(client):
    res = client.post("/api/recommend", json={
        "items": [{"material": "glass", "weight_kg": 1.0}], "country": ""})
    assert res.status_code == 200
    body = res.get_json()
    assert body["country"] is None
    assert body["provider"] == "local_knowledge_base"   # hermetic: no LLM key


# --- v3.6 LLM text layer (mocked OpenAI-compatible endpoint) ------------------

def _llm_http_response(status_code=200, content_text=""):
    """Fake requests.Response for a chat-completions call."""
    return types.SimpleNamespace(
        status_code=status_code,
        json=lambda: {"choices": [{"message": {"content": content_text}}]})


def _valid_generation_from(request_body):
    """Build a fully-covering, child-simple generation from the sent context."""
    ctx = jsonlib.loads(request_body["messages"][1]["content"])
    return jsonlib.dumps({"items": [
        {"index": item["index"], "paths": [
            {"method": path["method"],
             "encouraging_verdict": f"Rank {path['rank']} — nice and simple!",
             "environmental_pros": "Saves electricity for your town.",
             "environmental_cons": "Trash can end up on our beaches."}
            for path in item["paths"]]}
        for item in ctx["items"]]})


def test_llm_layer_enriches_fields_and_localizes_country(app, monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["url"], captured["body"], captured["headers"] = url, json, headers
        return _llm_http_response(200, _valid_generation_from(json))

    monkeypatch.setattr(requests, "post", fake_post)

    with app.app_context():
        app.config["LLM_API_KEY"] = "free-tier-key"
        out = rs.recommend_for_items(
            [{"material": "plastic", "weight_kg": 0.5}], country="my")

    assert out["provider"] == "llm_enriched"
    assert out["country"] == "MY"
    # OpenAI-compatible request shape: bearer auth, configured URL + model.
    with app.app_context():
        assert captured["url"] == app.config["LLM_API_URL"]
        assert captured["body"]["model"] == app.config["LLM_MODEL"]
    assert captured["headers"]["Authorization"] == "Bearer free-tier-key"
    # The hyper-simple constraints ride in the system prompt; the country and
    # the read-only numbers ride in the user message.
    system_prompt = captured["body"]["messages"][0]["content"]
    assert "25 words" in system_prompt
    context = jsonlib.loads(captured["body"]["messages"][1]["content"])
    assert context["country"] == "MY"
    assert context["items"][0]["paths"][0]["carbon_impact_kg"] == pytest.approx(-0.54)
    # Text fields replaced; numbers, ranks and method ids untouched.
    top = out["items"][0]["recommendations"][0]
    assert top["encouraging_verdict"] == "Rank 1 — nice and simple!"
    assert top["environmental_pros"] == "Saves electricity for your town."
    assert top["method"] == "recycling"
    assert top["carbon_impact_kg"] == pytest.approx(-0.54)
    RecommendResponse(**out)


def test_llm_layer_speaks_global_average_without_country(app, monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["body"] = json
        return _llm_http_response(200, _valid_generation_from(json))

    monkeypatch.setattr(requests, "post", fake_post)

    with app.app_context():
        app.config["LLM_API_KEY"] = "k"
        out = rs.recommend_for_items([{"material": "glass", "weight_kg": 1.0}])

    context = jsonlib.loads(captured["body"]["messages"][1]["content"])
    assert context["country"] == "global average"
    assert out["country"] is None
    assert out["provider"] == "llm_enriched"


def test_llm_transient_503_recovers_on_retry(app, monkeypatch):
    # The live failure mode: Gemini free tier throws a momentary 503 "model
    # overloaded". The layer must retry and still deliver llm_enriched.
    monkeypatch.setattr(rs.time, "sleep", lambda _s: None)   # no real backoff
    calls = []

    def flaky_post(url, json=None, headers=None, timeout=None):
        calls.append(1)
        if len(calls) < 3:
            return _llm_http_response(503)
        return _llm_http_response(200, _valid_generation_from(json))

    monkeypatch.setattr(requests, "post", flaky_post)

    with app.app_context():
        app.config["LLM_API_KEY"] = "k"
        out = rs.recommend_for_items([{"material": "plastic", "weight_kg": 0.5}])

    assert len(calls) == 3                               # 503, 503, success
    assert out["provider"] == "llm_enriched"
    assert out["items"][0]["recommendations"][0]["encouraging_verdict"] == \
        "Rank 1 — nice and simple!"


def test_llm_rate_limit_falls_back_to_local_grid(app, monkeypatch):
    monkeypatch.setattr(rs.time, "sleep", lambda _s: None)   # no real backoff
    calls = []

    def always_429(*a, **k):
        calls.append(1)
        return _llm_http_response(429)

    monkeypatch.setattr(requests, "post", always_429)

    with app.app_context():
        app.config["LLM_API_KEY"] = "k"
        out = rs.recommend_for_items([{"material": "paper", "weight_kg": 1.0}])

    assert len(calls) == 3                              # all attempts exhausted
    assert out["provider"] == "local_fallback"          # identity tag intact
    landfill = next(p for p in out["items"][0]["recommendations"]
                    if p["method"] == "landfill")
    assert landfill["environmental_cons"] == \
        rs.EXPERT_KNOWLEDGE["paper"]["landfill"]["cons"]   # grid copy served
    RecommendResponse(**out)


def test_llm_timeout_is_not_retried(app, monkeypatch):
    # A read timeout already burned the 60 s window — degrade immediately.
    calls = []

    def timeout_post(*a, **k):
        calls.append(1)
        raise requests.exceptions.Timeout("slow")

    monkeypatch.setattr(requests, "post", timeout_post)
    with app.app_context():
        app.config["LLM_API_KEY"] = "k"
        out = rs.recommend_for_items([{"material": "metal", "weight_kg": 1.0}])
    assert calls == [1]                                  # exactly one attempt
    assert out["provider"] == "local_fallback"


def test_llm_telegram_fragments_fail_the_quality_floor(app, monkeypatch):
    # Observed live on llama-3.3: fields compressed to "Great, rank 1!" /
    # "None" — below the minimum-words floor they are rejected and the
    # local grid (full sentences) is served instead.
    monkeypatch.setattr(rs.time, "sleep", lambda _s: None)   # no real backoff

    def terse_post(url, json=None, headers=None, timeout=None):
        ctx = jsonlib.loads(json["messages"][1]["content"])
        return _llm_http_response(200, jsonlib.dumps({"items": [
            {"index": item["index"], "paths": [
                {"method": p["method"], "encouraging_verdict": "Great, rank 1!",
                 "environmental_pros": "Saves 1.7 kg",
                 "environmental_cons": "None"}
                for p in item["paths"]]}
            for item in ctx["items"]]}))

    monkeypatch.setattr(requests, "post", terse_post)

    with app.app_context():
        app.config["LLM_API_KEY"] = "k"
        out = rs.recommend_for_items([{"material": "plastic", "weight_kg": 0.5}])

    assert out["provider"] == "local_fallback"
    for path in out["items"][0]["recommendations"]:
        assert len(path["encouraging_verdict"].split()) >= 5   # real sentences


def test_llm_garbage_output_falls_back(app, monkeypatch):
    monkeypatch.setattr(rs.time, "sleep", lambda _s: None)   # no real backoff
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: _llm_http_response(200, "sorry, no json"))
    with app.app_context():
        app.config["LLM_API_KEY"] = "k"
        out = rs.recommend_for_items([{"material": "metal", "weight_kg": 1.0}])
    assert out["provider"] == "local_fallback"


def test_partial_llm_coverage_is_rejected_atomically(app, monkeypatch):
    # The generation covers only ONE of the three paths → the whole enrichment
    # is discarded and NO field is left half-mutated.
    monkeypatch.setattr(rs.time, "sleep", lambda _s: None)   # no real backoff

    def fake_post(url, json=None, headers=None, timeout=None):
        ctx = jsonlib.loads(json["messages"][1]["content"])
        first = ctx["items"][0]["paths"][0]
        return _llm_http_response(200, jsonlib.dumps({"items": [
            {"index": 0, "paths": [{"method": first["method"],
                                    "encouraging_verdict": "x",
                                    "environmental_pros": "y",
                                    "environmental_cons": "z"}]}]}))

    monkeypatch.setattr(requests, "post", fake_post)

    with app.app_context():
        app.config["LLM_API_KEY"] = "k"
        out = rs.recommend_for_items([{"material": "plastic", "weight_kg": 1.0}])

    assert out["provider"] == "local_fallback"
    assert all(p["encouraging_verdict"] not in ("x",)
               for p in out["items"][0]["recommendations"])


def test_fallback_grid_and_verdicts_respect_the_word_budget():
    # v3.6 register lockstep: the local grid obeys the same "hyper-simple,
    # <= 25 words" standard the LLM prompt enforces (small tokenizer buffer).
    for material, methods in rs.DISPOSAL_PATHS.items():
        for method in methods:
            entry = rs.EXPERT_KNOWLEDGE[material][method]
            assert len(entry["pros"].split()) <= 28, (material, method)
            assert len(entry["cons"].split()) <= 28, (material, method)
        for path in rs.simulate_disposal_paths(material, 1.0):
            assert len(path["encouraging_verdict"].split()) <= 28, \
                (material, path["method"])
