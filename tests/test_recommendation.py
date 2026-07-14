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


@pytest.fixture(autouse=True)
def _reset_llm_rate_limit_breaker():
    """The 429 cooldown breaker is module-level state; isolate every test."""
    rs._llm_cooldown_until = 0.0
    yield
    rs._llm_cooldown_until = 0.0


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
                "is_applicable", "restriction_reason",
                "carbon_factor_kg_per_kg", "base_factor_kg_per_kg",
                "carbon_impact_kg", "action_steps",
                "encouraging_verdict", "environmental_pros", "environmental_cons"}
    for path in rs.simulate_disposal_paths("metal", 2.0):
        assert required <= set(path)
        assert path["encouraging_verdict"].strip()
        assert len(path["action_steps"]) == 2               # exactly two steps
        assert all(s.strip() for s in path["action_steps"])
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


# --- v3.7 national infrastructure applicability --------------------------------

def test_national_profiles_only_ban_known_methods():
    method_universe = {m for paths in rs.DISPOSAL_PATHS.values() for m in paths}
    for country, profile in cs.NATIONAL_INFRASTRUCTURE_PROFILES.items():
        assert len(country) == 2 and country.isupper()
        assert profile["banned_methods"] <= method_universe, country
        assert profile["reason"].strip(), country
        # A profile must never be able to wipe out a whole branch.
        for material, methods in rs.DISPOSAL_PATHS.items():
            assert set(methods) - profile["banned_methods"], (country, material)


def test_no_country_leaves_every_path_applicable():
    for path in rs.simulate_disposal_paths("plastic", 1.0):
        assert path["is_applicable"] is True
        assert path["restriction_reason"] is None


def test_sg_zero_landfill_reranks_among_applicable_paths():
    # country="SG" alone (no explicit grid_intensity) applies the ban at the
    # GB anchor; the grid datum is resolved by recommend_for_items, so here
    # we pass SG's fallback intensity exactly as the aggregate entry would.
    ranked = rs.simulate_disposal_paths("plastic", 0.5, country="SG",
                                        grid_intensity=0.408)
    assert len(ranked) == 3                       # banned path stays visible
    banned = next(p for p in ranked if p["method"] == "landfill")
    assert banned["is_applicable"] is False
    assert banned["rank"] is None and banned["status_tag"] == "Banned"
    assert "Singapore" in banned["restriction_reason"]
    assert "not an option" in banned["encouraging_verdict"]
    # Still priced (bio-decay branch: 0.09 x (1 + (ratio-1) x 0.5) x 0.5 kg).
    assert banned["carbon_impact_kg"] == pytest.approx(0.0668, abs=1e-4)

    # Ranking recomputed EXCLUSIVELY among the applicable pool: without the
    # ban, landfill held rank 2 — now incineration inherits it.
    applicable = [p for p in ranked if p["is_applicable"]]
    assert [p["method"] for p in applicable] == ["recycling", "incineration"]
    assert [p["rank"] for p in applicable] == [1, 2]
    assert [p["status_tag"] for p in applicable] == ["Optimal", "Acceptable"]
    # Verdict deltas draw from the applicable pool's SCALED numbers:
    # worst (2.35 - 0.5x0.201) x 0.5 = 1.1247, best -2.1287 x 0.5 = -1.0643.
    assert "2.189" in applicable[0]["encouraging_verdict"]


def test_de_landfill_ban_applies_and_my_stays_unrestricted():
    de = rs.simulate_disposal_paths("paper", 1.0, country="DE")
    assert next(p for p in de if p["method"] == "landfill")["is_applicable"] is False
    my = rs.simulate_disposal_paths("paper", 1.0, country="my")   # normalised
    assert all(p["is_applicable"] for p in my)


def test_recommend_summary_and_savings_use_applicable_pool_only():
    # Biodegradable under SG: landfill (the global worst) is banned, so the
    # worst-case baseline must pivot to composting. Numbers are the v3.8
    # SG-scaled factors (hermetic fallback map: 0.408 kg/kWh, ratio 1.9710):
    #   AD         -0.14 - 0.2 x (0.408 - 0.207)        = -0.1802
    #   composting  0.05 x (1 + 0.9710 x 0.3)           =  0.0646
    out = rs.recommend_for_items(
        [{"material": "biodegradable", "weight_kg": 1.0}], country="SG")

    item = out["items"][0]
    assert item["best_method"] == "anaerobic_digestion"
    assert item["max_saving_kg"] == pytest.approx(0.2448, abs=1e-4)
    assert out["summary"]["optimal_total_co2e_kg"] == pytest.approx(-0.1802)
    assert out["summary"]["worst_total_co2e_kg"] == pytest.approx(0.0646)
    assert out["summary"]["max_saving_kg"] == pytest.approx(0.2448, abs=1e-4)
    RecommendResponse(**out)


def test_all_paths_banned_fails_open(monkeypatch):
    monkeypatch.setitem(
        cs.NATIONAL_INFRASTRUCTURE_PROFILES, "XX",
        {"banned_methods": frozenset({"material_recovery", "incineration",
                                      "landfill"}),
         "reason": "test profile"})
    ranked = rs.simulate_disposal_paths("general rubbish", 1.0, country="XX")
    assert all(p["is_applicable"] for p in ranked)   # engine failed open
    assert [p["rank"] for p in ranked] == [1, 2, 3]


def test_endpoint_sg_flags_banned_path_and_validates_schema(client):
    res = client.post("/api/recommend", json={
        "items": [{"material": "biodegradable", "weight_kg": 1.0}],
        "country": "SG",
    })
    assert res.status_code == 200
    body = res.get_json()
    RecommendResponse(**body)
    paths = body["items"][0]["recommendations"]
    assert len(paths) == 3
    banned = next(p for p in paths if p["method"] == "landfill")
    assert banned["is_applicable"] is False
    assert banned["rank"] is None
    applicable = [p for p in paths if p["is_applicable"]]
    assert [p["rank"] for p in applicable] == [1, 2]
    assert body["items"][0]["best_method"] == applicable[0]["method"]


def test_llm_context_and_enrichment_exclude_banned_paths(app, monkeypatch):
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["body"] = json
        return _llm_http_response(200, _valid_generation_from(json))

    monkeypatch.setattr(requests, "post", fake_post)

    with app.app_context():
        app.config["LLM_API_KEY"] = "k"
        out = rs.recommend_for_items(
            [{"material": "biodegradable", "weight_kg": 1.0}], country="SG")

    assert out["provider"] == "llm_enriched"
    # The LLM saw ONLY the applicable pool…
    context = jsonlib.loads(captured["body"]["messages"][1]["content"])
    assert [p["method"] for p in context["items"][0]["paths"]] == \
        ["anaerobic_digestion", "composting"]
    # …its deltas were applicable-pool relative (SG-scaled numbers).
    assert context["items"][0]["paths"][0]["saving_vs_worst_kg"] == \
        pytest.approx(0.2448, abs=1e-4)
    # …and the banned path kept its deterministic policy verdict untouched.
    banned = next(p for p in out["items"][0]["recommendations"]
                  if p["method"] == "landfill")
    assert "not an option" in banned["encouraging_verdict"]
    assert banned["encouraging_verdict"] != "Rank None — nice and simple!"


# --- v3.8 grid-intensity proxy scaling engine -----------------------------------

def test_grid_scaling_formulas_follow_the_spec():
    gi = 0.585                                    # MY fallback intensity
    ratio = gi / cs.BASE_GRID_INTENSITY
    # Electricity-intensive: base x grid_ratio.
    assert cs.get_scaled_disposal_factor("plastic", "recycling", gi) == \
        pytest.approx(-1.08 * ratio)
    assert cs.get_scaled_disposal_factor("general rubbish", "material_recovery",
                                         gi) == pytest.approx(0.30 * ratio)
    # Energy-offsettable: base - energy_yield x (local - anchor) — the credit
    # is anchor-relative because the base already nets energy recovery at GB.
    delta = gi - cs.BASE_GRID_INTENSITY
    assert cs.get_scaled_disposal_factor("plastic", "incineration", gi) == \
        pytest.approx(2.35 - 0.5 * delta)
    assert cs.get_scaled_disposal_factor("biodegradable", "anaerobic_digestion",
                                         gi) == pytest.approx(-0.14 - 0.2 * delta)
    # Bio-decay: base x (1 + (grid_ratio - 1) x penalty_weight).
    assert cs.get_scaled_disposal_factor("paper", "landfill", gi) == \
        pytest.approx(1.29 * (1.0 + (ratio - 1.0) * 0.5))
    assert cs.get_scaled_disposal_factor("biodegradable", "composting", gi) == \
        pytest.approx(0.05 * (1.0 + (ratio - 1.0) * 0.3))


def test_grid_anchor_identity_leaves_baseline_untouched():
    # At the GB anchor (and with no intensity at all) every branch collapses
    # to the curated baseline EXACTLY — the engine's anchor property, and why
    # every legacy no-country expectation stays byte-identical.
    for material, methods in rs.DISPOSAL_PATHS.items():
        for method in methods:
            base = cs.DISPOSAL_METHOD_FACTORS[material][method]
            assert cs.get_scaled_disposal_factor(
                material, method, cs.BASE_GRID_INTENSITY) == pytest.approx(base)
            assert cs.get_scaled_disposal_factor(
                material, method, None) == pytest.approx(base)


def test_resolve_grid_intensity_ladder(app):
    cs._fetch_climatiq_grid_intensity.cache_clear()
    # (a) No country -> the baseline anchor, ratio exactly 1.
    out = cs.resolve_grid_intensity(None)
    assert out == {"intensity_kg_per_kwh": cs.BASE_GRID_INTENSITY,
                   "ratio": 1.0, "source": "baseline"}
    # (b) Hermetic (blank key): a known country hits the local average map.
    with app.app_context():
        app.config["CLIMATIQ_API_KEY"] = ""
        out = cs.resolve_grid_intensity("my")            # normalised too
    assert out["intensity_kg_per_kwh"] == pytest.approx(0.585)
    assert out["source"] == "local_grid_map"
    # (c) Unknown country -> anchor; the resolver NEVER raises.
    out = cs.resolve_grid_intensity("ZZ")
    assert out["source"] == "baseline" and out["ratio"] == 1.0


def test_resolve_grid_intensity_live_probe_and_graceful_fallback(app, monkeypatch):
    cs._fetch_climatiq_grid_intensity.cache_clear()
    captured = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["body"] = json
        return types.SimpleNamespace(status_code=200,
                                     json=lambda: {"co2e": 0.402})

    monkeypatch.setattr(requests, "post", fake_post)
    with app.app_context():
        app.config["CLIMATIQ_API_KEY"] = "test-key"
        out = cs.resolve_grid_intensity("SG")
    assert out == {"intensity_kg_per_kwh": 0.402,
                   "ratio": round(0.402 / cs.BASE_GRID_INTENSITY, 4),
                   "source": "climatiq"}
    sel = captured["body"]["emission_factor"]
    assert sel["activity_id"] == cs.CLIMATIQ_GRID_ACTIVITY_ID
    assert sel["region"] == "SG"
    assert captured["body"]["parameters"] == {"energy": 1, "energy_unit": "kWh"}

    # Probe failure NEVER raises — it degrades to the local average map.
    cs._fetch_climatiq_grid_intensity.cache_clear()
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **k: types.SimpleNamespace(status_code=500, json=lambda: {}))
    with app.app_context():
        app.config["CLIMATIQ_API_KEY"] = "test-key"
        out = cs.resolve_grid_intensity("SG")
    assert out["source"] == "local_grid_map"
    assert out["intensity_kg_per_kwh"] == pytest.approx(0.408)


def test_recommend_response_carries_grid_audit_block():
    out = rs.recommend_for_items([{"material": "plastic", "weight_kg": 1.0}],
                                 country="MY")
    grid = out["grid"]
    assert grid["country"] == "MY"
    assert grid["intensity_kg_per_kwh"] == pytest.approx(0.585)
    assert grid["base_intensity_kg_per_kwh"] == pytest.approx(0.207)
    assert grid["ratio"] == pytest.approx(0.585 / 0.207, abs=1e-4)
    assert grid["source"] == "local_grid_map"            # hermetic: no key
    # The SCALED factor priced the path; the GB baseline rides along.
    top = out["items"][0]["recommendations"][0]
    assert top["method"] == "recycling"
    assert top["base_factor_kg_per_kg"] == pytest.approx(-1.08)
    assert top["carbon_factor_kg_per_kg"] == pytest.approx(-3.0522, abs=1e-4)
    assert top["carbon_impact_kg"] == pytest.approx(-3.0522, abs=5e-4)
    RecommendResponse(**out)


def test_grid_scaling_can_reshuffle_ranks_regionally():
    # General rubbish at the GB anchor: MRF (0.30) beats incineration (0.45).
    # On Singapore's dirtier grid the energy-substitution credit flips the
    # ranking — incineration becomes the nationally Optimal path.
    baseline = rs.simulate_disposal_paths("general rubbish", 1.0)
    assert baseline[0]["method"] == "material_recovery"

    sg = rs.simulate_disposal_paths("general rubbish", 1.0, country="SG",
                                    grid_intensity=0.408)
    applicable = [p for p in sg if p["is_applicable"]]
    assert applicable[0]["method"] == "incineration"     # 0.3495 < 0.5913
    assert applicable[0]["status_tag"] == "Optimal"


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

def _llm_http_response(status_code=200, content_text="", headers=None):
    """Fake requests.Response for a chat-completions call."""
    return types.SimpleNamespace(
        status_code=status_code,
        headers=headers or {},
        json=lambda: {"choices": [{"message": {"content": content_text}}]})


def _valid_generation_from(request_body):
    """Build a fully-covering, child-simple generation from the sent context."""
    ctx = jsonlib.loads(request_body["messages"][1]["content"])
    return jsonlib.dumps({"items": [
        {"index": item["index"], "paths": [
            {"method": path["method"],
             "encouraging_verdict": f"Rank {path['rank']} — nice and simple!",
             "environmental_pros": "Saves electricity for your town.",
             "environmental_cons": "Trash can end up on our beaches.",
             "action_steps": ["Rinse it clean at home first.",
                              "Take it to the right local bin."]}
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
    assert context["country_name"] == "Malaysia"       # full name for grounding
    # v3.8: the LLM sees the MY-scaled number (fallback grid 0.585, ratio
    # 2.8261): recycling -1.08 x 2.8261 x 0.5 kg = -1.5261.
    assert context["items"][0]["paths"][0]["carbon_impact_kg"] == \
        pytest.approx(-1.5261, abs=1e-4)
    # Text fields replaced (incl. the 2-step action guide); numbers, ranks
    # and method ids untouched.
    top = out["items"][0]["recommendations"][0]
    assert top["encouraging_verdict"] == "Rank 1 — nice and simple!"
    assert top["environmental_pros"] == "Saves electricity for your town."
    assert top["action_steps"] == ["Rinse it clean at home first.",
                                   "Take it to the right local bin."]
    assert top["method"] == "recycling"
    assert top["carbon_impact_kg"] == pytest.approx(-1.5261, abs=1e-4)
    RecommendResponse(**out)


def test_llm_malformed_action_steps_falls_back_to_local_grid(app, monkeypatch):
    # The model returns only ONE step instead of two → the whole enrichment
    # is rejected atomically and the local neutral steps survive intact.
    monkeypatch.setattr(rs.time, "sleep", lambda _s: None)

    def one_step_post(url, json=None, headers=None, timeout=None):
        ctx = jsonlib.loads(json["messages"][1]["content"])
        return _llm_http_response(200, jsonlib.dumps({"items": [
            {"index": item["index"], "paths": [
                {"method": p["method"],
                 "encouraging_verdict": "Rank one, a really good clean pick!",
                 "environmental_pros": "Saves lots of power for the town.",
                 "environmental_cons": "Dirty items get thrown away instead.",
                 "action_steps": ["Only one step here — malformed."]}
                for p in item["paths"]]}
            for item in ctx["items"]]}))

    monkeypatch.setattr(requests, "post", one_step_post)
    with app.app_context():
        app.config["LLM_API_KEY"] = "k"
        out = rs.recommend_for_items([{"material": "plastic", "weight_kg": 0.5}])

    assert out["provider"] == "local_fallback"
    top = out["items"][0]["recommendations"][0]
    assert top["action_steps"] == rs.EXPERT_ACTION_STEPS["plastic"]["recycling"]
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
    assert rs._llm_cooling_down()                       # breaker opens on 429
    landfill = next(p for p in out["items"][0]["recommendations"]
                    if p["method"] == "landfill")
    assert landfill["environmental_cons"] == \
        rs.EXPERT_KNOWLEDGE["paper"]["landfill"]["cons"]   # grid copy served
    RecommendResponse(**out)


def test_llm_429_honors_retry_after_header(app, monkeypatch):
    # Groq sends Retry-After on 429 — the retry must pace to the server's
    # hint (not the blind default backoff), then succeed inside the window.
    sleeps = []
    monkeypatch.setattr(rs.time, "sleep", lambda s: sleeps.append(s))
    calls = []

    def flaky_post(url, json=None, headers=None, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            return _llm_http_response(429, headers={"retry-after": "3"})
        return _llm_http_response(200, _valid_generation_from(json))

    monkeypatch.setattr(requests, "post", flaky_post)

    with app.app_context():
        app.config["LLM_API_KEY"] = "k"
        out = rs.recommend_for_items([{"material": "glass", "weight_kg": 1.0}])

    assert out["provider"] == "llm_enriched"
    assert len(calls) == 2
    assert sleeps == [3.0]                      # the server's hint, verbatim


def test_llm_429_with_long_retry_after_degrades_instantly(app, monkeypatch):
    # A Retry-After far beyond the cap = the quota window is minutes away.
    # Burning two more doomed attempts helps nobody: ONE call, immediate
    # local fallback, and the cooldown breaker opens.
    monkeypatch.setattr(rs.time, "sleep", lambda _s: None)
    calls = []

    def rate_limited(*a, **k):
        calls.append(1)
        return _llm_http_response(429, headers={"retry-after": "60"})

    monkeypatch.setattr(requests, "post", rate_limited)

    with app.app_context():
        app.config["LLM_API_KEY"] = "k"
        out = rs.recommend_for_items([{"material": "paper", "weight_kg": 1.0}])

    assert len(calls) == 1                      # zero doomed retries
    assert out["provider"] == "local_fallback"
    assert rs._llm_cooling_down()               # breaker open (~60 s window)


def test_llm_cooldown_breaker_blocks_all_outbound_flights(app, monkeypatch):
    # While the breaker is open, /api/recommend must serve the local grid
    # with ZERO network calls — the 429 vector is sealed, not just retried.
    calls = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(1))
    rs._llm_cooldown_until = rs.time.monotonic() + 60

    with app.app_context():
        app.config["LLM_API_KEY"] = "k"
        out = rs.recommend_for_items([{"material": "metal", "weight_kg": 1.0}])

    assert calls == []                          # not a single outbound flight
    assert out["provider"] == "local_fallback"
    # Numbers, ranking and local copy stay fully intact meanwhile.
    assert len(out["items"][0]["recommendations"]) == 3
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
            # action_steps grid: 7x3 coverage, exactly 2 short child-simple steps.
            steps = rs.EXPERT_ACTION_STEPS[material][method]
            assert len(steps) == 2, (material, method)
            for step in steps:
                assert step.strip() and len(step.split()) <= 22, (material, method)
        for path in rs.simulate_disposal_paths(material, 1.0):
            assert len(path["encouraging_verdict"].split()) <= 28, \
                (material, path["method"])
        # The restricted-path policy verdicts obey the same register.
        for path in rs.simulate_disposal_paths(material, 1.0, country="SG"):
            assert len(path["encouraging_verdict"].split()) <= 28, \
                (material, path["method"], "SG")


def test_llm_context_carries_full_country_name_and_prompt_directive():
    # The system prompt must ship the geopolitical grounding directive, and
    # the context must translate the ISO code into the full country name the
    # directive references (so the model localizes to the right nation).
    assert "CRITICAL DIRECTIVE" in rs.LLM_SYSTEM_PROMPT
    assert "action_steps" in rs.LLM_SYSTEM_PROMPT
    results = rs.recommend_for_items(
        [{"material": "plastic", "weight_kg": 0.5}], country="DE")
    # recommend_for_items resolves numbers locally; build the context directly.
    ctx = rs._llm_context(results["items"], "DE")
    assert ctx["country"] == "DE"                 # raw code contract preserved
    assert ctx["country_name"] == "Germany"       # full name for the directive
    assert rs._country_display(None) == "global average"
    assert rs._country_display("my") == "Malaysia"
