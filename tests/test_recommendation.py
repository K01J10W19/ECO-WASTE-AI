"""
Tests for Module 3 — the Decision Making Module (recommendation_service) and
POST /api/recommend.

The DMM is deliberately local + deterministic (no network, no app context, no
API keys), so these tests exercise the real code paths end-to-end: taxonomy
branching, the parallel 3-path simulation, the ascending-CO2e sorting engine,
the expert knowledge base coverage, weight resolution (user weight vs the
box-area pixel proxy) and the endpoint contract.
"""
import pytest

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
    ]
    for payload in cases:
        res = client.post("/api/recommend", json=payload)
        assert res.status_code == 400, payload

    res = client.post("/api/recommend",
                      json={"items": [{"material": "vibranium", "weight_kg": 1}]})
    assert res.status_code == 400   # unknown material from the service layer
