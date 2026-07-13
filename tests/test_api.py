"""Smoke tests for the API layer."""


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    # The Step-7 SPA contract: canvas + all three JSON endpoints wired + GSAP
    # + the client-side confidence filter pill.
    for marker in (b"detection-canvas", b"/api/predict",
                   b"/api/calculate-impact", b"/api/recommend", b"gsap",
                   b"conf-slider"):
        assert marker in res.data, marker


def test_carbon_lab_page(client):
    res = client.get("/carbon-lab")
    assert res.status_code == 200
    assert b"/api/recommend" in res.data
