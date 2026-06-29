"""Smoke tests for the API layer."""


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
