"""Smoke tests for the API layer."""


def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    assert res.get_json()["status"] == "ok"


#: Every asset the decoupled index.html binds. Kept as one list so the
#: binding test and the 404 test can never drift apart.
SPA_ASSETS = (
    "/static/css/fonts.css",
    "/static/css/style.css",
    "/static/vendor/tailwind.js",
    "/static/vendor/gsap.min.js",
    "/static/js/app.js",
)


def test_index_page(client):
    """The SPA shell: semantic markup + bindings to the decoupled assets.

    Post-decoupling the template is structure only — the behavioural half of
    the Step-7 contract (endpoints, GSAP) is asserted against app.js in
    ``test_app_script_wires_the_spa_contract``.
    """
    res = client.get("/")
    assert res.status_code == 200

    # Structural markup that stays in the template.
    for marker in (b"detection-canvas", b"conf-slider"):
        assert marker in res.data, marker

    # The template must BIND each split asset rather than inline it.
    for asset in SPA_ASSETS:
        assert asset.encode() in res.data, asset

    # Separation of concerns must not regress back into the template...
    assert b"<style" not in res.data, "inline CSS leaked back into index.html"
    # ...and the vendored runtime must not regress back to a CDN.
    for cdn in (b"cdn.tailwindcss.com", b"jsdelivr", b"unpkg"):
        assert cdn not in res.data, cdn


def test_spa_assets_are_served(client):
    """Every asset index.html binds must actually resolve — no silent 404s."""
    for asset in SPA_ASSETS:
        assert client.get(asset).status_code == 200, asset


def test_app_script_wires_the_spa_contract(client):
    """The behavioural Step-7 contract, now living in the extracted app.js:
    all three JSON endpoints wired + GSAP orchestration."""
    js = client.get("/static/js/app.js").data
    for marker in (b"/api/predict", b"/api/calculate-impact",
                   b"/api/recommend", b"gsap"):
        assert marker in js, marker


def test_carbon_lab_page(client):
    res = client.get("/carbon-lab")
    assert res.status_code == 200
    assert b"/api/recommend" in res.data
