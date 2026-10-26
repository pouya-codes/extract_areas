import pytest
from fastapi.testclient import TestClient

from gateway.app import create_app
from inference_platform.contracts import TaskStatus
from inference_platform.registry import ModelRegistry
from inference_platform.settings import Settings
from tests.helpers import submission


@pytest.fixture
def client(redis_client, settings, catalog_file):
    app = create_app(settings, redis_client)
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").status_code == 200
    assert b"inference_jobs_submitted_total" in client.get("/metrics").content


def test_models_merge_catalog_and_registry(client, redis_client, settings):
    registry = ModelRegistry(redis_client, settings)
    registry.publish({"name": "stub", "version": "0.1.0", "hyperparameters": {"steps": {"type": "int"}}})
    registry.heartbeat("stub", "pod-a", {"busy": True})
    registry.heartbeat("stub", "pod-b", {"busy": False})
    registry.publish({"name": "orphan", "version": "9"})  # worker exists, not in catalog

    models = {m["name"]: m for m in client.get("/v1/models").json()["models"]}
    assert set(models) == {"stub", "pdac", "orphan"}
    assert models["stub"]["replicas"] == 2 and models["stub"]["replicas_busy"] == 1
    assert models["stub"]["hyperparameters"] == {"steps": {"type": "int"}}
    assert models["pdac"]["replicas"] == 0 and models["pdac"]["available"] is True
    assert models["pdac"]["scale_to_zero"] is True and models["pdac"]["pool"] == "heavy"
    assert models["orphan"]["in_catalog"] is False
    assert "disabled_one" not in models
    assert client.get("/v1/models/nope").status_code == 404


def test_submit_poll_cancel_overlay(client, redis_client, settings):
    r = client.post("/v1/jobs", json=submission())
    assert r.status_code == 202, r.text
    job = r.json()["job"]
    assert job["status"] == "queued" and job["regions_total"] == 2
    job_id = job["job_id"]

    # One stream entry per region, visible as queue lag.
    queues = client.get("/v1/queues").json()["queues"]
    assert queues["stub"]["depth"] == 2

    assert client.get(f"/v1/jobs/{job_id}").json()["job"]["job_id"] == job_id
    assert client.get("/v1/jobs").json()["jobs"][0]["job_id"] == job_id

    # Overlay before completion -> 409; after -> 200.
    assert client.get(f"/v1/jobs/{job_id}/regions/tumor:1/overlay").status_code == 409
    state = client.app.state.gateway
    state.store.update_task(job_id, "tumor:1", status=TaskStatus.DONE, has_overlay=True)
    state.store.put_overlay(job_id, "tumor:1", b"\x89PNGdata")
    ov = client.get(f"/v1/jobs/{job_id}/regions/tumor:1/overlay")
    assert ov.status_code == 200 and ov.headers["content-type"] == "image/png"

    c = client.post(f"/v1/jobs/{job_id}/cancel")
    assert c.status_code == 200 and c.json()["job"]["cancel_requested"] is True
    regions = {x["key"]: x for x in c.json()["job"]["regions"]}
    assert regions["tumor:2"]["status"] == "cancelled"
    assert client.post("/v1/jobs/missing/cancel").status_code == 404


def test_validation_and_limits(client):
    assert client.post("/v1/jobs", json=submission(model_name="unknown")).status_code == 404
    bad = submission()
    bad["regions"][0]["roi_bbox"] = [0, 0, 0, 10]
    assert client.post("/v1/jobs", json=bad).status_code == 422
    dup = submission()
    dup["regions"][1]["key"] = dup["regions"][0]["key"]
    assert client.post("/v1/jobs", json=dup).status_code == 422
    many = submission(regions=[{"key": f"k{i}", "roi_bbox": [0, 0, 1, 1]} for i in range(11)])
    assert client.post("/v1/jobs", json=many).status_code == 413


def test_owner_quota(client):
    assert client.post("/v1/jobs", json=submission()).status_code == 202
    assert client.post("/v1/jobs", json=submission()).status_code == 202
    r = client.post("/v1/jobs", json=submission())
    assert r.status_code == 429
    # A different owner is not affected.
    assert client.post("/v1/jobs", json=submission(owner={"user_id": 8})).status_code == 202


def test_api_token(redis_client, settings, catalog_file):
    secured = Settings(**{**settings.__dict__, "api_token": "s3cret"})
    with TestClient(create_app(secured, redis_client)) as c:
        assert c.get("/healthz").status_code == 200
        assert c.get("/v1/models").status_code == 401
        assert c.get("/v1/models", headers={"Authorization": "Bearer s3cret"}).status_code == 200
        assert c.get("/v1/models", headers={"X-Api-Token": "s3cret"}).status_code == 200
