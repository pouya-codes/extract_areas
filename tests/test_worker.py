"""End-to-end through the queue with the stub model and an in-memory slide."""
import threading
import time

from PIL import Image

from inference_platform.broker import TaskBroker
from inference_platform.contracts import JobSubmission, TaskStatus
from inference_platform.registry import ModelRegistry
from inference_platform.store import JobStore
from slide_region import SlideAccessError, polygon_mask
from tests.helpers import submission
from worker.core import InferenceCore
from worker.main import Worker
from worker.stub_model import StubModel


def fake_read_region(slide_path, bbox, downsample=1.0, max_pixels=None):
    x0, y0, w, h = (int(v) for v in bbox)
    size = (max(1, int(w / downsample)), max(1, int(h / downsample)))
    image = Image.new("RGB", size, (120, 80, 160))
    return image, {"bbox": [x0, y0, w, h], "downsample": downsample, "level": 0,
                   "level_downsample": 1.0, "size": list(size), "mpp": 0.25}


def fake_resolve(path, roots):
    if "missing" in path:
        raise SlideAccessError(f"slide file does not exist: {path}")
    return path


def make_worker(redis_client, settings, model=None, consumer="w1"):
    model = model or StubModel()
    model.initialize({})
    core = InferenceCore(settings, read_region=fake_read_region, resolve_slide_path=fake_resolve, polygon_mask=polygon_mask)
    return Worker(settings, redis_client=redis_client, model=model, core=core, consumer=consumer, serve_http=False)


def test_worker_processes_job(redis_client, settings):
    store = JobStore(redis_client, settings)
    broker = TaskBroker(redis_client, settings)
    job = store.create_job(JobSubmission(**submission()))
    broker.enqueue_many("stub", job["job_id"], job["task_keys"])

    worker = make_worker(redis_client, settings)
    assert worker.run(max_tasks=2) == 2

    view = store.job_view(job["job_id"])
    assert view["status"] == "succeeded", view
    for region in view["regions"]:
        assert region["status"] == TaskStatus.DONE
        assert region["has_overlay"] and region["overlay_width"] > 0
        assert region["progress_percent"] == 100 and region["patch_total"] == 2
        assert region["timing"]["model_seconds"] >= 0
        assert region["worker"] == "w1" and region["attempt"] == 1
    overlay = store.get_overlay(job["job_id"], "tumor:1")
    assert overlay and overlay.startswith(b"\x89PNG")
    # overlay_downsample=2 -> 100x80 region becomes 50x40
    assert view["regions"][0]["overlay_width"] == 50 and view["regions"][0]["overlay_height"] == 40
    # queue drained and acked
    assert broker.queue_stats("stub") == {"depth": 0, "pending": 0, "lag": 0}
    # descriptor published, replica deregistered on exit
    registry = ModelRegistry(redis_client, settings)
    assert registry.get("stub")["hyperparameters"]["steps"]["type"] == "int"
    assert registry.replicas("stub") == []


def test_worker_reports_slide_errors_as_failed(redis_client, settings):
    store = JobStore(redis_client, settings)
    broker = TaskBroker(redis_client, settings)
    job = store.create_job(JobSubmission(**submission(slide_path="/slides/missing.svs")))
    broker.enqueue_many("stub", job["job_id"], job["task_keys"])
    make_worker(redis_client, settings).run(max_tasks=2)
    view = store.job_view(job["job_id"])
    assert view["status"] == "failed"
    assert "does not exist" in view["error"]


def test_worker_skips_cancelled_and_duplicate(redis_client, settings):
    store = JobStore(redis_client, settings)
    broker = TaskBroker(redis_client, settings)
    job = store.create_job(JobSubmission(**submission()))
    broker.enqueue_many("stub", job["job_id"], job["task_keys"])
    store.request_cancel(job["job_id"])
    make_worker(redis_client, settings).run(max_tasks=2)
    view = store.job_view(job["job_id"])
    assert view["status"] == "cancelled"
    assert all(r["status"] == "cancelled" for r in view["regions"])


def test_worker_cancels_mid_inference(redis_client, settings):
    store = JobStore(redis_client, settings)
    broker = TaskBroker(redis_client, settings)
    job = store.create_job(JobSubmission(**submission(hyperparameters={"steps": 30, "step_seconds": 0.1})))
    broker.enqueue("stub", job["job_id"], "tumor:1")
    worker = make_worker(redis_client, settings)

    def cancel_soon():
        time.sleep(0.5)
        store.request_cancel(job["job_id"])

    threading.Thread(target=cancel_soon).start()
    started = time.monotonic()
    worker.run(max_tasks=1)
    assert time.monotonic() - started < 2.5  # did not run all 30 steps
    task = store.get_task(job["job_id"], "tumor:1")
    assert task["status"] == TaskStatus.CANCELLED


class CrashingModel(StubModel):
    def process(self, *args, **kwargs):
        raise RuntimeError("cuda exploded")


def test_model_exception_fails_task(redis_client, settings):
    store = JobStore(redis_client, settings)
    broker = TaskBroker(redis_client, settings)
    job = store.create_job(JobSubmission(**submission()))
    broker.enqueue("stub", job["job_id"], "tumor:1")
    make_worker(redis_client, settings, model=CrashingModel()).run(max_tasks=1)
    task = store.get_task(job["job_id"], "tumor:1")
    assert task["status"] == TaskStatus.FAILED and "cuda exploded" in task["error"]


def test_orphaned_task_is_reclaimed_then_given_up(redis_client, settings):
    """A worker that dies mid-task: another worker retries once, then fails it."""
    store = JobStore(redis_client, settings)
    broker = TaskBroker(redis_client, settings)
    job = store.create_job(JobSubmission(**submission()))
    broker.enqueue("stub", job["job_id"], "tumor:1")

    # Simulate a crash: take the message, mark running with attempt=1, never ack.
    broker.consume("stub", "dead", block_ms=10)
    store.update_task(job["job_id"], "tumor:1", status=TaskStatus.RUNNING, attempt=1, worker="dead")
    time.sleep(settings.task_lease_seconds + 0.2)

    # Second worker reclaims it (attempt 2 == max) and processes it successfully.
    worker = make_worker(redis_client, settings, consumer="w2")
    assert worker.run(max_tasks=1) == 1
    task = store.get_task(job["job_id"], "tumor:1")
    assert task["status"] == TaskStatus.DONE and task["attempt"] == 2 and task["worker"] == "w2"

    # Simulate a second crash on a different task past the attempt limit.
    broker.enqueue("stub", job["job_id"], "tumor:2")
    broker.consume("stub", "dead", block_ms=10)
    store.update_task(job["job_id"], "tumor:2", status=TaskStatus.RUNNING, attempt=2, worker="dead")
    time.sleep(settings.task_lease_seconds + 0.2)
    make_worker(redis_client, settings, consumer="w3").run(max_tasks=1)
    task = store.get_task(job["job_id"], "tumor:2")
    assert task["status"] == TaskStatus.FAILED and "stopped unexpectedly" in task["error"]
    assert broker.queue_stats("stub")["pending"] == 0


def test_slide_path_map(redis_client, settings):
    from inference_platform.settings import Settings, _parse_path_map, map_slide_path

    pm = _parse_path_map("/data=/x;/data/omero/data=/slides")
    assert pm == (("/data/omero/data", "/slides"), ("/data", "/x"))   # longest prefix first
    assert map_slide_path("/data/omero/data/a/b.svs", pm) == "/slides/a/b.svs"
    assert map_slide_path("/data/other.svs", pm) == "/x/other.svs"
    assert map_slide_path("/elsewhere.svs", pm) == "/elsewhere.svs"
    assert Settings.from_env({"SLIDE_PATH_MAP": ""}).slide_path_map == ()

    seen = {}

    def spy_resolve(path, roots):
        seen["path"] = path
        return path

    s = Settings(**{**settings.__dict__, "slide_path_map": pm})
    core = InferenceCore(s, read_region=fake_read_region, resolve_slide_path=spy_resolve, polygon_mask=polygon_mask)
    model = StubModel(); model.initialize({})
    out = core.run(model, {"slide_path": "/data/omero/data/s.svs", "roi_bbox": [0, 0, 10, 10], "polygon": []})
    assert out["status"] == "success" and seen["path"] == "/slides/s.svs"
