import time

from inference_platform.broker import TaskBroker
from inference_platform.contracts import JobStatus, JobSubmission, TaskStatus
from inference_platform.store import JobStore
from tests.helpers import submission


def test_create_job_and_view(redis_client, settings):
    store = JobStore(redis_client, settings)
    job = store.create_job(JobSubmission(**submission()))
    view = store.job_view(job["job_id"])
    assert view["status"] == JobStatus.QUEUED
    assert view["regions_total"] == 2
    assert [r["key"] for r in view["regions"]] == ["tumor:1", "tumor:2"]
    assert view["regions"][0]["status"] == TaskStatus.PENDING
    assert view["user_id"] == 7 and view["username"] == "alice"


def test_view_derives_status_from_tasks(redis_client, settings):
    store = JobStore(redis_client, settings)
    job_id = store.create_job(JobSubmission(**submission()))["job_id"]

    store.update_task(job_id, "tumor:1", status=TaskStatus.RUNNING)
    assert store.job_view(job_id)["status"] == JobStatus.RUNNING

    store.update_task(job_id, "tumor:1", status=TaskStatus.DONE, str_result="ok")
    store.update_task(job_id, "tumor:2", status=TaskStatus.FAILED, error="boom")
    view = store.job_view(job_id)
    assert view["status"] == JobStatus.SUCCEEDED  # partial success is success
    assert view["regions_done"] == 1 and view["regions_failed"] == 1
    assert view["error"] == "boom"
    assert view["completed_at"] is not None

    # all failed -> failed
    job_id2 = store.create_job(JobSubmission(**submission()))["job_id"]
    for key in ("tumor:1", "tumor:2"):
        store.update_task(job_id2, key, status=TaskStatus.FAILED, error="x")
    assert store.job_view(job_id2)["status"] == JobStatus.FAILED


def test_cancel_marks_pending_tasks(redis_client, settings):
    store = JobStore(redis_client, settings)
    job_id = store.create_job(JobSubmission(**submission()))["job_id"]
    store.update_task(job_id, "tumor:1", status=TaskStatus.RUNNING)
    store.request_cancel(job_id)
    assert store.is_cancel_requested(job_id)
    tasks = {t["key"]: t for t in store.list_tasks(store.get_job(job_id))}
    assert tasks["tumor:2"]["status"] == TaskStatus.CANCELLED
    assert tasks["tumor:1"]["status"] == TaskStatus.RUNNING  # worker finishes it
    assert store.job_view(job_id)["message"] == "Cancelling..."
    store.update_task(job_id, "tumor:1", status=TaskStatus.CANCELLED)
    assert store.job_view(job_id)["status"] == JobStatus.CANCELLED


def test_owner_quota_counts_only_active(redis_client, settings):
    store = JobStore(redis_client, settings)
    a = store.create_job(JobSubmission(**submission()))["job_id"]
    store.create_job(JobSubmission(**submission()))
    assert store.active_job_count("user:7") == 2
    for key in ("tumor:1", "tumor:2"):
        store.update_task(a, key, status=TaskStatus.DONE)
    assert store.active_job_count("user:7") == 1


def test_overlay_roundtrip(redis_client, settings):
    store = JobStore(redis_client, settings)
    job_id = store.create_job(JobSubmission(**submission()))["job_id"]
    assert store.get_overlay(job_id, "tumor:1") is None
    store.put_overlay(job_id, "tumor:1", b"\x89PNG")
    assert store.get_overlay(job_id, "tumor:1") == b"\x89PNG"


def test_broker_enqueue_consume_ack(redis_client, settings):
    broker = TaskBroker(redis_client, settings)
    ids = broker.enqueue_many("stub", "job1", ["a", "b"])
    assert len(ids) == 2
    assert broker.queue_stats("stub")["depth"] == 2

    got = broker.consume("stub", "c1", block_ms=10, count=1)
    assert len(got) == 1 and got[0].job_id == "job1" and got[0].task_key == "a"
    stats = broker.queue_stats("stub")
    assert stats["pending"] == 1 and stats["lag"] == 1

    broker.ack("stub", got[0].msg_id)
    assert broker.queue_stats("stub")["depth"] == 1
    assert broker.consume("stub", "c1", block_ms=10)[0].task_key == "b"


def test_broker_reclaims_stale_entries(redis_client, settings):
    broker = TaskBroker(redis_client, settings)
    broker.enqueue("stub", "job1", "a")
    first = broker.consume("stub", "dead-worker", block_ms=10)[0]
    # Not stale yet.
    assert broker.reclaim_stale("stub", "c2", min_idle_ms=500) == []
    time.sleep(0.6)
    reclaimed = broker.reclaim_stale("stub", "c2", min_idle_ms=500)
    assert len(reclaimed) == 1
    assert reclaimed[0].msg_id == first.msg_id
    assert reclaimed[0].deliveries >= 2
    # Renewal keeps it from being reclaimed again.
    broker.renew("stub", first.msg_id, "c2")
    assert broker.reclaim_stale("stub", "c3", min_idle_ms=500) == []
