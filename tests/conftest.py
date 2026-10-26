"""Fixtures: an in-memory Redis and settings that point at it."""
from __future__ import annotations

import fakeredis
import pytest

from inference_platform.settings import Settings


@pytest.fixture
def redis_client():
    server = fakeredis.FakeServer()
    return fakeredis.FakeRedis(server=server, version=(7, 2), decode_responses=False)


@pytest.fixture
def settings(tmp_path):
    return Settings(
        redis_url="redis://unused",
        slide_roots=(str(tmp_path),),
        task_lease_seconds=2,
        task_max_attempts=2,
        max_active_jobs_per_owner=2,
        max_regions_per_job=10,
        model_name="stub",
        progress_min_interval=0.0,
        worker_warmup=False,
        worker_consume_block_ms=10,
        worker_reclaim_interval=0.0,
        gateway_stats_interval=0.0,
        model_catalog_path=str(tmp_path / "catalog.json"),
    )


@pytest.fixture
def catalog_file(tmp_path):
    path = tmp_path / "catalog.json"
    path.write_text(
        '{"models": {"stub": {"enabled": true, "pool": "general"},'
        ' "pdac": {"enabled": true, "pool": "heavy", "scale_to_zero": true},'
        ' "disabled_one": {"enabled": false}}}'
    )
    return path
