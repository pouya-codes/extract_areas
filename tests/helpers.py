"""Shared test payload builders."""
from __future__ import annotations


def submission(**overrides):
    base = {
        "model_name": "stub",
        "slide_path": "/slides/a.svs",
        "regions": [
            {"key": "tumor:1", "region_id": 1, "label_slug": "tumor",
             "roi_bbox": [0, 0, 100, 80], "polygon": [[0, 0], [100, 0], [100, 80], [0, 80]]},
            {"key": "tumor:2", "region_id": 2, "label_slug": "tumor",
             "roi_bbox": [10, 10, 50, 50], "polygon": [[10, 10], [60, 10], [60, 60]]},
        ],
        "downsample": 1.0,
        "overlay_downsample": 2.0,
        "hyperparameters": {"steps": 2},
        "image_id": 42,
        "owner": {"user_id": 7, "username": "alice"},
    }
    base.update(overrides)
    return base
