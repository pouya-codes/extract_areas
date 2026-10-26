"""The new-model template must work as shipped: render it, load it through
the worker's generic loader, run a region through the worker loop."""
import importlib.util
import sys
from pathlib import Path

from PIL import Image

from inference_platform.broker import TaskBroker
from inference_platform.contracts import JobSubmission
from inference_platform.store import JobStore
from slide_region import polygon_mask
from tests.helpers import submission
from tests.test_worker import fake_read_region, fake_resolve
from worker.core import InferenceCore
from worker.main import Worker

ROOT = Path(__file__).resolve().parent.parent


def render_template(tmp_path, name="demo_model"):
    text = (ROOT / "worker/templates/model_template.py").read_text()
    for key, value in {"name": name, "dash": name.replace("_", "-"), "Class": "DemoModel", "Title": "Demo"}.items():
        text = text.replace("{{" + key + "}}", value)
    assert "{{" not in text, "unrendered placeholder in template"
    path = tmp_path / f"{name}_model.py"
    path.write_text(text)
    spec = importlib.util.spec_from_file_location(f"models.{name}_model", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_template_model_runs_directly(tmp_path):
    module = render_template(tmp_path)
    model = module.build_model({"weights_path": "x.pth", "pool": "general", "enabled": True}, "cpu")
    assert model.model_name == "demo_model" and model._is_initialized
    assert set(model.get_hyperparameters_schema()) == {"patch_size", "batch_size", "threshold"}
    image = Image.new("RGB", (1024, 512), (30, 30, 30))  # dark -> "positive"
    mask = Image.new("L", image.size, 255)
    result = model.process(image, mask, None, {"patch_size": 256})
    assert result["success"] and result["scores"]["tiles"] == 8 and result["scores"]["positive_tiles"] == 8
    assert result["processed_image"].size == image.size


def test_template_model_through_worker(redis_client, settings, tmp_path, monkeypatch):
    module = render_template(tmp_path)
    monkeypatch.setattr(settings.__class__, "model_name", "demo_model", raising=False)
    s = settings.__class__(**{**settings.__dict__, "model_name": "demo_model"})
    store, broker = JobStore(redis_client, s), TaskBroker(redis_client, s)
    job = store.create_job(JobSubmission(**submission(model_name="demo_model", hyperparameters={"patch_size": 32})))
    broker.enqueue_many("demo_model", job["job_id"], job["task_keys"])

    model = module.build_model({"weights_path": "x.pth"}, "cpu")
    core = InferenceCore(s, read_region=fake_read_region, resolve_slide_path=fake_resolve, polygon_mask=polygon_mask)
    Worker(s, redis_client=redis_client, model=model, core=core, consumer="tpl", serve_http=False).run(max_tasks=2)

    view = store.job_view(job["job_id"])
    assert view["status"] == "succeeded", view
    assert all(r["status"] == "done" and r["has_overlay"] for r in view["regions"])
    assert view["regions"][0]["patch_total"] > 0 and view["regions"][0]["progress_percent"] == 100


def test_generic_loader_finds_build_model(tmp_path, monkeypatch, settings):
    """worker.loaders falls back to models.<name>_model.build_model."""
    render_template(tmp_path)  # registers models.demo_model_model in sys.modules
    from worker import loaders

    s = settings.__class__(**{**settings.__dict__, "model_name": "demo_model", "device": "cpu"})
    model = loaders.build_model("demo_model", s, raw_config={"models": {"demo_model": {"weights_path": "w.pth"}}})
    assert model.model_name == "demo_model"
    try:
        loaders.build_model("no_such_model", s, raw_config={"models": {}})
    except ValueError as exc:
        assert "no_such_model" in str(exc)
    else:
        raise AssertionError("unknown model should raise")
