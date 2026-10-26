#!/usr/bin/env python3
"""Scaffold everything a new region-processing model needs to run as a worker.

    python scripts/new_model.py my_model --class MyModel --pool general --min 1 --max 4

Creates / updates:
  models/<name>_model.py                      from worker/templates/model_template.py
  worker/requirements/<name>.txt              pip deps for this model only
  docker/Dockerfile                           a worker-<name> stage
  docker/build.sh                             adds worker-<name> to TARGETS
  deploy/k8s/base/workers/worker-<name>.yaml  Deployment + ScaledObject
  deploy/k8s/base/kustomization.yaml          lists the new file
  config.json, deploy/k8s/base/config.container.json
                                              models.<name> sections (weights_path)

Then: put the weights where config.json says, implement the TODOs in the
model file, `make test`, `docker/build.sh`, `kubectl apply -k ...`.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def render(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace("{{" + key + "}}", value)
    return template


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="snake_case model id, e.g. tumor_grader")
    parser.add_argument("--class", dest="cls", help="Python class name (default from name)")
    parser.add_argument("--pool", default="general")
    parser.add_argument("--min", type=int, default=1, help="min replicas (0 = scale to zero)")
    parser.add_argument("--max", type=int, default=4)
    parser.add_argument("--weights", default=None, help="weights path for config.json (default models/<name>/weights.pth)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing model file")
    args = parser.parse_args()

    name = args.name
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        print("name must be snake_case: lowercase letters, digits, underscores", file=sys.stderr)
        return 2
    dash = name.replace("_", "-")
    cls = args.cls or "".join(part.capitalize() for part in name.split("_")) + "Model"
    title = name.replace("_", " ").title()
    values = dict(name=name, dash=dash, Class=cls, Title=title, min=str(args.min), max=str(args.max))
    created: list[str] = []

    # 1. model file
    model_path = ROOT / "models" / f"{name}_model.py"
    if model_path.exists() and not args.force:
        print(f"{model_path} exists (use --force to overwrite); leaving it alone")
    else:
        model_path.write_text(render((ROOT / "worker/templates/model_template.py").read_text(), **values))
        created.append(str(model_path))

    # 2. requirements
    req = ROOT / "worker/requirements" / f"{name}.txt"
    if not req.exists():
        req.write_text(f"# {name} worker: dependencies beyond worker/requirements/common.txt\n")
        created.append(str(req))

    # 3. Dockerfile stage
    dockerfile = ROOT / "docker/Dockerfile"
    text = dockerfile.read_text()
    if f"AS worker-{dash}" not in text:
        dockerfile.write_text(text.rstrip("\n") + "\n" + render((ROOT / "worker/templates/Dockerfile.stage").read_text(), **values))
        created.append(f"{dockerfile} (+stage worker-{dash})")

    # 4. build.sh targets
    build = ROOT / "docker/build.sh"
    text = build.read_text()
    match = re.search(r'TARGETS="\$\{TARGETS:-([^}]*)\}"', text)
    if match and f"worker-{dash}" not in match.group(1):
        new = match.group(0).replace(match.group(1), match.group(1) + f" worker-{dash}")
        build.write_text(text.replace(match.group(0), new))
        created.append(f"{build} (+worker-{dash})")

    # 5. k8s manifest + kustomization
    manifest = ROOT / "deploy/k8s/base/workers" / f"worker-{dash}.yaml"
    if not manifest.exists():
        manifest.write_text(render((ROOT / "worker/templates/worker.k8s.yaml").read_text(), **values))
        created.append(str(manifest))
    kust = ROOT / "deploy/k8s/base/kustomization.yaml"
    text = kust.read_text()
    entry = f"  - workers/worker-{dash}.yaml\n"
    if entry not in text:
        text = text.replace("configMapGenerator:", entry + "configMapGenerator:", 1)
        kust.write_text(text)
        created.append(f"{kust} (+{entry.strip()})")

    # 6. config sections
    weights_local = args.weights or f"models/{name}/weights.pth"
    weights_container = "/models/" + weights_local.split("models/", 1)[-1] if "models/" in weights_local else weights_local
    for cfg_path, weights in ((ROOT / "config.json", weights_local),
                              (ROOT / "deploy/k8s/base/config.container.json", weights_container)):
        cfg = json.loads(cfg_path.read_text())
        models = cfg.setdefault("models", {})
        if name not in models:
            models[name] = {
                "enabled": True,
                "pool": args.pool,
                "scale_to_zero": args.min == 0,
                "weights_path": weights,
                "device": "auto",
            }
            cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
            created.append(f"{cfg_path} (+models.{name})")

    print("Created / updated:")
    for item in created:
        print("  ", item)
    print(f"""
Next steps
  1. implement the TODOs in {model_path.relative_to(ROOT)}
  2. list extra pip deps in {req.relative_to(ROOT)}
  3. put the weights at {weights_local} (mounted as {weights_container} in containers)
  4. make test            # the stub-based suite still passes; add your own in tests/
  5. MODEL_NAME={name} python -m worker.main --describe   # publishes the panel schema
  6. docker/build.sh && kubectl apply -k deploy/k8s/overlays/<overlay>
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
