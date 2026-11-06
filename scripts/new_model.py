#!/usr/bin/env python3
"""Scaffold platform-side pieces for a new region-processing model worker.

    python scripts/new_model.py my_model --class MyModel --pool general --min 0 --max 4

Preferred day-2 path (multi-repo):
  1. Create a repo from aim-inference-worker-template (ADO or GitHub), or run
     scripts/seed_model_repos.sh and copy the template tree.
  2. Implement models/<name>_model.py + requirements.txt in that repo.
  3. Run this script in extract_areas to add the k8s Deployment / config only.
  4. Register Azure Pipeline worker-<name> (yml already in the template) and
     add Image Updater annotations on the Argo CD Application.

This script still also writes a local monorepo Dockerfile stage + model file so
`docker/build.sh` keeps working during the transition; canonical image builds
come from the per-model repo CI (`FROM …/worker-base:main`).

Creates / updates:
  models/<name>_model.py                      from worker/templates/model_template.py
  worker/requirements/<name>.txt              pip deps for this model only
  docker/Dockerfile                           a worker-<name> stage (local builds)
  docker/build.sh                             adds worker-<name> to TARGETS
  deploy/k8s/base/workers/worker-<name>.yaml  Deployment + ScaledObject
  deploy/k8s/base/kustomization.yaml          lists the new file
  config.json, deploy/k8s/base/config.container.json
                                              models.<name> sections (weights_path)
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
Next steps (multi-repo)
  1. Seed / clone aim-inference-worker-template → aim-inference-worker-{dash}
     (scripts/seed_model_repos.sh or GitHub \"Use this template\")
  2. Copy {model_path.relative_to(ROOT)} + {req.relative_to(ROOT)} into that repo;
     implement TODOs; keep Dockerfile FROM worker-base:main
  3. Put weights at {weights_local} on the k3s host (mounted as {weights_container})
  4. Create Azure Pipeline worker-{dash} from the model repo azure-pipelines.yml
  5. Add Image Updater image-list entry for aim-inference-worker-{dash}
  6. Push model repo main → ACR; Argo rolls the Deployment

Local monorepo fallback
  MODEL_NAME={name} python -m worker.main --describe
  docker/build.sh && kubectl apply -k deploy/k8s/overlays/<overlay>
""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
