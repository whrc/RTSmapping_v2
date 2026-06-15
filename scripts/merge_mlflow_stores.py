"""Merge per-run MLflow file stores into one combined store for reporting.

The GPU pool gives each run its own file store (`/outputs/<ver>/mlflow/<name>`)
because the file backend is not concurrency-safe (experiments.md §2.1). `build_report`
needs a single tracking URI, so this consolidates every run under one experiment.

Copies each run dir from the `rts-segmentation-v2` experiment of every per-run store
into `<dst>/<UNIFIED_EXP_ID>/<run_id>`, rewriting `experiment_id` + `artifact_uri`
in the run's meta.yaml. Idempotent (skips run_ids already present).

Usage: python scripts/merge_mlflow_stores.py [--src /outputs/v1.0/mlflow] [--dst /outputs/v1.0/mlflow_combined] [--exp-name rts-segmentation-v2]
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

UNIFIED = "1"


def _read_meta(p: Path) -> dict:
    out = {}
    for line in p.read_text().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default="/outputs/v1.0/mlflow", type=Path)
    ap.add_argument("--dst", default="/outputs/v1.0/mlflow_combined", type=Path)
    ap.add_argument("--exp-name", default="rts-segmentation-v2")
    args = ap.parse_args()

    exp_dir = args.dst / UNIFIED
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "meta.yaml").write_text(
        f"artifact_location: file://{exp_dir}\n"
        f"creation_time: 1700000000000\nexperiment_id: '{UNIFIED}'\n"
        f"last_update_time: 1700000000000\nlifecycle_stage: active\n"
        f"name: {args.exp_name}\n")

    copied = skipped = 0
    for store in sorted(p for p in args.src.iterdir() if p.is_dir()):
        if store.name.startswith("mlflow_combined"):
            continue
        for expd in store.iterdir():
            if not expd.is_dir() or expd.name in ("0", ".trash"):
                continue
            meta = expd / "meta.yaml"
            if not meta.exists() or _read_meta(meta).get("name") != args.exp_name:
                continue
            for run in expd.iterdir():
                if not run.is_dir() or not (run / "meta.yaml").exists():
                    continue
                dst_run = exp_dir / run.name
                if dst_run.exists():
                    skipped += 1
                    continue
                shutil.copytree(run, dst_run)
                rm = dst_run / "meta.yaml"
                lines = []
                for ln in rm.read_text().splitlines():
                    if ln.startswith("experiment_id:"):
                        ln = f"experiment_id: '{UNIFIED}'"
                    elif ln.startswith("artifact_uri:"):
                        ln = f"artifact_uri: file://{dst_run}/artifacts"
                    lines.append(ln)
                rm.write_text("\n".join(lines) + "\n")
                copied += 1
    print(f"merged: copied={copied} skipped={skipped} -> {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
