"""The interannual-inference stage table: what runs, in what order, and how we know it finished.

Every stage wraps a command we already run by hand. Nothing here reimplements
pipeline logic — `cmd` builds an invocation of an existing script, `evidence`
reads the artifact that script produced, and `probe` reports live progress for the
two stages that outlive their launcher.

Evidence is read from the **artifact**, never scraped from the log: a log format
change would otherwise silently turn a real number into a missing one.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Stage:
    """One step of one year's pipeline.

    Attributes:
        name: stage id, used on the command line and as the state key.
        prereqs: stages that must be ``done`` before this one may start.
        cmd: builds the argv to run; None for stages we only track.
        evidence: reads the produced artifact, returning numbers for the ledger.
        probe: live progress for a detached stage, as {done, total, pct}.
        detached: the launcher exits while the work continues (GEE tasks, nohup run).
        gate: needs a human sign-off before dependents may run.
        external: someone else runs it (Heidi's acquisition); we only observe.
        note: shown by status.py to say what a human must do.
    """

    name: str
    prereqs: tuple[str, ...] = ()
    cmd: Optional[Callable[[int, dict], list[str]]] = None
    evidence: Optional[Callable[[int, dict], dict]] = None
    probe: Optional[Callable[[int, dict], Optional[dict]]] = None
    detached: bool = False
    gate: bool = False
    external: bool = False
    note: str = ""
    mounts: tuple[tuple[str, str], ...] = field(default=())


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def fmt(template: str, year: int) -> str:
    """Fill ``{year}`` in a config path template."""
    return template.format(year=year)


def _split_gs(uri: str) -> tuple[str, str]:
    bucket, _, prefix = uri[5:].partition("/")
    return bucket, prefix.rstrip("/")


def _client():
    from google.cloud import storage
    return storage.Client()


def gcs_exists(uri: str) -> bool:
    """True if a single GCS object exists."""
    bucket, key = _split_gs(uri)
    return _client().bucket(bucket).blob(key).exists()


def count_lines(path: Path) -> int:
    """Data rows in a CSV (excludes the header)."""
    with open(path, "rb") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def adc_for(year: int, cfg: dict) -> str:
    """Path to the Google credential a year's work should run under.

    Earth Engine's concurrent-task allowance is **per user** (measured 2026-08-25: a
    second account's tasks went RUNNING within 1s while the first account already held
    all 3 slots, taking the total to 5). So each year runs under its own account, and
    `accounts:` records which — that mapping is the difference between a ~64-day serial
    campaign and a ~16-day one.
    """
    accounts = cfg.get("accounts") or {}
    path = accounts.get(year) or accounts.get("default")
    if not path:
        path = "~/.config/gcloud/application_default_credentials.json"
    return str(Path(path).expanduser())


def docker(image: str, argv: list[str], cfg: dict,
           extra: tuple[tuple[str, str], ...] = (), adc: str | None = None) -> list[str]:
    """Wrap `argv` in the standard project container invocation.

    The repo mounts at /app and /mnt/outputs at /outputs, matching the paths the
    existing scripts default to (mask_tiles_to_domain.py hardcodes /app/domain).
    """
    adc = adc or str(Path.home() / ".config/gcloud/application_default_credentials.json")
    cmd = [
        "sudo", "docker", "run", "--rm", "--entrypoint", "python",
        "-v", f"{REPO}:/app", "-v", "/mnt/outputs:/outputs",
        "-v", f"{adc}:/gcp_adc.json:ro",
        "-e", "GOOGLE_APPLICATION_CREDENTIALS=/gcp_adc.json",
        "-e", f"GOOGLE_CLOUD_PROJECT={cfg['docker']['project']}",
        "-e", "PYTHONPATH=/app", "-w", "/app", "--shm-size", "16g",
    ]
    for host, dest in extra:
        cmd += ["-v", f"{host}:{dest}"]
    return cmd + [image, *argv]


# --------------------------------------------------------------------------
# acquire — Heidi runs this; we read the status file her loop writes
# --------------------------------------------------------------------------
def _acquire_evidence(year: int, cfg: dict) -> dict:
    p = Path(fmt(cfg["paths"]["acquisition_status"], year))
    if not p.is_file():
        return {}
    s = json.loads(p.read_text())
    return {"n_ordered": s.get("n_ordered"), "n_total": s.get("n_total"),
            "n_failed": s.get("n_failed"), "n_done": s.get("n_done")}


def _acquire_probe(year: int, cfg: dict) -> Optional[dict]:
    e = _acquire_evidence(year, cfg)
    if not e or not e.get("n_total"):
        return None
    return {"done": e.get("n_done") or 0, "total": e["n_total"],
            "pct": 100.0 * (e.get("n_done") or 0) / e["n_total"]}


# --------------------------------------------------------------------------
# s2_export — launches GEE tasks and exits; progress is objects landing in GCS
# --------------------------------------------------------------------------
def _s2_cmd(year: int, cfg: dict) -> list[str]:
    ee_project = (cfg["docker"].get("ee_projects") or {}).get(year)
    if not ee_project:
        raise ValueError(
            f"docker.ee_projects has no entry for {year}, so there is no Earth Engine "
            "project authorised to run this export.\n"
            "  pdg-project-406720 is ours but cannot run batch exports (noncommercial "
            "restricted mode — tasks sit PENDING indefinitely).\n"
            "  abruptthawmapping is ours and unrestricted, but needs "
            "roles/serviceusage.serviceUsageConsumer granted to the ADC identity by a "
            "project owner.\n"
            "  Do NOT substitute a pdg-wg-* project: those belong to other teams.")
    return docker(cfg["docker"]["dataprep_image"], [
        "scripts/export_s2_composites.py", "--year", str(year),
        "--domain", "domain/circumpolar_south_domain.geojson",
        "--bucket", cfg["paths"]["s2_bucket"],
        "--prefix", fmt(cfg["paths"]["s2_prefix"], year),
        # EE compute quota; distinct from GOOGLE_CLOUD_PROJECT (GCS billing), which
        # docker() sets. Conflating the two fails with "Project was not passed".
        "--project", ee_project,
    ], cfg, adc=adc_for(year, cfg))


def _s2_probe(year: int, cfg: dict) -> Optional[dict]:
    """Distinct exported cells so far, via the tested s2_snapshot reader."""
    import sys
    sys.path.insert(0, str(REPO))
    from scripts.inference_progress import s2_snapshot
    uri = f"gs://{cfg['paths']['s2_bucket']}/{fmt(cfg['paths']['s2_prefix'], year)}"
    total = cfg["expect"]["n_s2_cells"]
    snap = s2_snapshot(uri, total)
    c = snap["cells"]
    return {"done": c["done"], "total": c["total"], "pct": c["pct"],
            "eta_hours": snap.get("eta_hours")}


def _s2_evidence(year: int, cfg: dict) -> dict:
    p = _s2_probe(year, cfg) or {}
    return {"n_cells": p.get("done"), "n_cells_expected": p.get("total")}


# --------------------------------------------------------------------------
# quad_index
# --------------------------------------------------------------------------
def _quad_cmd(year: int, cfg: dict) -> list[str]:
    return docker(cfg["docker"]["train_image"], [
        "scripts/build_quad_index.py",
        "--bucket", cfg["paths"]["planet_bucket"],
        "--prefix", fmt(cfg["paths"]["planet_prefix"], year),
        "--output", f"/outputs/inference/quad_index_{year}q3.csv",
        "--expect-quads", str(cfg["expect"]["n_quads"]),
        "--tolerance", str(cfg["expect"]["quad_tolerance"]),
    ], cfg)


def _quad_evidence(year: int, cfg: dict) -> dict:
    p = Path(cfg["paths"]["local_inference"]) / f"quad_index_{year}q3.csv"
    return {"n_quads": count_lines(p)} if p.is_file() else {}


# --------------------------------------------------------------------------
# s2_index
# --------------------------------------------------------------------------
def _s2idx_cmd(year: int, cfg: dict) -> list[str]:
    return docker(cfg["docker"]["train_image"], [
        "scripts/build_s2_index.py",
        "--bucket", cfg["paths"]["s2_bucket"],
        "--prefix", fmt(cfg["paths"]["s2_prefix"], year),
        "--output", f"/outputs/inference/s2_index_{year}_south.csv",
    ], cfg)


def _s2idx_evidence(year: int, cfg: dict) -> dict:
    p = Path(cfg["paths"]["local_inference"]) / f"s2_index_{year}_south.csv"
    return {"n_s2_cells_indexed": count_lines(p)} if p.is_file() else {}


# --------------------------------------------------------------------------
# drift_check — human gate
# --------------------------------------------------------------------------
def _drift_cmd(year: int, cfg: dict) -> list[str]:
    return docker(cfg["docker"]["train_image"], [
        "scripts/check_inference_normalization.py",
        "--deployment-package", cfg["paths"]["drift_package"],
        "--quad-index", f"/outputs/inference/quad_index_{year}q3.csv",
        "--n-quads", str(cfg["expect"]["drift_sample_quads"]),
        "--output", f"/outputs/inference/drift_report_{year}q3.csv",
    ], cfg)


def _drift_evidence(year: int, cfg: dict) -> dict:
    """Worst per-channel drift vs the 2025 QUAD baseline (see README + inference.md 5.4)."""
    p = Path(cfg["paths"]["local_inference"]) / f"drift_report_{year}q3.csv"
    base = Path(cfg["paths"]["quad_baseline"])
    if not base.is_absolute():
        base = REPO / base
    if not (p.is_file() and base.is_file()):
        return {}
    import csv

    def rows(f: Path) -> dict[str, dict]:
        # The committed baseline carries a comment header explaining what it is.
        with open(f) as fh:
            data = (ln for ln in fh if not ln.startswith("#"))
            return {r["channel"]: r for r in csv.DictReader(data)}

    cur, ref = rows(p), rows(base)
    worst_mean, worst_std = 0.0, 0.0
    for ch, r in cur.items():
        if ch not in ref:
            continue
        b = ref[ch]
        sd = float(b["sample_std"])
        worst_mean = max(worst_mean, abs(float(r["sample_mean"]) - float(b["sample_mean"])) / sd)
        worst_std = max(worst_std, abs(float(r["sample_std"]) / sd - 1.0))
    return {"worst_mean_drift_sigma": round(worst_mean, 4),
            "worst_std_ratio": round(worst_std, 4),
            "baseline": cfg["paths"]["quad_baseline"]}


# --------------------------------------------------------------------------
# tile_grid
# --------------------------------------------------------------------------
def _grid_cmd(year: int, cfg: dict) -> list[str]:
    """Two scripts in one shell so the pair is a single tracked stage."""
    inner = (
        f"python scripts/generate_tile_grid.py "
        f"--quad-index /outputs/inference/quad_index_{year}q3.csv "
        f"--config configs/deployment.yaml "
        f"--output /outputs/inference/tiles_{year}q3_full.csv && "
        f"python scripts/mask_tiles_to_domain.py "
        f"/outputs/inference/tiles_{year}q3_full.csv "
        f"/outputs/inference/tiles_{year}q3_domain_full.csv"
    )
    argv = docker(cfg["docker"]["train_image"], ["-c", inner], cfg)
    argv[argv.index("--entrypoint") + 1] = "bash"
    return argv


def _grid_evidence(year: int, cfg: dict) -> dict:
    p = Path(cfg["paths"]["local_inference"]) / f"tiles_{year}q3_domain_full.csv"
    return {"n_tiles": count_lines(p)} if p.is_file() else {}


# --------------------------------------------------------------------------
# shard
# --------------------------------------------------------------------------
def _shard_cmd(year: int, cfg: dict) -> list[str]:
    return docker(cfg["docker"]["train_image"], [
        "scripts/shard_tiles.py",
        "--tile-list", f"/outputs/inference/tiles_{year}q3_domain_full.csv",
        "--output", fmt(cfg["paths"]["inference_base"], year),
        "--shard-size", str(cfg["expect"]["shard_size"]),
    ], cfg)


def _shard_evidence(year: int, cfg: dict) -> dict:
    base = fmt(cfg["paths"]["inference_base"], year)
    bucket, prefix = _split_gs(base)
    blob = _client().bucket(bucket).blob(f"{prefix}/shards/index.json")
    if not blob.exists():
        return {}
    idx = json.loads(blob.download_as_text())
    return {"n_shards": idx["n_shards"], "n_tiles": idx["n_tiles"]}


# --------------------------------------------------------------------------
# infer — nohup'd supervisor; progress via the tested inference_snapshot
# --------------------------------------------------------------------------
def _infer_cmd(year: int, cfg: dict) -> list[str]:
    base = fmt(cfg["paths"]["inference_base"], year)
    return ["bash", "-c",
            f"BASE={base} "
            f"QUAD_INDEX={base}/quad_index_{year}q3.csv "
            f"S2_INDEX={base}/s2_index_{year}_south.csv "
            f"PACKAGES='{cfg['paths']['packages']}/seed42 {cfg['paths']['packages']}/seed43 "
            f"{cfg['paths']['packages']}/seed44' "
            f"LOGDIR=/mnt/outputs/inference/{year}q3_south/logs "
            f"STOP_FILE=/mnt/outputs/inference/{year}q3_south/STOP "
            f"nohup bash {REPO}/scripts/launch_south_inference.sh "
            f"> /mnt/outputs/inference/{year}q3_south/launch.log 2>&1 &"]


def _infer_probe(year: int, cfg: dict) -> Optional[dict]:
    import sys
    sys.path.insert(0, str(REPO))
    from scripts.inference_progress import inference_snapshot
    snap = inference_snapshot(fmt(cfg["paths"]["inference_base"], year))
    t = snap["tiles"]
    return {"done": t["done"], "total": t["total"], "pct": t["pct"],
            "shards_done": snap["shards"]["done"], "shards_total": snap["shards"]["total"],
            "rate_tiles_s": snap["rate_tiles_s"], "eta_hours": snap["eta_hours"],
            "stale_workers": len(snap["stale_workers"])}


def _infer_evidence(year: int, cfg: dict) -> dict:
    p = _infer_probe(year, cfg) or {}
    return {"tiles_done": p.get("done"), "tiles_total": p.get("total"),
            "shards_done": p.get("shards_done"), "shards_total": p.get("shards_total")}


# --------------------------------------------------------------------------
# reconcile — the exactly-once check the 2025 run passed at 41,567,572
# --------------------------------------------------------------------------
def _reconcile_evidence(year: int, cfg: dict) -> dict:
    p = _infer_probe(year, cfg) or {}
    done, total = p.get("shards_done"), p.get("shards_total")
    return {"shards_done": done, "shards_total": total,
            "exact": bool(total) and done == total,
            "tiles_done": p.get("done"), "tiles_total": p.get("total")}


# --------------------------------------------------------------------------
# The ordered table
# --------------------------------------------------------------------------
ORDER: list[Stage] = [
    Stage("acquire", external=True, evidence=_acquire_evidence, probe=_acquire_probe,
          note="Heidi runs planetscope-download/run_year.sh on the acquisition VM"),
    Stage("s2_export", detached=True, cmd=_s2_cmd, evidence=_s2_evidence, probe=_s2_probe,
          note="GEE batch export; independent of Planet, so it need not wait on acquire"),
    Stage("quad_index", prereqs=("acquire",), cmd=_quad_cmd, evidence=_quad_evidence),
    Stage("s2_index", prereqs=("s2_export",), cmd=_s2idx_cmd, evidence=_s2idx_evidence),
    Stage("drift_check", prereqs=("quad_index",), gate=True, cmd=_drift_cmd,
          evidence=_drift_evidence,
          note="compare against the 2025 QUAD baseline, not the training-tile stats"),
    Stage("tile_grid", prereqs=("quad_index",), cmd=_grid_cmd, evidence=_grid_evidence),
    # shard needs only the tile list — it never reads imagery. Gating it on s2_index
    # would idle it behind the ~7-11 day S2 export for no reason; it is *inference*
    # that needs NDVI, so that is where the s2_index prerequisite belongs.
    Stage("shard", prereqs=("tile_grid",), cmd=_shard_cmd, evidence=_shard_evidence),
    Stage("infer", prereqs=("shard", "drift_check", "s2_index"), detached=True, cmd=_infer_cmd,
          evidence=_infer_evidence, probe=_infer_probe),
    Stage("reconcile", prereqs=("infer",), evidence=_reconcile_evidence,
          note="shards done must equal shards total, exactly"),
    Stage("merge", prereqs=("reconcile",), external=True,
          note="post-inference; tracked here, driven by hand (post-inference.md)"),
    Stage("vectorize", prereqs=("merge",), external=True,
          note="post-inference; vectorize_region.py --threshold 0.30 --min-area-m2 0"),
    Stage("qc", prereqs=("vectorize",), external=True, gate=True,
          note="human review of the year's map before it counts as delivered"),
]

BY_NAME: dict[str, Stage] = {s.name: s for s in ORDER}
NAMES: list[str] = [s.name for s in ORDER]


def get(name: str) -> Stage:
    """Look up a stage, with a helpful error listing the valid names."""
    if name not in BY_NAME:
        raise KeyError(f"unknown stage {name!r}; known stages: {', '.join(NAMES)}")
    return BY_NAME[name]


def unmet_prereqs(stage: Stage, st: dict) -> list[str]:
    """Prereq stages that are not yet ``done`` for this year."""
    return [p for p in stage.prereqs
            if st["stages"].get(p, {}).get("status") != "done"]
