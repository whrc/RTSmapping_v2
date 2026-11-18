# v3-cgm — a classification-guided existence gate (`cgm`)

Branch `v3-cgm` (git worktree; renamed from `v3-alpha` on 2026-11-18). **Purpose: test whether the `cgm` tile-level existence gate reduces false positives.** Nothing else belongs on this branch. **v2 is not changed by anything here.** Whether `cgm` replaces v2 for any year is decided after the experiments. (The file keeps the name `v3_alpha.md`, and the plan text outside the repo still says v3-alpha.)

**Status (2026-11-18): PAUSED after Phase 0.1/0.2.** The 2025 review verdicts are about to be used to extend the training set (more hard negatives; more positives in poorly represented regions; val and test regions held out) in a separate session. `cgm` is evaluated after that, with a new control, because the old controls, `σ_ctrl` and any probe would be tied to the old training set. See *Pickup* at the end.

## 1. Spec (from the approved plan)

**Problem.** Too many false positives (FPs) in the delivered 2025 South map: 9,636 of 22,008 decided census
verdicts are `false` (43.8%, snapshot 2026-11-17). Scope is model architecture only.

**Arm: `cgm`.** Add a tile-level existence head to the existing UNet++/EfficientNet-B5 and let its
probability gate the dense output (classification-guided module of UNet 3+, soft gate).
- Head: smp's built-in `aux_params` classification head on the deepest encoder features
  (max-pool → dropout → linear). Verified on the pinned smp 0.5.0: `UnetPlusPlus(aux_params=...)` returns
  `(mask, tile logit)` with shapes `(B,1,512,512)` and `(B,1)`; the head has 513 parameters.
- Tile label from existing labels: ≥ 50 positive pixels → 1, none → 0, 1–49 → skipped.
- Output contract unchanged: one logit map `logit(σ(dense)·σ(z))`, one temperature, one threshold.
- Why UNet++ and not UNet 3+: the gate is separable from the decoder; swapping decoders would confound
  the test; the decoder sweep never moved the gate (`training/experiments.md:478` made UNet 3+ conditional
  on it); UNet 3+ is not in smp.

**Evaluation.** E1 = v1.1 Val-Realistic (PR-AUC non-inferiority, FP objects per 1,000 tiles at matched
object recall). E2 = verified 2025 polygons in the val regions (polygon-max-probability AUROC, `rts` vs
`false`), compared against the v1.1 control scored on the same polygons.

**Build gates (pre-registered; proposed thresholds).**
- 0.1(b): ≥ 50% of census FPs have no verified `rts` in their own tile.
- 0.4 probe: a linear probe on the *frozen* control encoder beats the dense head's tile `max_prob` by
  ≥ 0.02 AUROC on E2 tiles.

**Lock** (3 seeds, 3/3 signs): E2 AUROC Δ ≥ max(0.02, 2 × paired SE), and E1 FP objects at matched recall
down by ≥ 2σ_ctrl, and E1 PR-AUC Δ ≥ −0.01.

## 2. Phase 0.1 / 0.2 — census error analysis (done 2026-11-18)

Reproduce: `python scripts/diag_fp_census.py --snapshot rts_candidates_review_status_20261117T230236Z.geojson.gz
--attrs south_rts_attributes.parquet --regions circumpolar_subregions.geojson --splits splits.yaml --out-dir ...`
(inputs: `gs://rts-arctic-usw1/inference/2025q3_south/products/` and `gs://rts-arctic-us/training/v1.0/`).
Tables are in [v3_alpha_phase0/](v3_alpha_phase0/). Status counts reproduce the documented snapshot exactly
(12,372 rts · 9,636 false · 5,792 unsure · 32,367 unreviewed; 60,167 polygons).

**The reviewed set.** The queue runs in descending `max_prob`, so the high tier (≥ 0.65) is fully reviewed,
the medium tier (0.45–0.65) is 74% reviewed, and the low tier (0.30–0.45) is untouched. Every number below
describes the reviewed part only.

### 0.1a — where the FPs are

| `max_prob` band | rts | false | decided | false rate | share of all FPs |
|---|---:|---:|---:|---:|---:|
| 0.45–0.65 | 1,336 | 5,250 | 6,586 | 79.7% | 54.5% |
| 0.65–0.80 | 2,229 | 2,393 | 4,622 | 51.8% | 24.8% |
| 0.80–0.90 | 1,777 | 1,310 | 3,087 | 42.4% | 13.6% |
| 0.90–0.95 | 1,684 | 406 | 2,090 | 19.4% | 4.2% |
| 0.95–1.00 | 5,346 | 277 | 5,623 | 4.9% | 2.9% |

- **92.9% of FPs have `max_prob` < 0.9, and 79.3% are < 0.8.** Only 683 FPs are ≥ 0.9.
- **Latitude.** False rate by |lat|: < 55° 96.6% · 55–60° 95.6% · 60–65° 72.0% · 65–70° 50.3% · ≥ 70° 35.5%.
  FPs below 65°N number 2,076 (21.5%).
- **Region.** The top 5 regions hold **58.2%** of all FPs: Taimyr–Central Siberian tundra 2,538 (42% false),
  Canadian Middle Arctic 1,063 (17%), Kalaallit Nunaat Arctic steppe (Greenland) 1,051 (98%), Canadian High
  Arctic 622 (80%), Canadian Low Arctic 330 (28%). The 13 regions with false rate ≥ 0.9 hold **26.2%** of all
  FPs and only 83 verified RTS. 607 FPs (6.3%) have a centroid outside every region polygon.
- **Area.** FPs sit at 2,000–50,000 m² (90%); the false rate is roughly flat (33–49%) across size bands.
- **NoData.** Polygons with `nodata_frac` ≥ 0.05 are **61 FPs = 0.63%** of all FPs (137 decided polygons).

### 0.1b — how many FPs a tile-level gate could remove

Tile = ±1,223 m *projected* in EPSG:3857 (512 px × 4.7773 m), centred on the polygon; that is about 420 m on
the ground at 70°N.

| bound | share of FPs | meaning |
|---|---:|---|
| lower | **74.3%** | the tile holds no other rts / unsure / unreviewed polygon (certainly alone) |
| upper | **98.8%** | the tile holds no *verified* rts (unreviewed neighbours counted as not-rts) |

By `max_prob` band the lower bound is 71.8% (0.45–0.65), 77.4%, 77.0%, 78.6% (≥ 0.9). **Gate 0.1(b) passes**
(≥ 50% required). This is co-location, not recognisability: it says most FPs are not next to a real slump,
not that a head can tell them apart.

### 0.2 — decided verdicts per split

| split | rts | false | decided |
|---|---:|---:|---:|
| train regions | 11,169 | 7,883 | 19,052 |
| val_realistic regions | 912 | 697 | 1,609 |
| test_realistic regions (one-shot only) | 247 | 448 | 695 |
| unassigned (region not in `splits.yaml`, or no region) | 44 | 608 | 652 |

The val regions are Canadian Low Arctic (831 rts / 330 false), Interior Alaska–Yukon lowland taiga (7 / 155),
Northeast Siberian coastal tundra (55 / 101), Novosibirsk Islands Arctic desert (7 / 86), Beringia lowland
tundra (12 / 25).

## 3. Corrections to the approved plan

1. **"Precision peaks at 0.65 and falls above it" does not hold at the polygon level.** The plan took it
   from `south_products.md` caveat 1 and used it to argue that the dense head's confidence is a poor
   existence signal. In the census, precision rises monotonically with `max_prob`:

   | keep `max_prob` ≥ | decided | precision | FPs kept |
   |---|---:|---:|---:|
   | 0.45 | 22,008 | 56.2% | 9,636 |
   | 0.65 | 15,422 | 71.6% | 4,386 |
   | 0.80 | 10,800 | 81.5% | 1,993 |
   | 0.90 | 7,713 | 91.1% | 683 |
   | 0.95 | 5,623 | 95.1% | 277 |

   Caveat 1 describes the *pixel-threshold* sweep of the H.2 object-precision grid (re-cutting the map at a
   higher probability fragments true slumps). That is a different quantity. **`max_prob` is already a
   decent ranker** (AUROC 0.839 over all decided polygons, 0.799 in the high tier, 0.752 in the val
   regions). The FP problem is mostly mid-confidence polygons, not confident look-alikes. The `cgm` motivation
   is therefore weaker than the plan stated: it has to add information *beyond* a score that already ranks
   well.
2. **The E2 metric needs to be within-region.** Pooled val-region AUROC mixes regions that are almost all
   FPs (taiga 7 rts / 155 false; Novosibirsk 7 / 86) with one RTS-rich region (Canadian Low Arctic
   831 / 330). A tile head can gain pooled AUROC just by learning that taiga scenes host no RTS, a regional
   prior, which is not the look-alike rejection `cgm` is for. The primary E2 metric should be the AUROC
   **within Canadian Low Arctic** (baseline `max_prob` AUROC 0.772, n = 1,161), plus a region-stratified
   pooled number. Power is limited: one region.
3. **`s1` (NoData) is now measured, not assumed.** The build gate needed ≥ 10% of FPs to have
   `nodata_frac` ≥ 0.05; the census gives 0.63%. `s1` stays dropped.
4. **Tile neighbourhood.** The plan said "within 1.2 km". The tile is a projected square, so the test uses
   ±1,223 m in EPSG:3857, which is much smaller on the ground at high latitude.

## 4. Environment notes (this laptop)

`.venv` in the worktree (gitignored) built from `requirements_frozen.txt` (smp 0.5.0, timm 1.0.26,
torch 2.11.0+cpu) plus `requirements-v3a.txt` (geopandas, shapely, pyogrio). Two local deviations:
`rasterio==1.4.4` instead of 1.5.0 (1.5 needs Python ≥ 3.12; this machine has 3.11.5), and a venv-local
`sitecustomize.py` that preloads scikit-learn's `msvcp140.dll`, because the system copy is older than
torch 2.11 needs (`import torch` otherwise dies with WinError 1114). Run with `PYTHONNOUSERSITE=1`; the
user-site NumPy 2.4.6 there is built against a mismatched scipy.

**Baseline of the existing suite on this laptop (before any model change).** About 590 tests pass and 36
fail or error, in six files: `test_review_crops.py` (19), `test_train_smoke.py` (10), `test_downsample_max.py`
(3), `test_interannual_inference.py` (2), `test_prob_writer.py` (1), `test_planetscope_download.py` (1). The
causes are the laptop, not the code: no GDAL command-line tools (`WinError 2`), no `osgeo` Python bindings,
POSIX-path assertions that see backslashes, and mlflow file URIs on Windows. None touches this branch's files.
The run used `-q` twice, which drops pytest's summary line, so the counts come from the progress lines.
Treat the VM run as the authoritative baseline.

## 5. Pickup

**Done and committed on branch `v3-cgm`:** the census error analysis (`scripts/diag_fp_census.py`, 9 tests,
tables in `v3_alpha_phase0/`), the smp `aux_params` check, the pinned local environment, the corrections in §3.
**Not done:** Phase 0.3 (E2 tile set), 0.4 (encoder probe), 0.5 (control scoring), and everything in Phase 1–2.
No `cgm` code exists yet, and no training has run.

**Reusable by the data-extension work:** the region and split assignment (`assign_split`), the tile window
logic, the per-region counts in `v3_alpha_phase0/false_rate_by_region.csv` (verified rts and false per region,
to rank where to add data), and the rule that val and test regions stay held out.

**Decisions still open (asked 2026-11-18):** how to treat `rts` outlines as positives (model-drawn 0.30
contours: hand refine, or boundary-ignore ring); whether to cap or balance hard negatives by scene type; whether
the 652 polygons from regions in no split may be used for training.

**To resume `cgm`:**
1. Rebuild the controls on the extended set (single seeds are enough to start; the recipe is `configs/v1_1_seed42.yaml`).
2. Re-run `diag_fp_census.py` on a fresh snapshot; re-check the corrections in §3 (precision rises with `max_prob`).
3. E2 primary metric: AUROC within Canadian Low Arctic, plus region-stratified pooled; secondary: unassigned
   polygons as extra false positives, train-region polygons as a labelled memorisation reference.
4. Phase 0.3–0.5 as in §1 (VM: CPU, about 16 vCPU, us-west1), then the go/no-go on the probe (≥ +0.02 AUROC).
5. Phase 1–2 only if it passes. Phase 2 needs a training GPU quota (still none in `abruptthawmapping`).

**Environment to recreate:** `python -m venv .venv`, install `requirements_frozen.txt` (with
`--extra-index-url https://download.pytorch.org/whl/cpu`) then `requirements-v3a.txt`; see §4 for the two
laptop-only workarounds. Raw inputs live in GCS (paths in §2), not in the repo. The approved plan text is
in `C:/Users/Yili Yang/.claude/plans/plan-for-a-new-temporal-kitten.md` (outside the repo; §3 above supersedes
its claim about precision falling above 0.65).
