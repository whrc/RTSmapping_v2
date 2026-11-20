# `cgm` — a classification-guided tile existence gate

**Purpose: test whether a tile-level existence gate reduces false positives.** `cgm` is one model
parameter (architecture family E): the `model.cgm` block in a training config switches it on; without
the block the model is the plain UNet++ and nothing changes. M1.0 (deployed) is not changed by anything
here. Whether `cgm` is adopted is decided by the criteria in §6, on whatever data version it is tested.
It gets a model id (`versions.yaml`) only if it becomes a deployment candidate.

**Status (2026-11-20): BUILT, READY TO TRAIN, NOT RUN.** Code is on `main` (§5). Two ready screen
configs on D1.0: `configs/cgm_max_seed42.yaml`, `configs/cgm_avg_seed42.yaml`. Launch order in §7.
(The work was done on the branch `v3-cgm`, now merged and deleted; "v3" meant nothing under the version
scheme, so the name was dropped.)

## 1. Spec

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

**Evaluation.** E1 = the data version's Val-Realistic (PR-AUC non-inferiority, FP objects per 1,000 tiles
at matched object recall). E2 = verified 2025 polygons in the val regions (polygon-max-probability AUROC,
`rts` vs `false`; primary: within Canadian Low Arctic, see §3.2), compared against the control scored on
the same polygons. The adoption criteria are in §6.

**Build gate 0.1(b)** (passed, §2): ≥ 50% of census FPs have no verified `rts` in their own tile. The
frozen-encoder probe (0.4) was dropped on 2026-11-20 — see §6.

## 2. Phase 0.1 / 0.2 — census error analysis (done 2026-11-18)

Reproduce: `python scripts/diag_fp_census.py --snapshot rts_candidates_review_status_20261117T230236Z.geojson.gz
--attrs south_rts_attributes.parquet --regions circumpolar_subregions.geojson --splits splits.yaml --out-dir ...`
(inputs: `gs://rts-arctic-usw1/inference/2025q3_south/products/` and `gs://rts-arctic-us/training/v1.0/`).
Tables are in [cgm_phase0/](cgm_phase0/). Status counts reproduce the documented snapshot exactly
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
torch 2.11.0+cpu) plus `requirements-cgm.txt` (geopandas, shapely, pyogrio). Two local deviations:
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

## 5. Build (done 2026-11-20)

| piece | where |
|---|---|
| `gate_logits(dense, z)` — `logit(σ(d)·σ(z))` in log space, float32, NaN-free gradients at saturation | `models/segmentation.py` |
| `ExistenceGate` — wraps the smp model built with `aux_params`; head linear starts at weight 0, bias `init_bias`; keeps `last_tile_logit` for the loss; delegates `.encoder`/`.segmentation_head` | `models/segmentation.py` |
| `build_model` reads `model.cgm`; UNet++ with `fusion` `early`/`stem_init` only | `models/segmentation.py` |
| `tile_targets`, `tile_existence_loss` — tile label from the augmented mask; 0 (not NaN) when a batch keeps no tile | `losses/segmentation_losses.py` |
| loss = dense loss + `loss_weight` × tile BCE; logs `train_tile_loss`, `val_tile_loss` | `scripts/train.py` |
| user-facing spec and the config block | `training/training.md` §3.4 |
| tests | `tests/test_cgm.py` |

Inference, calibration, packaging and evaluation need no change: the output contract is the same and the
package's `model_config.yaml` carries the whole `model` block, so `predictor.load_deployment_package`
rebuilds the gate.

**Keys and defaults** (`model.cgm`): `pooling: max` · `dropout: 0.2` · `init_bias: 3.0` · `loss_weight: 0.1`
· `min_pos_px: 50`.

**Why `loss_weight` (λ) = 0.1.** The focal loss (α 0.25, γ 2, mean over valid pixels) is about
0.005–0.05 per batch. The tile BCE starts near 0.7 and should fall to about 0.1–0.3. At λ = 1 the tile term
would be 10× the dense term and could pull the shared encoder toward tile classification. λ = 0.1 keeps the
two the same size. The dense loss also trains `z` through the gate, so the BCE does not need to be large.
This is a judgement, not a measurement: if the screen passes, a λ ∈ {0.03, 0.3} check is cheap.

**Facts found while building.** At the same seed the `cgm` model's encoder init equals the plain model's,
but smp draws the head's init before the decoder's, so the decoder init differs like a seed change. The
gate can only lower a pixel's probability (σ(d)·σ(z) ≤ σ(d)); at init it lowers every logit by about
0.05 and keeps the ranking.

## 6. How to test `cgm` on any data version — and the adoption criteria

**Pair rule.** Every test is a pair: two runs identical in every key except `model.cgm` (same data
version, recipe, seed). On D1.0 the control already exists: M1.0's `aug_trivialaugment_deploy`
(`_seed43`, `_seed44`). On D4.0 or a later set, the control is that set's own baseline run, which that
training needs anyway; the arm is its config plus the `model.cgm` block (`training/training.md` §3.4).

**Pre-registered criteria** (fixed before any `cgm` result is read):

| step | runs | passes if |
|---|---|---|
| Screen | seed 42, `pooling: max` and `avg` | Δ has the right sign and reaches half the lock target on E2 AUROC and E1 FP objects at matched recall |
| Lock | the better pooling on 3 seeds, 3/3 signs | E2 AUROC Δ ≥ max(0.02, 2 × paired SE) **and** E1 FP objects at matched recall down by ≥ 2σ_ctrl **and** E1 PR-AUC Δ ≥ −0.01 |
| Adoption candidate | one Test-Realistic one-shot | lock met **and** test object F1 not below the control's beyond the bootstrap CI |

E1 FP objects at matched recall: each model's threshold is set so its object recall equals the control's at
the control's own val-tuned point (MMU 600). A fail at any step is a recorded finding (ledger family E);
the incumbent continues unchanged.

**Phase 0.3–0.5, decided 2026-11-20.**

| phase | status | why |
|---|---|---|
| 0.3 E2 tile set | **moved to after training, before any verdict** | E2 is the primary lock metric, so no result can be judged without it. Build it for the data version under test (its split may move regions). |
| 0.4 frozen-encoder probe | **dropped** | It only protected GPU money; it never decided adoption. The screen does: a `cgm` that adds nothing fails it. The probe's one extra — a *reason* for a fail (no existence signal in the encoder vs the head did not learn it) — can still be had afterwards on the saved control checkpoint. |
| 0.5 control scoring (σ_ctrl) | **moved to after training, before the arm is scored** | σ_ctrl sets the lock thresholds. It must be written here before anyone looks at a `cgm` score, or the thresholds are no longer pre-registered. On a new data version it needs the new control's 3 seeds. |

## 7. Pickup — launch order

1. Train the pair. On D1.0: `computing/train_wave.sh --commit <sha> cgm_max_seed42 cgm_avg_seed42`
   (control = existing `aug_trivialaugment_deploy`). On a new data version: copy its control config, add
   the `model.cgm` block, name it `cgm_<pooling>_seed42`, and train it next to the control.
2. Check the run: `train_tile_loss` falls; `val_tile_loss` is finite; `val_realistic_pr_auc_geomean` is in
   the control's range. A gate that collapses shut shows up as PR-AUC near 0.
3. Build E2 for that data version (Phase 0.3: tiles centred on val-region verdict polygons, read through
   `inference.tiles.read_tile`); re-run `diag_fp_census.py` on a fresh snapshot and re-check §3.
4. Score the control(s) on E1 and E2; write σ_ctrl into this doc (Phase 0.5).
5. Only then score the `cgm` runs and apply §6. Record the result under family E in
   `docs/experiment_ledger.md` and run the update ritual (`CLAUDE.md`).

**Still open from the data side (asked 2026-11-18):** how to treat `rts` outlines as positives (model-drawn
0.30 contours: hand refine, or boundary-ignore ring); whether to cap or balance hard negatives by scene type;
whether the 652 polygons from regions in no split may be used for training. These belong to the data
version, not to `cgm`.

**Environment to recreate (laptop):** `python -m venv .venv`, install `requirements_frozen.txt` (with
`--extra-index-url https://download.pytorch.org/whl/cpu`) then `requirements-cgm.txt`; see §4 for the two
laptop-only workarounds. Raw inputs live in GCS (paths in §2), not in the repo. The original approved plan
is in `C:/Users/Yili Yang/.claude/plans/plan-for-a-new-temporal-kitten.md` (outside the repo; §3 and §6
above supersede it).
