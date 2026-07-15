# Pretraining — v2.1 SSL program spec

Self-supervised masked-image-modeling (MIM) pretraining of segmentation encoders on unlabeled
pan-arctic PlanetScope imagery. Goal: test whether a domain-adapted encoder beats ImageNet/FCMAE init
under the locked v2 recipe. Experiments SSoT: `docs/experiment_ledger_v21.md` (gate, arms, decision
rules live there — not here).

## 1. Scope & staging

- **Stage 1 (this spec, active):** FCMAE-lite continue-pretraining of ConvNeXt-B on the arctic corpus.
- **Stage 2 (conditional, sketch):** ViT-MAE continue-pretrain from satellite/IN MAE weights; plugs into
  segmentation via `models/foundation.py` (forward_intermediates → SFP → FPN) unchanged. Triggered by
  the Stage-2 rule in the v2.1 ledger gate section.
- Out of scope: hard-negative mining, pseudo-labeling, re-staging (see v2.0 ledger K list).

## 2. Corpus (`scripts/build_pretrain_corpus.py`)

- **Source:** 2025-Q3 Planet visual quads (`gs://pdg-planet-data/global_quarterly/2025/q3/`, indexed by
  `/mnt/outputs/inference/quad_index_2025q3.csv`) + S2 NDVI (`s2_index`). Readers reused verbatim from
  `inference/tiles.py` (`read_rgb_tile` mosaic path + `read_ndvi_tile`) — never reimplement (CLAUDE Rule 3).
- **Tiles:** 512×512 EPSG:3857, 4-ch RGB+NDVI, raw values on disk (normalization at load time, per
  `data/data_format.md`). One `.npz` per tile (`rgb` uint8 [3,512,512], `ndvi` float16 [512,512]) +
  `manifest.csv` (tile_id, lon, lat, band, sector, quad ids).
- **Sampling:** target ~1–2M tiles, stratified uniform per (2° lat band × 20° lon sector) cells from
  `tiles_2025q3_domain_full.csv`; 2× oversample of the training-label footprint neighborhoods.
- **Exclusions:** tiles intersecting val/test region footprints (anti-join vs corrected-split region
  polygons); tiles >50% RGB NoData or all-NaN NDVI.
- **Stats:** fresh corpus per-dataset z-score stats written to `normalization_stats.json` in the corpus
  prefix and used for *pretraining only*. Fine-tuning keeps the v2 training-set stats (intentional:
  stats travel with each dataset; verify the two are near-identical and record in the v2.1 ledger).
- **Layout:** `gs://rts-mapping-v2/RTS_MODEL_V21/PRETRAIN_CORPUS/{tiles/, manifest.csv,
  normalization_stats.json}`; pilot (5k tiles) under `PRETRAIN_CORPUS_PILOT/`.

## 3. Stage-1 method — FCMAE-lite (`pretraining/mim_model.py`, `scripts/pretrain.py`)

In-repo minimal ConvNeXt-V2-style masked autoencoding, dense (non-sparse) convs accepted:

- Encoder: timm ConvNeXt-B (init from `convnextv2_base.fcmae` weights, patch-embed inflated 3→4 ch by
  copying the red-channel kernel for NDVI — same trick as v2's smp 4-ch inflation).
- Masking: random 32×32 patches, mask ratio 0.6, masked patches zeroed at input.
- Decoder: single lightweight conv block projecting the stride-32 feature map to per-patch pixel
  predictions.
- Loss: normalized-pixel MSE on masked patches only (per-patch target normalization, MAE-style).
- Training: AdamW, lr 1.5e-4·(global_batch/256), cosine + warmup, bf16 AMP, DDP via
  `torchrun --nproc_per_node=8` on `a100-8x-train` (single job across all 8 GPUs; global batch
  512 = 64×8). ~200 epochs continue-pretrain.
- Output: encoder-only state_dict checkpoint consumable by `model.encoder_init`.

## 4. Fine-tune protocol

Locked v2 recipe verbatim (config inherits `base_v2_fast.yaml`), changing only:
`model.backbone: tu-convnext*` and the new `model.encoder_init: <path>` key (loader in
`models/segmentation.py` applies the state_dict to the encoder after construction). 3 seeds 42/43/44;
seeds of one arm run in parallel via the per-GPU queue pattern. Scores harvested from
`run_summary.json` into the v2.1 ledger.

## 5. Files

| File | Role |
|---|---|
| `pretraining/corpus.py` | sampling, exclusion, tile writing helpers |
| `pretraining/mim_dataset.py` | npz corpus Dataset + masking collate |
| `pretraining/mim_model.py` | FCMAE-lite encoder+decoder wrapper |
| `scripts/build_pretrain_corpus.py` | corpus builder entry point |
| `scripts/pretrain.py` | DDP pretraining entry point (mirrors train.py MLflow/run_summary conventions) |
| `configs/v21/*.yaml` | pretrain + fine-tune configs |
| `tests/test_pretrain_corpus.py`, `tests/test_pretrain_mim.py` | unit tests (CPU-only) |
