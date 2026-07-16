# Pretraining — v2.1 SSL program spec

Self-supervised masked-image-modeling (MAE) continue-pretraining of the **DINOv3-Large ViT** encoder on
the unlabeled arctic corpus. Goal: test whether *domain-adapted* pretraining beats the off-the-shelf
sat493m init under the locked v2 recipe. Experiments SSoT: `docs/experiment_ledger_v21.md` (gate, arms,
decision rules live there — not here).

## 1. Scope & direction (decided 2026-07-15)

**Direction: MAE continue-pretrain DINOv3-Large; do not pretrain a convnet.** Rationale: masked
pretraining is ViT-native, and the convnet path hit a hard blocker — the locked **UNet++ decoder is
incompatible with ConvNeXt** (its stride-4 stem yields a phantom 0-channel skip stage at every decoder
depth; ResNet works but has no timm MIM weights). DINOv3-L sidesteps this entirely: it is **already an
integrated, validated encoder** in this repo (family E, `models/foundation.py`: forward_intermediates →
Simple Feature Pyramid → FPN) and **already has a fair-recipe baseline (0.9191, ties EffB5)**. The only
family-E lever never tested is *domain-adapted* ViT weights — precisely what this program provides. This
collapses the earlier two-stage (convnet→ViT) plan into one ViT program.

- Backbone: `vit_large_patch16_dinov3.sat493m` (the family-E encoder), 4-ch patch-embed (RGB copied,
  NDVI zero-init — `FoundationSegmenter._expand_patch_embed`).
- Out of scope: convnet/FCMAE, hard-negative mining, pseudo-labeling, re-staging (v2.0 ledger K list).

## 2. Corpus (`scripts/build_pretrain_corpus.py`)

- **Source:** 2025-Q3 Planet visual quads (`gs://pdg-planet-data/global_quarterly/2025/q3/`, indexed by
  `/mnt/outputs/inference/quad_index_2025q3.csv`) + S2 NDVI (`s2_index`). Readers reused verbatim from
  `inference/tiles.py` (`read_rgb_tile` mosaic path + `read_ndvi_tile`) — never reimplement (CLAUDE Rule 3).
- **Tiles:** 512×512 EPSG:3857, 4-ch RGB+NDVI, raw values on disk (normalization at load time, per
  `data/data_format.md`). One `.npz` per tile (`rgb` uint8 [3,512,512], `ndvi` float16 [512,512]) +
  `manifest.csv` (tile_id, lon, lat, band, sector, quad ids).
- **Geographic scope (decided 2026-07-15):** **4-ch RGB+NDVI, south-only.** NDVI (the v2 4th channel)
  is derived from the S2 composites, which currently cover only the south region
  (`gs://rts-mapping-v2-usw1/S2_RGB/2025_south`). To keep the pretraining input channel-identical to
  the v2/fine-tune input, the corpus is restricted to the **S2-covered footprint** — defined precisely
  as domain-grid tiles whose bbox intersects an S2 index cell (so NDVI is available). Pan-arctic RGB-only
  pretraining and a future pan-arctic NDVI export were considered and deferred. Consequence: the encoder
  sees no high-arctic/Siberian-north terrain; recorded as a corpus limitation in the v2.1 ledger.
- **Sampling:** target ~1–2M tiles, stratified uniform per (2° lat band × 20° lon sector) cells over the
  S2-covered candidate tiles (pre-filtered from `tiles_2025q3_domain_full.csv` by the S2 footprint
  envelope, then exact-intersect confirmed); 2× oversample of the training-label footprint neighborhoods.
- **Exclusions:** tiles intersecting val/test region footprints (anti-join vs corrected-split region
  polygons); tiles >50% RGB NoData or all-NaN NDVI.
- **Stats:** fresh corpus per-dataset z-score stats written to `normalization_stats.json` in the corpus
  prefix and used for *pretraining only*. Fine-tuning keeps the v2 training-set stats (intentional:
  stats travel with each dataset; verify the two are near-identical and record in the v2.1 ledger).
- **Layout:** `gs://rts-mapping-v2/RTS_MODEL_V21/PRETRAIN_CORPUS/{tiles/, manifest.csv,
  normalization_stats.json}`; pilot (5k tiles) under `PRETRAIN_CORPUS_PILOT/`.
- **Size + storage (decided 2026-07-15):** target **~300k tiles**, **materialized to local disk**.
  Measured **0.649 MB/tile** compressed → ~300k ≈ 195 GB (255 GB free holds ~350k). Materialize, don't
  stream-from-GCS: pretraining is multi-epoch, so streaming re-pays the ~2 s/tile cross-region read
  cost *every* epoch and starves the A100s; the build pays it once and every epoch then reads local.
  ~300k is ample for domain-adaptive *continue*-pretraining of the strong sat493m init (the 1–2M figure
  was sized for from-scratch FCMAE, now moot).
- **Parallel build (two-step, `scripts/build_pretrain_corpus.py`):** `--plan-only` runs the expensive
  candidate step once (3.6 GB domain CSV + S2 filter + eval exclusion) → `sample_manifest.csv`; then N
  `--from-sample --shard k --n-shards N` processes materialize disjoint slices across the 96 CPUs
  (I/O-bound, so high N is fine); `--merge` pools the per-shard manifests + stats (exact pooling,
  unit-tested). One-time build ≈ a few hours at high shard count.

## 3. Method — MAE on DINOv3-L (`pretraining/mim_model.py`, `scripts/pretrain.py`)

Masked autoencoding of the DINOv3-L ViT, continue-pretrained from the sat493m init:

- Encoder: the timm ViT built exactly as the fine-tune encoder (`FoundationSegmenter.encoder`), 4-ch
  patch-embed, so the pretrained state_dict drops straight into fine-tune via `model.encoder_init`.
- Tokens: patch16 → 32×32 = 1024 tokens at 512px. Mask ratio 0.75 (standard MAE).
- Masking route: masked patches replaced by a learnable mask token at the ViT input (SimMIM-style,
  which keeps the standard timm ViT forward — full-sequence — rather than the token-gather surgery true
  MAE needs on an Eva model; the reconstruction objective is identical). Revisit true encoder-side token
  dropping only if pretraining throughput is the bottleneck.
- Decoder: a lightweight ViT/linear decoder head predicting per-patch pixels; discarded after
  pretraining.
- Loss: normalized-pixel MSE on masked patches only (per-patch target normalization, MAE-style). NDVI
  channel included (4-ch target); NoData NDVI already neutralized to 0 by `apply_norm`.
- Training: AdamW, lr 1.5e-4·(global_batch/256), cosine + warmup, bf16 AMP, DDP via
  `torchrun --nproc_per_node=8` on `a100-8x-train` (single job across all 8 GPUs). **SimMIM feeds the
  full 1024-token sequence through ViT-L@512 → memory-heavy**, so batch is 32/GPU (global 256) with
  **gradient checkpointing** on the transformer blocks (~25% compute for a large activation-memory cut).
  Measured throughput ~1 s/step (256 img/s) → ~19 min/epoch; **80 epochs ≈ 25 h** continue-pretrain
  (sat493m is a strong init; checkpoints every 20 epochs). Run offline with `HF_HOME=/outputs/hf_cache
  HF_HUB_OFFLINE=1` (sat493m weights cached from family E).
- Output: encoder-only state_dict checkpoint consumable by `model.encoder_init`.

## 4. Fine-tune protocol

Locked v2 recipe verbatim via the **family-E fair-recipe config** (`configs/fm_dinov3sat_l_ndvi_locked.yaml`
— the same one that produced the 0.9191 baseline), adding only the new `model.encoder_init: <path>` key
(loader in `models/segmentation.py` applies the state_dict to `model.encoder` after construction; verified
it round-trips through the Eva ViT with the 4-ch patch-embed). 3 seeds 42/43/44 in parallel via the
per-GPU queue. Scores harvested from `run_summary.json` into the v2.1 ledger.

## 5. Files

| File | Role |
|---|---|
| `pretraining/corpus.py` | sampling, exclusion, tile writing helpers |
| `pretraining/mim_dataset.py` | npz corpus Dataset + masking collate |
| `pretraining/mim_model.py` | MAE (mask-token ViT + light decoder) wrapper around the DINOv3-L encoder |
| `scripts/build_pretrain_corpus.py` | corpus builder entry point |
| `scripts/pretrain.py` | DDP pretraining entry point (mirrors train.py MLflow/run_summary conventions) |
| `configs/v21/*.yaml` | pretrain + fine-tune configs |
| `tests/test_pretrain_corpus.py`, `tests/test_pretrain_mim.py` | unit tests (CPU-only) |
