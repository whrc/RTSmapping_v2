# Dev-log archive — v2.0 / v0.x preliminary phase (2026-04 → 2026-06-12)

Condensed history of the preliminary phase, kept for context. **Full verbatim entries are in git**
at tag `v2.0-preliminary-archive` (`git show v2.0-preliminary-archive:current_working_status.md`).
All numbers below were on the v2.0/v0.2 dataset destroyed 2026-06-12 — superseded by v1.0.

- **2026-04-22 → 04-23** — Living doc seeded; Phase 0 data pipeline (PR #8) merged; Phase 1 training
  loop shipped in 7 commits (models, losses, train.py, MLflow, deployment package, feasibility gates);
  113 tests green. Flat layout + two-tier verification (synthetic pytest / real-bucket) locked.
- **2026-05-01 → 05-02** — Pre-real-data audit (3 reviewers, 10 Critical): LR-range-test implemented,
  `output_bias_prior=0.005`, placeholder configs deleted (recreated per-phase), EXTRA per-group
  normalization intent captured in data.md §9. Pre-smoke fixes plan executed.
- **2026-05-28** — Phase 1 real-data smoke passed on L4 (v2.0, 4572 tiles). Schema migration
  `Tile_id→Tile_ID`, `TrainClass` lowercased. Docker image `rts-train:v2` built + pushed. PR-AUC
  bootstrap OOM fix; negative tiles return synthetic zero labels.
- **2026-05-29 → 05-30** — Exp Phase 0 infra (arm/lr/seed configs); Phase 0a Arm A = 0.5525.
- **2026-06-04 → 06-07** — Phase 0 complete on the frozen 15,528-tile v0.2 snapshot:
  **μ₀=0.5683, σ₀=0.0125, gate G=max(2σ₀,0.01)=0.025** (medium-noise). Gate metric = honest ratios
  [5,10,20]. SSoT-drift repair: `training/experiments.md` made THE program SSoT; gate corrected from
  an erroneous μ₀−2σ₀ floor. Phase 2 data-scaling (0.5361→0.5607, still rising) + Phase 3 loss family
  (compound 2:1 = 0.6035 candidate win; tversky out) run on the A100.
- **2026-06-10 → 06-11** — VM interruption recovery; EMA-resume bug fixed (shadow→model device).
- **2026-06-12** — Migration `ml-training-vm` → **`a100-8x-train`** (8× A100-80GB). Then the
  **v2.0 dataset was destroyed** by an external in-place rewrite of the data-production bucket
  (versioning Suspended → unrecoverable). Lesson: stage frozen snapshots into our own bucket. This
  triggered the v1.0 re-baseline (see `docs/v1.0_rebaseline.md`).

Successor docs: `docs/v1.0_rebaseline.md` (re-baseline), `docs/v1.0_qc.md` (standard-dataset QC),
`current_working_status.md` (live diary).
