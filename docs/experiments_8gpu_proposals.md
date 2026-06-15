# Experiment proposals for the 8× A100 node

2026-06-12. The experiment program (`training/experiments.md`, the SSoT) was designed under
single-GPU scarcity: sequential elimination, single-seed first passes, conditional one-point
probes. We now have **8× A100-80GB** (`a100-8x-train`) and a $70k compute budget expiring
Sep 2026. Reference unit: one 300-epoch run ≈ **14 h on one A100** (observed 13.7 h on v2.0).

These are proposals, not commitments; each needs a go/no-go (recorded in experiments.md §11.3
where program-blocking). Estimated totals if everything ran: ~500–800 GPU-h ≈ $2–4k —
**compute is not the constraint; labeled data is.**

| # | Proposal | What/why | Cost (GPU-h / wallclock) | Status |
|---|---|---|---|---|
| 1 | **Selective parallel multi-seed** | Keep §3.4 single-seed first pass, but when a candidate lands within 1σ of the gate, launch seeds 43+44 *in parallel immediately* (the compound-2:1 tiebreak took 3 serial calendar days; parallel = +0 days). | +14–28 / +0 per borderline candidate | Adopt into v2.1 queue (low risk) |
| 2 | **Loss × boundary factorial** | Sequential elimination assumes the best boundary handling is loss-independent. Running all 3 losses × 2 boundary settings (6 GPUs, simultaneously) exposes interactions (e.g. ignore-band helps compound but hurts focal). | 84 / ~14 h | Judgment call — adopt only if boundary handling is plausibly loss-dependent |
| 3 | **Multi-scale / context-expanded training arm** | 2026-06-12 evidence: single-GSD model collapses at 2× GSD (max prob 0.047 vs 0.145, 0 blobs). If large-RTS / wide-FOV coverage matters, train an arm on 2×-area-downsampled tiles → real §6.4 gate input. | 14–28 + ~1 day pipeline work | Decide at v2.1 re-baseline |
| 4 | **Self-supervised encoder pretraining** | MAE-style pretrain on *unlabeled* Arctic PlanetScope quads (no labels needed; 2025 quads on GCS), then fine-tune on v2.1. Directly attacks the diagnosed regime: data-limited (slope 4.4) + over-parameterized (gap 0.43). The only proposal that can use idle GPUs before labels exist. | ~200–550 / 1–3 days (8 GPUs) + 1–2 days coding | Proposal recorded; user said not yet (2026-06-12) |
| 5 | **Ensemble deployment** | Train final config at 3–5 seeds, average probabilities at inference; classic +1–2% PR-AUC via variance reduction, aligned with precision-over-recall. Inference cost ×k: ~14 GPU-h per pan-arctic pass per member (preliminary ~150 tiles/s; benchmark pending). | 42–70 train + ~14/member/pass | Final-phase decision |
| 6 | **Regularization grid** | wd {1e-2, 5e-2} × aug {base, strong} = 4 cells in parallel, replacing the two conditional single-point probes (§6.3 trigger already fired: gap 0.43 > 0.4). | 56 / ~14 h | Adopt into v2.1 queue (replaces wd_5e2 + aug_strong probes) |
| 7 | **5-seed final lock** | The σ-protocol's stronger option becomes one wallclock unit. | 70 / ~14 h | Decide at Final phase |

## Inference compute (quota question)

Pan-arctic inference is forward-only bf16 and partly GCS-I/O-bound — it does **not** need
A100s. Options:
- **(a) Reuse this node between training phases**: zero new quota; preliminary full-pass cost
  ≈ 14 GPU-h → ~2 h wallclock on 8 GPUs.
- **(b) L4 fleet for the PDG chunked workflow** (`g2-standard-8`, 1× L4, ~$0.85/hr): right ask
  if inference must not contend with training. Suggested quota request: **8× L4,
  us-central1** — cheap, rarely stocked out, sized to a ~half-day pan-arctic pass at
  L4 ≈ ⅓ A100 throughput (to be confirmed by the benchmark).

Firm numbers land with the A100 throughput benchmark (inference.md §11.3).
