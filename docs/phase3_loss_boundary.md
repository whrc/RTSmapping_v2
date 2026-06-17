# Phase 3 — loss family + boundary handling (v1.0)

Metric: `val_realistic_pr_auc_geomean` (best-smoothed). Gate (`experiments.md §1.4`):
μ₀ = 0.7912, σ₀ = 0.0056, **G = max(2σ₀, 0.01) = 0.0112**, winner bar = μ₀ + G = **0.8024**.

> **Split caveat.** All Phase-3 runs are on the **pre-hotfix (leaky) split** (see
> `docs/v1.0_region_leakage.md`): absolute scores are optimistic, but every candidate
> shares the same split, so **relative gate comparisons hold**. The honest final test +
> Phase-4 EXTRA ablation use the corrected split.

## §6.1 Loss family (single seed 42)

| Loss | best-smoothed | Δ vs μ₀ | verdict |
|------|---|---|---|
| focal (baseline) | 0.7912 | — | — |
| compound 1:2 | 0.7998 | +0.0086 | near-miss (< G) |
| compound 2:1 | 0.7933 | +0.0021 | below gate |
| compound 1:1 | 0.7878 | −0.0034 | below gate |
| tversky 3:7 | 0.5902 | — | poor |
| tversky 2:8 | 0.0729 | — | collapse |

**No loss clears the gate → focal stays the loss winner.** compound 1:2 is the near-miss,
carried into the boundary factorial as the second loss arm.

## §6.2 Boundary factorial — {focal, compound 1:2} × ignore width {1, 2, 3}

Single-seed (42) screen, then a 3-seed confirm (42/43/44) for the two cells that cleared the bar.

| Cell | seed 42 | seed 43 | seed 44 | **3-seed mean** | all > bar |
|------|---|---|---|---|---|
| **focal · ignore w2** | 0.8046 | 0.8200 | 0.8054 | **0.8100** (+0.0188) | ✅ |
| **compound 1:2 · ignore w3** | 0.8025 | 0.8170 | 0.8153 | **0.8116** (+0.0204) | ✅ |
| focal · ignore w1 | 0.7872 | — | — | below bar | — |
| focal · ignore w3 | 0.7973 | — | — | below bar | — |
| compound 1:2 · ignore w1 | 0.7996 | — | — | below bar | — |
| compound 1:2 · ignore w2 | 0.8003 | — | — | below bar | — |

**The win is the boundary, not the loss** — both loss families clear the gate once the
`ignore` band is added; neither cleared it without. The two winners are **tied within noise**
(Δ = 0.0016 ≪ σ₀ = 0.0056).

### Decision — boundary winner: **focal · ignore width 2**
Co-winners by score; `focal · ignore w2` is locked for **simplicity** (single loss vs
focal+dice) and a **narrower ignore band** (less label area discarded), at statistically
identical performance to `compound 1:2 · ignore w3`. This is the architecture for the
Phase-4 combination and the final lock: **UNet++/EffB5 + focal + ignore_w2**.

## §8 Architecture sweep (single seed, RGB-only, vs UNet++/EffB5 baseline) — partial

| Arch | best-smoothed | vs baseline |
|------|---|---|
| FPN | 0.7939 | ≈ baseline, below bar |
| MAnet | 0.6213 | much worse |
| DeepLabV3+ | *running* | — |
| PSPNet | *running* | — |

So far **no smp decoder beats the UNet++ baseline**; two runs pending (will be folded in).

## Bearing on Phase 4
Phase-4 EXTRA runs on the **corrected split** against a matched RGB control at
boundary=none; the boundary win is **additive** and locked separately (sequential
elimination, §6.5 additivity assumption), so the EXTRA delta is measured cleanly without
rebasing onto the boundary winner.
