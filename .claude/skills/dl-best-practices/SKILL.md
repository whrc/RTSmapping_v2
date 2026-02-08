---
name: dl-best-practices
description: Deep learning system design best practices for RTSMAPPING DL model development. Covers project lifecycle, code quality, testing, reproducibility, workflow, deployment, governance, and continuous improvement.
user-invocable: false
---

# Deep Learning System Design: Professional Standards & Best Practices

High-level guidance and professional conventions for developing production-grade deep learning systems as a solo researcher. This document provides meta-level supervision and design philosophy without duplicating technical specifications.

---

## 1. Project Lifecycle Philosophy

### Versioning Strategy

**Models**: Use semantic versioning (major.minor.patch)
- Major: Architecture changes
- Minor: New features (auxiliary channels, multi-scale)
- Patch: Bug fixes, hyperparameter tuning

**Data**: Track versions with checksums and lineage documentation
- Document path: Raw imagery → Tiling → Labeling → Training dataset
- Maintain manifest file with data version, date, known issues

**Release Readiness**
- Test on held-out data
- Document performance and limitations
- Create model card (architecture, data, metrics, intended use, known issues)
- Freeze dependencies with exact versions

### Rollback Planning
- Keep 3-5 previous model checkpoints accessible
- Document how to revert (which files, which commands)
- Test rollback procedure once before deployment

---

## 2. Code Quality Principles

### Organization Philosophy

**Separation of Concerns**
- Data handling separate from model definitions
- Training logic separate from inference logic
- Metrics and losses as pure functions
- Configuration in YAML, not hardcoded in scripts

**Key Principles**
- **DRY**: Don't copy-paste; extract into functions
- **Single Responsibility**: Each file/function does one thing well
- **Configuration over Code**: Parameters in config files, not magic numbers
- **Fail Fast**: Validate inputs early with assertions

### Practical Quality Standards

**For Solo Researchers**
- Use consistent naming conventions
- Add docstrings to non-obvious functions
- Use type hints for function signatures (helps debugging)
- Keep main scripts under 500 lines (split into modules)

**Minimal Tooling**
- Auto-formatter (black or autopep8) to avoid style decisions
- Basic linting (flake8) to catch obvious errors
- No need for comprehensive type checking (mypy) unless helpful

### Configuration Management

**Hierarchical Configs**
- Base config with shared defaults
- Experiment configs that override base
- Always save exact config used with each model checkpoint

---

## 3. Testing Strategy (Pragmatic)

### Essential Tests for ML

**Data Sanity Checks** (run before training)
- Shapes correct (512×512×3 images, matching labels)
- Value ranges valid (0-255 for uint8, 0-1 for normalized)
- No train/val/test overlap (spatial blocking verified)
- Label distribution reasonable (not all zeros)

**Model Sanity Checks** (run before full training)
- Forward pass produces expected output shape
- Model can overfit single batch (loss → 0)
- Gradients flow (check with small test)

**Inference Checks**
- Predictions deterministic with fixed seed
- Output format matches specification
- Handles edge cases (empty tiles, boundary tiles)

**Pragmatic Approach**
- Write tests as discovery scripts initially
- Convert critical ones to automated tests
- Run manually before major experiments
- No need for 100% coverage; focus on critical paths

---

## 4. Reproducibility Essentials

### Five Pillars

| Pillar | Implementation |
|--------|----------------|
| **Code** | Git commit hash; clean working directory |
| **Data** | Data version hash in manifest |
| **Environment** | Dockerfile or frozen requirements.txt |
| **Randomness** | Set all seeds (Python, NumPy, PyTorch) |
| **Hardware** | Document GPU type, CUDA version |

### Experiment Tracking

**Minimum to Track**
- Hyperparameters (LR, batch size, loss function, etc.)
- Performance metrics (train/val IoU, PR-AUC)
- Git commit hash
- Random seeds used
- Training duration and cost

**Practical Tools**
- MLflow for experiment tracking (low overhead)
- Simple naming: `2025-02-08_unetpp_focal_001`
- Tag important runs: `baseline`, `production`, `failed-interesting`

### Deterministic vs Fast Mode

**Trade-off**: Deterministic mode is ~15% slower
- Use deterministic for final runs and debugging
- Use fast mode for hyperparameter search
- Switch with environment variable

---

## 5. Development Workflow

### Iterative Cycle (Realistic)

| Stage | Time | Purpose |
|-------|------|---------|
| **Prototype** | Hours | Validate idea on 100 samples in Colab |
| **Debug** | 1-2 days | Fix bugs on small subset locally |
| **Validate** | 2-3 days | Test on 1000 samples, short epochs |
| **Scale** | 1-2 weeks | Full training on complete dataset |

**Principle**: Start small, validate assumptions, then scale

### Debugging Strategy

**Common Issues and First Steps**

| Issue | First Check |
|-------|-------------|
| Loss not decreasing | Can model overfit single batch? |
| Loss NaN | Check learning rate (too high?), add gradient clipping |
| High train loss | Visualize predictions (labels correct?) |
| Train-val gap large | Too much capacity? Try early stopping |
| Slow training | Profile: data loading or GPU bottleneck? |

**Visualization Habits**
- Save sample predictions every 10 epochs
- Plot training curves (loss, metrics over time)
- Generate PR curves on validation set

### Performance Profiling

**Quick Checks**
- Monitor GPU utilization (`nvidia-smi`)
- Time data loading vs forward pass
- If GPU <90%, increase batch size or optimize I/O
- If data loading >10% of time, increase num_workers

---

## 6. Production Deployment Mindset

### Key Considerations

**Before Deployment**
- Benchmark inference speed on representative tiles
- Test on edge cases (cloudy tiles, partial RTS, empty regions)
- Verify normalization stats match training exactly
- Document expected throughput and costs

**Monitoring in Production**
- Track inference errors and log failures
- Monitor prediction confidence distribution
- Compare outputs to validation set statistics (detect drift)
- Regular sanity checks on random sample tiles

**Maintenance Triggers**
- Performance drops on benchmark tiles
- New labeled data available (>20% increase)
- Systematic errors discovered in deployed predictions

---

## 7. Data & Model Governance

### Data Lineage

**Document the Path**
- Source: PlanetScope 2024 Q3 → processing version → training tiles v2.1
- Include: Processing scripts + versions, quality checks passed, known issues
- Maintain: metadata.csv with tile provenance

### Model Registry

**For Each Production Model**
- Model weights (.pth)
- Training config (YAML)
- Normalization stats (JSON)
- Model card (markdown: architecture, data, metrics, limitations)
- Performance report (test metrics, PR curves)
- Requirements.txt (exact versions)

**Approval Process** (Solo Researcher)
- Self-review checklist: tests pass, performance acceptable, limitations documented
- External validation: Ask collaborator/advisor to spot-check predictions
- Sign-off: Document date and decision rationale

### Bias Assessment

**Geographic Fairness**
- Report performance by Arctic subregion
- Document any systematic differences (e.g., lower recall in Greenland)
- Explain mitigation (region-specific thresholds, more training data)

**Size Fairness**
- Report metrics for small/medium/large RTS
- Document if model struggles with specific sizes

---

## 8. Knowledge Management (Solo)

### Documentation Habits

**Living Documentation**
- Update README when workflow changes
- Add comments in config files explaining choices
- Maintain experiment journal (simple markdown file)

**Experiment Journal**
- Date, experiment ID, goal
- Results summary (1-2 sentences)
- Surprising findings or failures
- Next steps

**Pattern Library** (Informal)
- Keep notes on what worked (e.g., "focal loss + curriculum solved imbalance")
- Document what failed (e.g., "SegFormer OOM on 512px tiles")
- Saves time when revisiting similar problems

### Architecture Decision Records (Light)

For major decisions, create simple markdown file:

```
# Decision: Use UNet++ over DeepLabV3+
Date: 2025-02-08

Why: Superior IoU (0.72 vs 0.68) on v1 experiments
Trade-off: Slower, more memory
Alternatives: DeepLabV3+ (competitive), FPN (faster but worse)
```

---

## 9. Risk Management (Pragmatic)

### Common Failures

| Failure | Prevention | Recovery |
|---------|------------|----------|
| Training crash (OOM) | Profile memory first | Resume from checkpoint |
| Loss NaN | Gradient clipping, LR warmup | Restart from earlier checkpoint |
| Model file corrupted | Checksum validation | Restore from backup |
| Wrong predictions | Test normalization consistency | Fix stats, retrain or recalibrate |

### Backup Strategy

**Essential Backups**
- Code: GitHub (automatic)
- Data: GCS with versioning
- Models: Keep 3 best checkpoints per experiment
- Configs: Committed to Git with each experiment

**Recovery Plan**
- Code loss: Clone from GitHub
- Data loss: Restore from GCS
- Model loss: Retrain from frozen config (should be deterministic)

### Budget Management

**Track Costs**
- Monitor GCP billing weekly
- Set spending alert at 80% of budget
- Log GPU hours per experiment

**Optimize Spending**
- Debug on small subsets locally or Colab
- Use preemptible VMs for fault-tolerant jobs (60-90% cheaper)
- Delete old experiment artifacts regularly

---

## 10. Performance Optimization Philosophy

### Optimization Order (Impact)

1. **Fix data loading first** (often the bottleneck)
   - Increase num_workers if CPU underutilized
   - Use pin_memory=True
   - Cache preprocessed data if possible

2. **Enable mixed precision** (FP16)
   - 2× speedup, minimal accuracy loss
   - Easy to enable in PyTorch

3. **Use gradient accumulation** (if memory limited)
   - Simulate larger batch sizes
   - Accumulate N steps before optimizer step

4. **Distributed training** (multi-GPU)
   - Near-linear speedup with multiple GPUs
   - Only when single GPU too slow

### Benchmarking Approach

**For Inference**
- Warm-up: 100 inferences
- Measure: 1000 inferences, report mean and P95 latency
- Vary batch size to find optimal throughput
- Document: hardware, batch size, speed

**Don't Optimize Prematurely**
- First make it work correctly
- Then measure where time is spent
- Then optimize the slowest part

---

## 11. Ethical Considerations

### Responsible AI (Lightweight)

**Transparency**
- Document model limitations in model card
- Share failure modes honestly (in paper/reports)

**Fairness**
- Report performance across regions and RTS sizes
- Acknowledge systematic biases
- Document mitigation attempts

**Environmental Impact**
- Track GPU hours
- Estimate rough carbon footprint (for awareness)
- Consider efficiency in model selection

### Usage Guidelines

**For RTS Detection**
- Intended use: Scientific research, climate monitoring
- Low risk of misuse (not surveillance, no human subjects)
- Plan to release dataset publicly (with attribution)

---

## 12. Continuous Improvement

### Post-Experiment Review

**After Each Major Experiment**
1. What was the goal?
2. What happened? (results)
3. Why? (analysis)
4. What did we learn?
5. What's next?

Keep it lightweight (1-page notes), focus on insights

### Metric-Driven Progress

**North Star Metrics**
- F0.5 (precision-weighted) for model quality
- Cost per km² for inference efficiency
- Experiment iteration time (aim for <1 week)

**Track Over Time**
- Plot metrics across experiments
- Identify when hitting diminishing returns
- Celebrate improvements

### Failure Library

**Document What Didn't Work**
- "Dice loss → unstable"
- "Global threshold → bad regional performance"
- "No curriculum → model collapsed to all-background"

Prevents repeating failed experiments

---

## 13. Project Health Self-Assessment

### Monthly Check-In

| Indicator | Healthy | Action Needed |
|-----------|---------|---------------|
| **Progress** | 1+ experiment/week | <1 every 2 weeks |
| **Cost** | Within budget | >90% of budget |
| **Documentation** | Recent notes | Outdated >1 month |
| **Reproducibility** | Can reproduce results | Cannot reproduce |
| **Quality** | Test metrics stable | Degrading performance |

**If Unhealthy**
- Identify blockers (computational? data? conceptual?)
- Adjust scope or timeline
- Ask for help (advisors, collaborators)

---

## 14. Practical Workflow Summary

### Daily/Weekly Habits

**When Coding**
- Commit frequently with clear messages
- Test on small data before scaling
- Save configs with experiments

**When Training**
- Monitor first few epochs closely
- Visualize predictions early
- Log to MLflow automatically

**When Analyzing**
- Document surprising results immediately
- Update experiment journal
- Plot metrics over time

**Weekly Review**
- Check spending
- Review experiment progress
- Plan next experiment

### When Stuck

**Debugging Checklist**
1. Can model overfit 10 samples?
2. Are labels correct (visualize)?
3. Is learning rate reasonable?
4. Check gradient flow (any zero grads?)
5. Compare to baseline (simpler model)

**Ask for Help When**
- Stuck >2 days on same issue
- Results contradict expectations with no explanation
- Need validation of approach

---

## 15. Key Takeaways

### Core Principles for Solo Researchers

1. **Start Small, Scale Up**: Prototype on subset, validate, then full training
2. **Document as You Go**: Future you will thank present you
3. **Reproducibility First**: Track everything needed to reproduce
4. **Fail Fast, Learn**: Quick experiments beat perfect experiments
5. **Good Enough > Perfect**: 80/20 rule—focus on high-impact practices
6. **Measure Before Optimize**: Profile before spending time optimizing
7. **Backup Everything**: Code, data, models—assume failure will happen
8. **Track Costs**: Know your burn rate, optimize spending

### What NOT to Do

- Don't optimize code before it works
- Don't train on full dataset without validating on subset
- Don't skip saving configs
- Don't ignore failing tests
- Don't train without checkpointing
- Don't deploy without testing edge cases
- Don't assume reproducibility without verification

---

## 16. References

**Recommended Reading**
- "A Recipe for Training Neural Networks" (Andrej Karpathy)
- "Practical Deep Learning for Coders" (fast.ai)
- Model Cards for Model Reporting (Mitchell et al., 2019)
- MLOps best practices (Google, Microsoft)

**Tools Worth Learning**
- MLflow (experiment tracking)
- DVC (data versioning) - if needed for large datasets
- Docker (reproducible environments)
- Weights & Biases (alternative to MLflow, more features)

---

## Conclusion

Deep learning projects succeed through **disciplined iteration**, not just clever models. As a solo researcher:

- Focus on fundamentals: reproducibility, testing, documentation
- Build incrementally: prototype → validate → scale
- Learn from failures: document what doesn't work
- Optimize pragmatically: measure before optimizing
- Stay organized: future you will thank present you

These practices scale from initial development through production deployment while remaining realistic for one-person teams.

**Remember**: The goal is reliable, reproducible science—not perfect engineering. Good enough practices consistently applied beat perfect practices occasionally followed.