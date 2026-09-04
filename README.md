https://optiver-close-sigma.vercel.app/

Kaggle, *Optiver — Trading at the Close*, built as a harness first and models
second. `CLAUDE.md` is the spec (the split, the non-negotiables, what each
phase is allowed to claim); `RESEARCH.md` is the log, null results included;
`BENCHMARKS.md` puts the numbers beside the published competition results on
the one statistic that crosses periods.

Phase 1 — fixture, purged forward-chaining CV, the predict-zero floor, ridge.
Phase 2 — LightGBM on causal features with memory (+2.02% over the floor).
Phase 3 — an MLP with a stock embedding beside it, blended with a weight
fitted forward in time; the FULL run is `notebooks/colab_phase3.ipynb`.

```bash
python -m pytest -q          # green on a fresh clone; no raw data needed
```
