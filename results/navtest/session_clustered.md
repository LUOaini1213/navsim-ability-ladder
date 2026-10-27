# Session-cluster bootstrap on navtest

Whole sessions are resampled with replacement; the statistic remains scene-weighted.
Learned-cell contrasts average paired scores over the three fixed training seeds.
Intervals condition on those fitted models and do not estimate training-seed uncertainty.
Exploratory contrasts, percentile 95% intervals; no multiplicity correction.

| comparison (b - a) | scenes | sessions | delta PDMS | 95% session-cluster CI |
|---|---:|---:|---:|---|
| rule: add map | 12146 | 44 | +0.2963 | [+0.2328, +0.3563] |
| rule: add boxes + brake | 12146 | 44 | +0.0395 | [+0.0280, +0.0511] |
| pose: add map, no boxes | 12146 | 44 | +0.1811 | [+0.1586, +0.2020] |
| pose: add boxes, no map | 12146 | 44 | +0.0174 | [-0.0082, +0.0414] |
| pose: add boxes, with map | 12146 | 44 | -0.0407 | [-0.0667, -0.0163] |
| pose: map x boxes interaction | 12146 | 44 | -0.0581 | [-0.0707, -0.0460] |
| rule path: learn speed from ego | 12146 | 44 | +0.0194 | [+0.0135, +0.0261] |
| rule path: learn speed from ego + map | 12146 | 44 | +0.0313 | [+0.0245, +0.0393] |
| rule path: add boxes to speed model | 12146 | 44 | -0.0047 | [-0.0085, -0.0013] |
| map input: rule path vs full pose | 12146 | 44 | -0.0060 | [-0.0201, +0.0063] |
| pose seed 0: dropout 0.2 | 12146 | 44 | -0.0124 | [-0.0325, +0.0065] |
| pose seed 0: weight decay 1e-4 | 12146 | 44 | +0.0236 | [+0.0120, +0.0362] |
