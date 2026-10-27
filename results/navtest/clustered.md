# Log-cluster bootstrap on navtest

Whole logs are resampled with replacement; the statistic remains scene-weighted.
Learned-cell contrasts average paired scores over the three fixed training seeds.
Intervals condition on those fitted models and do not estimate training-seed uncertainty.
Exploratory contrasts, percentile 95% intervals; no multiplicity correction.

| comparison (b - a) | scenes | logs | delta PDMS | 95% log-cluster CI |
|---|---:|---:|---:|---|
| rule: add map | 12146 | 136 | +0.2963 | [+0.2586, +0.3356] |
| rule: add boxes + brake | 12146 | 136 | +0.0395 | [+0.0294, +0.0494] |
| pose: add map, no boxes | 12146 | 136 | +0.1811 | [+0.1621, +0.1999] |
| pose: add boxes, no map | 12146 | 136 | +0.0174 | [+0.0022, +0.0320] |
| pose: add boxes, with map | 12146 | 136 | -0.0407 | [-0.0575, -0.0249] |
| pose: map x boxes interaction | 12146 | 136 | -0.0581 | [-0.0704, -0.0465] |
| rule path: learn speed from ego | 12146 | 136 | +0.0194 | [+0.0147, +0.0239] |
| rule path: learn speed from ego + map | 12146 | 136 | +0.0313 | [+0.0249, +0.0382] |
| rule path: add boxes to speed model | 12146 | 136 | -0.0047 | [-0.0081, -0.0015] |
| map input: rule path vs full pose | 12146 | 136 | -0.0060 | [-0.0176, +0.0061] |
| pose seed 0: dropout 0.2 | 12146 | 136 | -0.0124 | [-0.0297, +0.0032] |
| pose seed 0: weight decay 1e-4 | 12146 | 136 | +0.0236 | [+0.0134, +0.0344] |
