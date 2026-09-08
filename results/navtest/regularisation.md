# Regularising the map-conditioned model (navtest, seed 0)

Same architecture and data; only the regularisation changes. The checkpoint is always the
last epoch, so the open-loop validation L1 below (60 held-out navtrain logs) is the number a
loss-based model selection would actually see -- it is not itself a selection criterion here.

| regularisation | val L1 (navtrain) | dL1 vs baseline | PDMS | dPDMS vs baseline | 95% CI | verdict |
|---|---:|---:|---:|---:|---|---|
| none (baseline) | 0.3184 | -- | **0.818** | -- | -- | -- |
| wd 1e-4 | 0.2866 | -0.0318 | **0.841** | +0.0236 | [+0.0192, +0.0283] | CI excludes 0 |
| wd 1e-3 | 0.3099 | -0.0085 | **0.822** | +0.0042 | [-0.0013, +0.0098] | CI spans 0 |
| wd 1e-2 | 0.3646 | +0.0462 | **0.721** | -0.0970 | [-0.1038, -0.0902] | CI excludes 0 |
| dropout 0.2 | 0.2945 | -0.0240 | **0.805** | -0.0124 | [-0.0182, -0.0068] | CI excludes 0 |
| dropout 0.2 + wd 1e-3 | 0.3347 | +0.0162 | **0.762** | -0.0557 | [-0.0621, -0.0494] | CI excludes 0 |
| hidden 128 | 0.2992 | -0.0192 | **0.819** | +0.0015 | [-0.0038, +0.0068] | CI spans 0 |

Two regularisers improve the open-loop fit by a comparable amount and move the simulation
score in opposite directions: weight decay 1e-4 (dL1 -0.032) is worth +0.024 PDMS, dropout 0.2
(dL1 -0.024) costs -0.012, and both intervals exclude zero. Heavy weight decay ruins both
numbers, so the loss is not useless -- across these seven runs its rank correlation with PDMS
is strong. It is among the models that are actually competitive that the ordering breaks:
the unregularised baseline has the worst open-loop fit of the six that still work, and still
outscores dropout 0.2. Treat the open-loop loss as a coarse filter, not a selection criterion.
