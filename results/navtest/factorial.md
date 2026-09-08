# 2x2: what a learned trajectory model is allowed to see (navtest)

One MLP per cell regresses the eight future poses. Inputs are ego kinematics (8),
optionally the on-route map centerline (20x4) and optionally the nearest eight dynamic
GT boxes plus the lead gap (8x11 + 2). Three seeds per cell, trained on navtrain.

| inputs | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|
| kin | 0.646 | 0.649 | 0.630 | **0.642** |
| kin+agents | 0.650 | 0.654 | 0.673 | **0.659** |
| kin+map | 0.818 | 0.811 | 0.839 | **0.823** |
| kin+map+agents | 0.774 | 0.780 | 0.792 | **0.782** |

## Main effects and the interaction, per seed

| comparison | n | dPDMS | 95% CI | dDAC | dNC | dEP | verdict |
|---|---:|---:|---|---:|---:|---:|---|
| add map (no boxes), seed 0 | 12146 | +0.1714 | [+0.1643, +0.1790] | +0.152 | +0.048 | +0.142 | CI excludes 0 |
| add boxes (no map), seed 0 | 12146 | +0.0033 | [-0.0035, +0.0100] | -0.003 | +0.003 | +0.002 | CI spans 0 |
| add map (with boxes), seed 0 | 12146 | +0.1244 | [+0.1163, +0.1323] | +0.123 | +0.029 | +0.117 | CI excludes 0 |
| add boxes (with map), seed 0 | 12146 | -0.0437 | [-0.0512, -0.0373] | -0.032 | -0.016 | -0.024 | CI excludes 0 |
| add map (no boxes), seed 1 | 12146 | +0.1626 | [+0.1551, +0.1703] | +0.150 | +0.034 | +0.156 | CI excludes 0 |
| add boxes (no map), seed 1 | 12146 | +0.0054 | [-0.0017, +0.0124] | +0.005 | +0.002 | +0.013 | CI spans 0 |
| add map (with boxes), seed 1 | 12146 | +0.1258 | [+0.1179, +0.1336] | +0.115 | +0.030 | +0.109 | CI excludes 0 |
| add boxes (with map), seed 1 | 12146 | -0.0314 | [-0.0382, -0.0248] | -0.029 | -0.002 | -0.033 | CI excludes 0 |
| add map (no boxes), seed 2 | 12146 | +0.2094 | [+0.2012, +0.2171] | +0.189 | +0.048 | +0.193 | CI excludes 0 |
| add boxes (no map), seed 2 | 12146 | +0.0436 | [+0.0357, +0.0511] | +0.046 | +0.011 | +0.049 | CI excludes 0 |
| add map (with boxes), seed 2 | 12146 | +0.1190 | [+0.1117, +0.1267] | +0.105 | +0.029 | +0.106 | CI excludes 0 |
| add boxes (with map), seed 2 | 12146 | -0.0468 | [-0.0524, -0.0413] | -0.038 | -0.008 | -0.038 | CI excludes 0 |

The map effect is large and consistent. The box effect is not significant in two of the
three seeds on its own, and is significantly **negative** in all three once the model already
has the map -- while the open-loop validation loss barely moves. Ground-truth perception is
not a free input: the capacity spent on it does not pay off under this score.
