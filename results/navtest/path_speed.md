# Who draws the path, who paces it (navtest)

The rule path is the on-route map centerline; the rule speed is the constant-acceleration
kinematic law. `ds` models predict only progress along the rule path; `pose` models predict
the whole trajectory, and the `_rs` variant keeps that geometry but re-times it with the rule
speed. Human speed is the upper bound at fixed geometry.

| path | speed | seed 0 | seed 1 | seed 2 | mean |
|---|---|---:|---:|---:|---:|
| rule (map centerline) | rule (kinematic) | 0.785 | -- | -- | **0.785** |
| rule (map centerline) | learned, blind | 0.804 | 0.806 | 0.804 | **0.805** |
| rule (map centerline) | learned, sees map | 0.814 | 0.819 | 0.818 | **0.817** |
| rule (map centerline) | learned, map + boxes | 0.812 | 0.812 | 0.812 | **0.812** |
| learned (sees map) | rule (kinematic) | 0.787 | 0.807 | 0.825 | **0.806** |
| learned (sees map) | learned | 0.818 | 0.811 | 0.839 | **0.823** |
| rule (map centerline) | human (upper bound) | 0.835 | -- | -- | **0.835** |

## Paired deltas, per seed

| comparison | n | dPDMS | 95% CI | dDAC | dNC | dEP | verdict |
|---|---:|---:|---|---:|---:|---:|---|
| rule speed -> learned speed, rule path, seed 0 | 12146 | +0.0284 | [+0.0245, +0.0324] | +0.014 | +0.019 | +0.023 | CI excludes 0 |
| learned speed -> human speed, rule path, seed 0 | 12146 | +0.0209 | [+0.0170, +0.0247] | -0.016 | +0.007 | +0.023 | CI excludes 0 |
| rule path -> learned path, rule speed, seed 0 | 12146 | +0.0017 | [-0.0045, +0.0080] | +0.001 | +0.039 | -0.015 | CI spans 0 |
| rule path -> learned path, learned speed, seed 0 | 12146 | +0.0040 | [-0.0019, +0.0102] | -0.022 | +0.012 | +0.005 | CI spans 0 |
| learned speed also sees boxes, seed 0 | 12146 | -0.0018 | [-0.0045, +0.0009] | +0.002 | -0.003 | +0.000 | CI spans 0 |
| rule speed -> learned speed, rule path, seed 1 | 12146 | +0.0335 | [+0.0296, +0.0373] | +0.016 | +0.022 | +0.025 | CI excludes 0 |
| learned speed -> human speed, rule path, seed 1 | 12146 | +0.0158 | [+0.0120, +0.0194] | -0.018 | +0.005 | +0.020 | CI excludes 0 |
| rule path -> learned path, rule speed, seed 1 | 12146 | +0.0214 | [+0.0157, +0.0274] | -0.001 | +0.032 | +0.000 | CI excludes 0 |
| rule path -> learned path, learned speed, seed 1 | 12146 | -0.0076 | [-0.0134, -0.0017] | -0.026 | -0.002 | +0.011 | CI excludes 0 |
| learned speed also sees boxes, seed 1 | 12146 | -0.0069 | [-0.0098, -0.0040] | +0.001 | -0.008 | +0.002 | CI excludes 0 |
| rule speed -> learned speed, rule path, seed 2 | 12146 | +0.0321 | [+0.0284, +0.0360] | +0.016 | +0.020 | +0.029 | CI excludes 0 |
| learned speed -> human speed, rule path, seed 2 | 12146 | +0.0172 | [+0.0133, +0.0210] | -0.019 | +0.007 | +0.017 | CI excludes 0 |
| rule path -> learned path, rule speed, seed 2 | 12146 | +0.0391 | [+0.0333, +0.0447] | +0.016 | +0.037 | +0.017 | CI excludes 0 |
| rule path -> learned path, learned speed, seed 2 | 12146 | +0.0216 | [+0.0163, +0.0269] | -0.006 | +0.007 | +0.029 | CI excludes 0 |
| learned speed also sees boxes, seed 2 | 12146 | -0.0052 | [-0.0081, -0.0024] | +0.000 | -0.004 | -0.004 | CI excludes 0 |

Given the centerline, the learned component earns its place in the speed profile and
nowhere else: letting the model draw the line instead of pacing it is not separable from
the hand rule, while pacing a hand-drawn line beats the hand speed law in every seed.
