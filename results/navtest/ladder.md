# The ladder on navtest (12,146 scenes)

Per-scene means of the official PDM sub-scores. `Priv*` and `Human` consume ground truth
or map privilege: they are upper bounds, not deployable planners.

| agent | what it sees | deployable | PDMS | NC | DAC | TTC | C | EP | DDC | NC fail | DAC fail | TTC fail |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ConstantVelocity | speed only | yes | **0.207** | 0.680 | 0.578 | 0.500 | 1.000 | 0.194 | 0.799 | 34.7% | 42.2% | 50.0% |
| Kinematic rule | v, a, driving command | yes | **0.489** | 0.882 | 0.635 | 0.763 | 0.999 | 0.463 | 0.828 | 13.7% | 36.5% | 23.7% |
| PrivBrake rule | + GT boxes | no | **0.529** | 0.948 | 0.702 | 0.872 | 0.926 | 0.422 | 0.871 | 6.4% | 29.8% | 12.8% |
| EgoStatusMLP (official, seed 0) | learned kinematics, blind | yes | **0.655** | 0.930 | 0.773 | 0.836 | 1.000 | 0.626 | 0.907 | 7.9% | 22.7% | 16.4% |
| EgoStatusMLP (official, seed 1) | learned kinematics, blind | yes | **0.674** | 0.932 | 0.793 | 0.847 | 1.000 | 0.637 | 0.904 | 7.7% | 20.7% | 15.3% |
| EgoStatusMLP (official, seed 2) | learned kinematics, blind | yes | **0.663** | 0.930 | 0.782 | 0.837 | 1.000 | 0.632 | 0.902 | 8.0% | 21.8% | 16.3% |
| PrivMap rule, IDM speed | + GT map centerline | no | **0.775** | 0.940 | 0.926 | 0.867 | 0.952 | 0.700 | 0.972 | 6.5% | 7.4% | 13.3% |
| PrivMapKin rule | + GT map centerline | no | **0.785** | 0.938 | 0.930 | 0.854 | 0.972 | 0.734 | 0.973 | 6.7% | 7.0% | 14.6% |
| PrivGTPathKin | logged path, rule speed | no | **0.796** | 0.993 | 0.946 | 0.980 | 0.761 | 0.702 | 0.984 | 0.8% | 5.4% | 2.0% |
| PrivMapGTSpd | GT map path, human speed | no | **0.835** | 0.964 | 0.927 | 0.913 | 0.974 | 0.779 | 0.974 | 4.1% | 7.3% | 8.7% |
| Human | logged future | no | **0.946** | 1.000 | 1.000 | 1.000 | 0.999 | 0.870 | 0.988 | 0.0% | 0.0% | 0.0% |

## Paired deltas

| comparison | n | dPDMS | 95% CI | dDAC | dNC | dEP | verdict |
|---|---:|---:|---|---:|---:|---:|---|
| add kinematics | 12146 | +0.2826 | [+0.2743, +0.2906] | +0.056 | +0.202 | +0.269 | CI excludes 0 |
| add GT boxes + brake | 12146 | +0.0395 | [+0.0345, +0.0442] | +0.067 | +0.065 | -0.042 | CI excludes 0 |
| add GT map centerline | 12146 | +0.2963 | [+0.2876, +0.3049] | +0.295 | +0.055 | +0.270 | CI excludes 0 |
| GT boxes -> GT map | 12146 | +0.2568 | [+0.2479, +0.2652] | +0.228 | -0.010 | +0.312 | CI excludes 0 |
| drop IDM on the centerline | 12146 | +0.0109 | [+0.0079, +0.0139] | +0.004 | -0.003 | +0.034 | CI excludes 0 |
| swap in human speed (path fixed) | 12146 | +0.0493 | [+0.0453, +0.0532] | -0.002 | +0.027 | +0.046 | CI excludes 0 |
| swap in the logged path (speed fixed) | 12146 | +0.0101 | [+0.0031, +0.0168] | +0.017 | +0.055 | -0.032 | CI excludes 0 |
| remaining gap to Human | 12146 | +0.1108 | [+0.1057, +0.1161] | +0.073 | +0.036 | +0.090 | CI excludes 0 |
| hand kinematics -> learned kinematics (official, seed 0) | 12146 | +0.1661 | [+0.1583, +0.1735] | +0.139 | +0.048 | +0.163 | CI excludes 0 |
| learned blind -> hand rule with the map | 12146 | +0.1303 | [+0.1217, +0.1391] | +0.156 | +0.007 | +0.107 | CI excludes 0 |

The map is worth about seven times what ground-truth detection boxes are worth, and it
buys drivable-area compliance; the boxes buy collision avoidance and give back progress.
