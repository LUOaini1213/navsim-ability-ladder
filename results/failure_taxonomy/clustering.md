# Where clustering changes a conclusion

For the seed-averaged effects (E1 single run), each category mean Δ_c gets three 95% intervals:
the iid scene bootstrap of the published tables (`bootstrap_large`, 2,000 draws), the log-cluster
and the session-cluster bootstrap (10,000 draws). Listed: every category where the three
verdicts ("excludes 0" / "spans 0") are not all the same.

| effect | feature | category | scenes | Δ_c | iid 95% | log 95% | session 95% | verdicts iid / log / session |
|---|---|---|---:|---:|---|---|---|---|
| E3_avg | city | pittsburgh | 2437 | -0.020 | [-0.032, -0.008] | [-0.071, +0.012] | [-0.043, +0.002] | excludes 0 / spans 0 / spans 0 |
| E4_avg | command | left | 2501 | -0.020 | [-0.034, -0.005] | [-0.041, +0.002] | [-0.041, +0.002] | excludes 0 / spans 0 / spans 0 |
| E3_avg | command | right | 1575 | -0.016 | [-0.029, -0.002] | [-0.043, +0.013] | [-0.049, +0.022] | excludes 0 / spans 0 / spans 0 |
| E4_avg | command | right | 1575 | -0.028 | [-0.051, -0.007] | [-0.062, +0.005] | [-0.059, +0.005] | excludes 0 / spans 0 / spans 0 |
| E3_avg | curvature | bend | 1640 | -0.035 | [-0.047, -0.023] | [-0.063, -0.007] | [-0.068, +0.005] | excludes 0 / excludes 0 / spans 0 |
| E1 | lane_change | yes | 1260 | +0.078 | [+0.047, +0.110] | [+0.002, +0.159] | [-0.003, +0.162] | excludes 0 / excludes 0 / spans 0 |
| E3_avg | lead | close | 2876 | -0.009 | [-0.016, -0.002] | [-0.020, +0.002] | [-0.022, +0.005] | excludes 0 / spans 0 / spans 0 |
| E4_avg | speed | slow | 5149 | -0.011 | [-0.019, -0.002] | [-0.026, +0.005] | [-0.022, +0.003] | excludes 0 / spans 0 / spans 0 |
| E4_avg | speed | stopped | 848 | -0.016 | [-0.030, -0.002] | [-0.033, +0.002] | [-0.034, +0.002] | excludes 0 / spans 0 / spans 0 |
| E4_avg | stop_go | starting | 1209 | -0.016 | [-0.027, -0.005] | [-0.034, +0.000] | [-0.034, +0.001] | excludes 0 / spans 0 / spans 0 |
| E1 | stop_go | stopping | 901 | +0.019 | [+0.005, +0.033] | [-0.001, +0.039] | [-0.001, +0.039] | excludes 0 / spans 0 / spans 0 |

11 of 104 category-level effect intervals change verdict between the three bootstraps.

Interval width relative to iid (median over the 104 rows): log 2.07x, session 2.13x.
