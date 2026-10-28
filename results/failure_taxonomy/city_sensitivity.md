# Exploratory: each headline effect with one city left out

Added after the city breakdown in `headline_effects.md` had been seen (Singapore carries a
large share of the box penalty from 16% of the scenes and only 2 sessions), so this is a
post-hoc sensitivity check, not a pre-registered test. Each row drops every scene of one city
and recomputes the scene-weighted effect with log- and session-cluster 95% intervals
(10,000 draws, seed 12345).

| effect | city left out | scenes | sessions | Δ | log 95% | session 95% | verdict log / session |
|---|---|---:|---:|---:|---|---|---|
| E1 | vegas | 8082 | 24 | +0.365 | [+0.319, +0.413] | [+0.294, +0.429] | excludes 0 / excludes 0 |
| E2_avg | vegas | 8082 | 24 | +0.206 | [+0.183, +0.228] | [+0.181, +0.230] | excludes 0 / excludes 0 |
| E2_s0 | vegas | 8082 | 24 | +0.196 | [+0.174, +0.216] | [+0.170, +0.217] | excludes 0 / excludes 0 |
| E2_s1 | vegas | 8082 | 24 | +0.186 | [+0.160, +0.211] | [+0.155, +0.213] | excludes 0 / excludes 0 |
| E2_s2 | vegas | 8082 | 24 | +0.237 | [+0.208, +0.267] | [+0.203, +0.277] | excludes 0 / excludes 0 |
| E3_avg | vegas | 8082 | 24 | -0.043 | [-0.068, -0.020] | [-0.079, -0.005] | excludes 0 / excludes 0 |
| E3_s0 | vegas | 8082 | 24 | -0.046 | [-0.082, -0.012] | [-0.098, +0.006] | excludes 0 / spans 0 |
| E3_s1 | vegas | 8082 | 24 | -0.033 | [-0.063, -0.005] | [-0.079, +0.017] | excludes 0 / spans 0 |
| E3_s2 | vegas | 8082 | 24 | -0.049 | [-0.066, -0.033] | [-0.064, -0.031] | excludes 0 / excludes 0 |
| E4_avg | vegas | 8082 | 24 | -0.053 | [-0.070, -0.037] | [-0.070, -0.037] | excludes 0 / excludes 0 |
| E4_s0 | vegas | 8082 | 24 | -0.045 | [-0.078, -0.016] | [-0.083, -0.009] | excludes 0 / excludes 0 |
| E4_s1 | vegas | 8082 | 24 | -0.030 | [-0.052, -0.010] | [-0.055, -0.003] | excludes 0 / excludes 0 |
| E4_s2 | vegas | 8082 | 24 | -0.083 | [-0.113, -0.052] | [-0.130, -0.041] | excludes 0 / excludes 0 |
| E1 | pittsburgh | 9709 | 33 | +0.265 | [+0.222, +0.311] | [+0.188, +0.339] | excludes 0 / excludes 0 |
| E2_avg | pittsburgh | 9709 | 33 | +0.163 | [+0.143, +0.183] | [+0.139, +0.183] | excludes 0 / excludes 0 |
| E2_s0 | pittsburgh | 9709 | 33 | +0.161 | [+0.140, +0.181] | [+0.133, +0.185] | excludes 0 / excludes 0 |
| E2_s1 | pittsburgh | 9709 | 33 | +0.146 | [+0.123, +0.169] | [+0.119, +0.171] | excludes 0 / excludes 0 |
| E2_s2 | pittsburgh | 9709 | 33 | +0.183 | [+0.160, +0.205] | [+0.158, +0.207] | excludes 0 / excludes 0 |
| E3_avg | pittsburgh | 9709 | 33 | -0.046 | [-0.063, -0.030] | [-0.076, -0.016] | excludes 0 / excludes 0 |
| E3_s0 | pittsburgh | 9709 | 33 | -0.056 | [-0.081, -0.034] | [-0.098, -0.016] | excludes 0 / excludes 0 |
| E3_s1 | pittsburgh | 9709 | 33 | -0.037 | [-0.059, -0.015] | [-0.075, +0.004] | excludes 0 / spans 0 |
| E3_s2 | pittsburgh | 9709 | 33 | -0.045 | [-0.058, -0.031] | [-0.058, -0.031] | excludes 0 / excludes 0 |
| E4_avg | pittsburgh | 9709 | 33 | -0.056 | [-0.068, -0.045] | [-0.068, -0.045] | excludes 0 / excludes 0 |
| E4_s0 | pittsburgh | 9709 | 33 | -0.058 | [-0.079, -0.039] | [-0.088, -0.034] | excludes 0 / excludes 0 |
| E4_s1 | pittsburgh | 9709 | 33 | -0.034 | [-0.052, -0.017] | [-0.056, -0.010] | excludes 0 / excludes 0 |
| E4_s2 | pittsburgh | 9709 | 33 | -0.077 | [-0.099, -0.054] | [-0.111, -0.045] | excludes 0 / excludes 0 |
| E1 | boston | 8415 | 33 | +0.317 | [+0.266, +0.371] | [+0.226, +0.397] | excludes 0 / excludes 0 |
| E2_avg | boston | 8415 | 33 | +0.186 | [+0.163, +0.210] | [+0.156, +0.213] | excludes 0 / excludes 0 |
| E2_s0 | boston | 8415 | 33 | +0.176 | [+0.154, +0.199] | [+0.144, +0.203] | excludes 0 / excludes 0 |
| E2_s1 | boston | 8415 | 33 | +0.174 | [+0.148, +0.200] | [+0.140, +0.202] | excludes 0 / excludes 0 |
| E2_s2 | boston | 8415 | 33 | +0.208 | [+0.181, +0.237] | [+0.175, +0.246] | excludes 0 / excludes 0 |
| E3_avg | boston | 8415 | 33 | -0.055 | [-0.078, -0.035] | [-0.084, -0.026] | excludes 0 / excludes 0 |
| E3_s0 | boston | 8415 | 33 | -0.059 | [-0.091, -0.030] | [-0.103, -0.016] | excludes 0 / excludes 0 |
| E3_s1 | boston | 8415 | 33 | -0.052 | [-0.078, -0.028] | [-0.087, -0.015] | excludes 0 / excludes 0 |
| E3_s2 | boston | 8415 | 33 | -0.055 | [-0.070, -0.041] | [-0.065, -0.042] | excludes 0 / excludes 0 |
| E4_avg | boston | 8415 | 33 | -0.064 | [-0.080, -0.050] | [-0.080, -0.049] | excludes 0 / excludes 0 |
| E4_s0 | boston | 8415 | 33 | -0.054 | [-0.083, -0.028] | [-0.088, -0.022] | excludes 0 / excludes 0 |
| E4_s1 | boston | 8415 | 33 | -0.052 | [-0.072, -0.034] | [-0.068, -0.034] | excludes 0 / excludes 0 |
| E4_s2 | boston | 8415 | 33 | -0.087 | [-0.114, -0.060] | [-0.130, -0.046] | excludes 0 / excludes 0 |
| E1 | singapore | 10232 | 42 | +0.255 | [+0.222, +0.289] | [+0.211, +0.301] | excludes 0 / excludes 0 |
| E2_avg | singapore | 10232 | 42 | +0.174 | [+0.155, +0.194] | [+0.151, +0.198] | excludes 0 / excludes 0 |
| E2_s0 | singapore | 10232 | 42 | +0.159 | [+0.141, +0.176] | [+0.140, +0.177] | excludes 0 / excludes 0 |
| E2_s1 | singapore | 10232 | 42 | +0.150 | [+0.131, +0.170] | [+0.128, +0.173] | excludes 0 / excludes 0 |
| E2_s2 | singapore | 10232 | 42 | +0.214 | [+0.188, +0.240] | [+0.182, +0.246] | excludes 0 / excludes 0 |
| E3_avg | singapore | 10232 | 42 | -0.022 | [-0.036, -0.010] | [-0.034, -0.008] | excludes 0 / excludes 0 |
| E3_s0 | singapore | 10232 | 42 | -0.018 | [-0.037, +0.001] | [-0.034, -0.001] | spans 0 / excludes 0 |
| E3_s1 | singapore | 10232 | 42 | -0.008 | [-0.026, +0.007] | [-0.026, +0.012] | spans 0 / spans 0 |
| E3_s2 | singapore | 10232 | 42 | -0.040 | [-0.052, -0.029] | [-0.050, -0.030] | excludes 0 / excludes 0 |
| E4_avg | singapore | 10232 | 42 | -0.059 | [-0.071, -0.047] | [-0.072, -0.046] | excludes 0 / excludes 0 |
| E4_s0 | singapore | 10232 | 42 | -0.032 | [-0.052, -0.013] | [-0.047, -0.016] | excludes 0 / excludes 0 |
| E4_s1 | singapore | 10232 | 42 | -0.032 | [-0.048, -0.016] | [-0.052, -0.011] | excludes 0 / excludes 0 |
| E4_s2 | singapore | 10232 | 42 | -0.112 | [-0.131, -0.093] | [-0.132, -0.090] | excludes 0 / excludes 0 |
