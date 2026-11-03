# Pre-registered hypotheses (docs/FAILURE_TAXONOMY.md)

Decided under both log- and session-cluster bootstrap (10,000 draws, seed 12345); for E2/E3
the seed average and each of the three seeds must agree. "confirmatory" here means "specified
before the analysis"; navtest had already been used in this project, so this is not fresh-data
confirmation. Group differences resample all clusters together; draws in which a group is
empty are dropped (counted below).

| hypothesis | effect | estimate | log 95% | session 95% | dropped draws (log/session) | holds here |
|---|---|---:|---|---|---|---|
| H1 | E2_avg | +0.017 | [-0.014, +0.048] | [-0.015, +0.047] | 0 / 0 | no |
| H1 | E2_s0 | +0.052 | [+0.021, +0.084] | [+0.019, +0.083] | 0 / 0 | yes |
| H1 | E2_s1 | +0.011 | [-0.023, +0.046] | [-0.027, +0.046] | 0 / 0 | no |
| H1 | E2_s2 | -0.014 | [-0.050, +0.023] | [-0.049, +0.024] | 0 / 0 | no |
| H1r | E1 | -0.033 | [-0.080, +0.013] | [-0.083, +0.017] | 0 / 0 | no |
| H3a | E3_avg | -0.067 | [-0.100, -0.034] | [-0.102, -0.020] | 0 / 0 | yes |
| H3a | E3_s0 | -0.060 | [-0.111, -0.007] | [-0.122, +0.019] | 0 / 0 | no |
| H3a | E3_s1 | -0.072 | [-0.111, -0.034] | [-0.112, -0.015] | 0 / 0 | yes |
| H3a | E3_s2 | -0.070 | [-0.092, -0.048] | [-0.081, -0.052] | 0 / 0 | yes |
| H3b | E3_avg | +0.041 | [+0.010, +0.073] | [+0.001, +0.069] | 0 / 0 | no |
| H3b | E3_s0 | +0.025 | [-0.026, +0.074] | [-0.047, +0.078] | 0 / 0 | no |
| H3b | E3_s1 | +0.063 | [+0.026, +0.099] | [+0.017, +0.093] | 0 / 0 | no |
| H3b | E3_s2 | +0.036 | [+0.010, +0.062] | [+0.014, +0.055] | 0 / 0 | no |

## Verdicts

- **H1** (learned map gain larger on curved routes: Δ_E2(bend ∪ turn) − Δ_E2(straight) > 0): not supported (1 of 4 effect/seed rows hold under both clusterings).
- **H1r** (rule map gain larger on curved routes: Δ_E1(bend ∪ turn) − Δ_E1(straight) > 0): not supported (0 of 1 effect/seed rows hold under both clusterings).
- **H3a** (box penalty present without a lead: Δ_E3(lead = none) < 0): not supported (3 of 4 effect/seed rows hold under both clusterings).
- **H3b** (box penalty larger with a lead: Δ_E3(lead ∈ close, mid, far) − Δ_E3(none) < 0): not supported (0 of 4 effect/seed rows hold under both clusterings).
