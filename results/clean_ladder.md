# Clean ladder — held-out val logs only

The mini dataset has 64 logs; **62 of them are the warmup_test_e2e logs**. Training on `mini` therefore contaminates the full 563-scene evaluation for the one learned agent, and retraining on "mini minus warmup" is not possible — only 2 logs would remain. So instead of dropping the learned row, every agent is scored on the same held-out val logs.

- train logs 51 / val logs 13 (`available_mini_logs.yaml`)
- scenes: **135 held-out val** vs 428 train-overlap (563 total)


## Ladder on the held-out val scenes (n=135)

| agent | sees | deployable | PDMS (val) | PDMS (train-overlap) | gap |
|---|---|---|---:|---:|---:|
| CV | speed only | yes | **0.181** | 0.250 | +0.069 |
| Kinematic | v, a, command | yes | **0.310** | 0.665 | +0.355 |
| PrivBrake | GT boxes, no map | no | **0.330** | 0.688 | +0.358 |
| MLP (learned) | learned kinematics, blind | yes | **0.475** | 0.692 | +0.216 |
| MapMLP (learned+map) | learned, sees centerline | no | **0.527** | 0.555 | +0.028 |
| PrivMap(IDM) | map centerline + IDM | no | **0.718** | 0.807 | +0.089 |
| PrivMapKin | map centerline + kinematic | no | **0.730** | 0.824 | +0.095 |
| PrivGTPathKin | logged path + kinematic | no | **0.769** | 0.854 | +0.085 |
| PrivMapGTSpd | map centerline + human speed | no | **0.809** | 0.883 | +0.074 |
| Human | logged future | no | **0.914** | 0.954 | +0.040 |

Only the learned row should show a systematic train-overlap advantage; the eight non-learned agents have no training set, so their gap is scene difficulty, not leakage.


## Paired deltas on the clean val scenes

| comparison | dPDMS | 95% CI | verdict |
|---|---:|---|---|
| Add kinematics | +0.1290 | [+0.0525, +0.2058] | CI excludes 0 |
| Learned blind planner vs hand rule | +0.1654 | [+0.0851, +0.2460] | CI excludes 0 |
| Add GT boxes + brake | +0.0202 | [-0.0006, +0.0463] | CI spans 0 |
| Add on-route map centerline | +0.2543 | [+0.1716, +0.3386] | CI excludes 0 |
| Drop IDM on the centerline | +0.0117 | [-0.0177, +0.0426] | CI spans 0 |
| Swap in human speed (geometry fixed) | +0.0795 | [+0.0390, +0.1254] | CI excludes 0 |
| Swap in logged path (speed fixed) | +0.0389 | [-0.0166, +0.0983] | CI spans 0 |
| Remaining gap to Human | +0.1047 | [+0.0580, +0.1568] | CI excludes 0 |
| Give the LEARNED model the centerline too | +0.0513 | [-0.0197, +0.1239] | CI spans 0 |
| Same centerline: hand rule vs learned | +0.2030 | [+0.1176, +0.2905] | CI excludes 0 |

## Speed vs geometry on the clean val scenes

- +0.0406, 95% CI [-0.0169, +0.0963] — not separable at n=135


## Caveats

- n=135 is small; CIs are correspondingly wide. Effects that were significant on 563 scenes may not separate here — that is the honest cost of removing the contaminated scenes, not a defect of the analysis.
- Priv* and Human are upper bounds, not deployable systems.
- `privileged_brake_mini` has two runs (0.602 / 0.593 on the full split); the first is used everywhere for consistency and the discrepancy is unexplained.
- `warmup_test_e2e` is not the official navtest leaderboard.
