# Paired bootstrap CIs over the agent ladder

Split `warmup_test_e2e`, 563 scenes common to all 10 agents, same metric cache. 10000 bootstrap resamples, seed 12345. No evaluation was re-run — this reads the CSVs already in `exp/`.


## Ladder

| agent | sees | PDMS | DAC | DAC fail |
|---|---|---:|---:|---:|
| CV | speed only | 0.233 | 0.643 | 35.7% |
| Kinematic | v, a, command | 0.580 | 0.737 | 26.3% |
| PrivBrake | GT boxes, no map | 0.602 | 0.766 | 23.4% |
| MLP | learned kinematics (blind) | 0.640 | 0.806 | 19.4% |
| MapMLP (learned+map) | learned, sees centerline | 0.548 | 0.798 | 20.2% |
| PrivMap(IDM) | map centerline + IDM | 0.786 | 0.934 | 6.6% |
| PrivMapKin | map centerline + kinematic | 0.802 | 0.950 | 5.0% |
| PrivGTPathKin | logged path + kinematic | 0.833 | 0.973 | 2.7% |
| PrivMapGTSpd | map centerline + human arc-length | 0.866 | 0.950 | 5.0% |
| Human | logged future | 0.945 | 0.998 | 0.2% |

## Paired deltas (positive = the second agent is better)

| comparison | dPDMS | 95% CI | verdict |
|---|---:|---|---|
| Add kinematics (accel + command yaw) | +0.3465 | [+0.3084, +0.3841] | CI excludes 0 |
| Learned blind planner vs hand rule | +0.0600 | [+0.0255, +0.0959] | CI excludes 0 |
| Add GT detection boxes + brake | +0.0221 | [+0.0080, +0.0371] | CI excludes 0 |
| Add on-route map centerline | +0.1617 | [+0.1255, +0.1979] | CI excludes 0 |
| Drop IDM on the centerline (counter-intuitive) | +0.0157 | [+0.0023, +0.0297] | CI excludes 0 |
| Swap in human arc-length speed (geometry fixed) | +0.0641 | [+0.0450, +0.0834] | CI excludes 0 |
| Swap in logged path (speed fixed) | +0.0317 | [+0.0063, +0.0574] | CI excludes 0 |
| Remaining gap to Human | +0.0790 | [+0.0596, +0.1003] | CI excludes 0 |
| Give the LEARNED model the centerline too | -0.0918 | [-0.1270, -0.0554] | CI excludes 0 |
| Same centerline: hand rule vs learned | +0.2535 | [+0.2127, +0.2938] | CI excludes 0 |

## Speed vs geometry (difference in differences)

Gain from swapping in human speed **minus** gain from swapping in the logged path, same 563 paired scenes:

- +0.0324, 95% CI [+0.0075, +0.0563] — speed is worth significantly more


## Caveats

- `privileged_brake_mini` has **two** runs with different results (12:52 PDMS 0.602 / DAC 0.766; 12:55 PDMS 0.593 / DAC 0.785). This report and `pdm_report.md` both use the first. The gap is unexplained and should be pinned down before the number is quoted.
- MLP and MapMLP are the two learned agents; on this split both scores are contaminated (428 of 563 scenes were training logs). The clean comparison is the held-out 135 (MLP 0.475 vs CV 0.181).
- Priv* and Human consume ground truth or map privilege and are **not deployable** — they are upper bounds, not results.
- Ten comparisons were made; no multiplicity correction is applied. The two smallest effects (drop-IDM, logged-path) would not survive a strict Bonferroni threshold and should be reported as suggestive.
- `warmup_test_e2e` is not the official navtest leaderboard.
