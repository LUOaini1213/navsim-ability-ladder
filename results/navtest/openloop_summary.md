# Open-loop error against the simulation score (navtest)

Recomputed from `openloop.csv` (45 scoring runs) with the standard library only.
Spearman is Pearson on average ranks. The `PDMS` column of that CSV is the per-scene mean
of the matching file under `per_scene/` and `val_l1_navtrain` is `final_val_l1` in
`cells.csv`; the tests check both. The displacement errors come from the training runs.

## The 9 map-conditioned runs

Trajectory models that see the centerline and no GT boxes, scored on their own speed --
the cluster the ordering question is about. Ranks are 1 = lowest value.

| run | val L1 (navtrain) | ADE (navtest) | PDMS | ADE rank | PDMS rank |
|---|---:|---:|---:|---:|---:|
| reg_pose_map_wd1e-4_s0 | 0.2866 | 0.7333 | 0.8415 | 2 | 9 |
| pose_map_s2 | 0.2675 | 0.7022 | 0.8392 | 1 | 8 |
| reg_pose_map_wd1e-3_s0 | 0.3099 | 0.7630 | 0.8221 | 5 | 7 |
| reg_pose_map_h128_s0 | 0.2992 | 0.7594 | 0.8194 | 4 | 6 |
| pose_map_s0 | 0.3184 | 0.8204 | 0.8179 | 7 | 5 |
| pose_map_s1 | 0.2847 | 0.7409 | 0.8113 | 3 | 4 |
| reg_pose_map_do0.2_s0 | 0.2945 | 0.7684 | 0.8054 | 6 | 3 |
| reg_pose_map_do0.2_wd1e-3_s0 | 0.3347 | 0.8626 | 0.7622 | 8 | 2 |
| reg_pose_map_wd1e-2_s0 | 0.3646 | 0.8842 | 0.7209 | 9 | 1 |

## What the correlation is made of

Dropping the lowest-scoring run one at a time, from all 9 down to 6.

| runs kept | run dropped | ADE range | PDMS range | Spearman(ADE, PDMS) |
|---:|---|---|---|---:|
| 9 | -- | 0.7022 - 0.8842 | 0.7209 - 0.8415 | -0.8333 |
| 8 | reg_pose_map_wd1e-2_s0 | 0.7022 - 0.8626 | 0.7622 - 0.8415 | -0.7619 |
| 7 | reg_pose_map_do0.2_wd1e-3_s0 | 0.7022 - 0.8204 | 0.8054 - 0.8415 | -0.6429 |
| 6 | reg_pose_map_do0.2_s0 | 0.7022 - 0.8204 | 0.8113 - 0.8415 | -0.5429 |

Down that ladder the correlation goes -0.8333 -> -0.7619 -> -0.6429 -> -0.5429.
It weakens at every step, so the full-group figure is carried by the distance between the
working models and the broken ones more than by any ordering among the runs worth shipping.
Dropout 0.2 against the unregularised baseline: ADE 0.7684 against 0.8204, PDMS 0.8054 against
0.8179 -- the better open-loop fit is the worse score.

## Within a single run

Per-scene rank correlation between displacement error and PDMS, over all 45 runs in the
CSV. The five largest by magnitude:

| run | per-scene Spearman(ADE, PDMS) |
|---|---:|
| pose_agents_s1 | -0.3744 |
| pose_agents_s0 | -0.3650 |
| pose_agents_s1_rs | -0.3628 |
| pose_agents_s2 | -0.3625 |
| pose_kin_s1 | -0.3608 |

Largest magnitude over all 45 runs: 0.3744 (`pose_agents_s1`).
