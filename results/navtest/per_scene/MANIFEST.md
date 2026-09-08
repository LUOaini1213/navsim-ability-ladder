# Per-scene PDM scores on navtest, as written by NAVSIM's run_pdm_score.py

One gzipped CSV per scoring run: 12,146 scenes, columns `token / valid / the six sub-scores /
score`, floats rounded to six decimals. Written by `analysis/export_per_scene.py` from the
workspace; every table under `results/navtest/` is recomputed from these files by
`analysis/analyze_navtest.py`, and the tests compare the result byte for byte.
`token_log.csv.gz` maps each scene token to its source log (136 logs), taken from the
metric-cache directory layout.

All runs share one metric cache, built once with NAVSIM v1.1 on 2026-09-03.

## Naming

| prefix | meaning |
|---|---|
| *(none)* | hand-written rule agent or an official NAVSIM baseline checkpoint |
| `l_` | a learned cell from `results/navtest/cells.csv`, scored with its own predictions |
| `_rs` | same cell, but the predicted geometry is re-timed with the rule speed law |

`pose_*` cells predict the eight future poses; `ds_*` cells predict only progress along the
rule centerline. `_s0/_s1/_s2` is the training seed.

## Files

| file | scenes | kind |
|---|---:|---|
| `cv.csv.gz` | 12146 | rule / official baseline |
| `egomlp_s0.csv.gz` | 12146 | rule / official baseline |
| `egomlp_s1.csv.gz` | 12146 | rule / official baseline |
| `egomlp_s2.csv.gz` | 12146 | rule / official baseline |
| `human.csv.gz` | 12146 | rule / official baseline |
| `kin.csv.gz` | 12146 | rule / official baseline |
| `l_ds_kin_s0.csv.gz` | 12146 | learned cell |
| `l_ds_kin_s1.csv.gz` | 12146 | learned cell |
| `l_ds_kin_s2.csv.gz` | 12146 | learned cell |
| `l_ds_map_agents_s0.csv.gz` | 12146 | learned cell |
| `l_ds_map_agents_s1.csv.gz` | 12146 | learned cell |
| `l_ds_map_agents_s2.csv.gz` | 12146 | learned cell |
| `l_ds_map_s0.csv.gz` | 12146 | learned cell |
| `l_ds_map_s1.csv.gz` | 12146 | learned cell |
| `l_ds_map_s2.csv.gz` | 12146 | learned cell |
| `l_pose_agents_s0.csv.gz` | 12146 | learned cell |
| `l_pose_agents_s0_rs.csv.gz` | 12146 | learned cell, rule speed |
| `l_pose_agents_s1.csv.gz` | 12146 | learned cell |
| `l_pose_agents_s1_rs.csv.gz` | 12146 | learned cell, rule speed |
| `l_pose_agents_s2.csv.gz` | 12146 | learned cell |
| `l_pose_agents_s2_rs.csv.gz` | 12146 | learned cell, rule speed |
| `l_pose_kin_s0.csv.gz` | 12146 | learned cell |
| `l_pose_kin_s0_rs.csv.gz` | 12146 | learned cell, rule speed |
| `l_pose_kin_s1.csv.gz` | 12146 | learned cell |
| `l_pose_kin_s1_rs.csv.gz` | 12146 | learned cell, rule speed |
| `l_pose_kin_s2.csv.gz` | 12146 | learned cell |
| `l_pose_kin_s2_rs.csv.gz` | 12146 | learned cell, rule speed |
| `l_pose_map_agents_s0.csv.gz` | 12146 | learned cell |
| `l_pose_map_agents_s0_rs.csv.gz` | 12146 | learned cell, rule speed |
| `l_pose_map_agents_s1.csv.gz` | 12146 | learned cell |
| `l_pose_map_agents_s1_rs.csv.gz` | 12146 | learned cell, rule speed |
| `l_pose_map_agents_s2.csv.gz` | 12146 | learned cell |
| `l_pose_map_agents_s2_rs.csv.gz` | 12146 | learned cell, rule speed |
| `l_pose_map_s0.csv.gz` | 12146 | learned cell |
| `l_pose_map_s0_rs.csv.gz` | 12146 | learned cell, rule speed |
| `l_pose_map_s1.csv.gz` | 12146 | learned cell |
| `l_pose_map_s1_rs.csv.gz` | 12146 | learned cell, rule speed |
| `l_pose_map_s2.csv.gz` | 12146 | learned cell |
| `l_pose_map_s2_rs.csv.gz` | 12146 | learned cell, rule speed |
| `l_reg_pose_map_do0.2_s0.csv.gz` | 12146 | learned cell |
| `l_reg_pose_map_do0.2_s0_rs.csv.gz` | 12146 | learned cell, rule speed |
| `l_reg_pose_map_do0.2_wd1e-3_s0.csv.gz` | 12146 | learned cell |
| `l_reg_pose_map_do0.2_wd1e-3_s0_rs.csv.gz` | 12146 | learned cell, rule speed |
| `l_reg_pose_map_h128_s0.csv.gz` | 12146 | learned cell |
| `l_reg_pose_map_h128_s0_rs.csv.gz` | 12146 | learned cell, rule speed |
| `l_reg_pose_map_wd1e-2_s0.csv.gz` | 12146 | learned cell |
| `l_reg_pose_map_wd1e-2_s0_rs.csv.gz` | 12146 | learned cell, rule speed |
| `l_reg_pose_map_wd1e-3_s0.csv.gz` | 12146 | learned cell |
| `l_reg_pose_map_wd1e-3_s0_rs.csv.gz` | 12146 | learned cell, rule speed |
| `l_reg_pose_map_wd1e-4_s0.csv.gz` | 12146 | learned cell |
| `l_reg_pose_map_wd1e-4_s0_rs.csv.gz` | 12146 | learned cell, rule speed |
| `privbrake.csv.gz` | 12146 | rule / official baseline |
| `privgtpath_kin.csv.gz` | 12146 | rule / official baseline |
| `privmap_gtspd.csv.gz` | 12146 | rule / official baseline |
| `privmap_idm.csv.gz` | 12146 | rule / official baseline |
| `privmapkin.csv.gz` | 12146 | rule / official baseline |
| `token_log.csv.gz` | 12146 | scene token -> source log |
