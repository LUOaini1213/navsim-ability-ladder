"""Smoke test: can we load navtest logs, build scenes, touch the map, and run a rule agent?"""
import os, sys, time
from pathlib import Path
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader

split = sys.argv[1] if len(sys.argv) > 1 else "navtest"
data_split = {"navtest": "test", "navtrain": "trainval"}[split]
cfg_dir = str(Path(os.environ["NAVSIM_DEVKIT_ROOT"]) / "navsim/planning/script/config/common/train_test_split/scene_filter")
with initialize_config_dir(config_dir=cfg_dir, version_base=None):
    sf_cfg = compose(config_name=split)
scene_filter: SceneFilter = instantiate(sf_cfg)
print(f"[{split}] filter: {len(scene_filter.log_names)} log names, history={scene_filter.num_history_frames}, future={scene_filter.num_future_frames}")

t0 = time.time()
loader = SceneLoader(
    sensor_blobs_path=None,
    data_path=Path(os.environ["OPENSCENE_DATA_ROOT"]) / "navsim_logs" / data_split,
    scene_filter=scene_filter,
    sensor_config=SensorConfig.build_no_sensors(),
)
print(f"[{split}] {len(loader.tokens)} scenes across {len(loader.get_tokens_list_per_log())} logs  ({time.time()-t0:.1f}s)")

tok = loader.tokens[0]
t0 = time.time()
scene = loader.get_scene_from_token(tok)
frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
print(f"token {tok}: map={scene.scene_metadata.map_name}, ego_pose={frame.ego_status.ego_pose}, roadblocks={len(frame.roadblock_ids)}  ({time.time()-t0:.1f}s)")

from navsim.agents.centerline_util import centerline_features, route_path
t0 = time.time()
rp = route_path(scene)
cl = centerline_features(scene)
print(f"route_path ok={rp is not None}, centerline feats shape={None if cl is None else cl.shape}  ({time.time()-t0:.1f}s)")

from navsim.agents.kinematic_agent import KinematicAgent
ai = loader.get_agent_input_from_token(tok)
traj = KinematicAgent().compute_trajectory(ai)
print("kinematic traj poses:", traj.poses.shape)
print("SMOKE TEST OK")
