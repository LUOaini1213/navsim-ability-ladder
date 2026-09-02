"""Shared on-route centerline geometry for the learned map agents.

One implementation, used by both the training scripts and the agents, so the
features seen at training time are byte-for-byte the ones seen at inference.
"""

from __future__ import annotations

import numpy as np
from shapely.geometry import Point

from nuplan.common.maps.maps_datatypes import SemanticMapLayer

from navsim.common.dataclasses import Scene
from navsim.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import Dijkstra
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath

K_POINTS = 20
LOOKAHEAD_M = 60.0
SEARCH_DEPTH = 15


def route_path(scene: Scene, search_depth: int = SEARCH_DEPTH):
    """Return (PDMPath along the on-route centerline, ego pose xyθ, s0) or None."""
    frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
    ids = list(dict.fromkeys(frame.roadblock_ids))
    blocks, lanes = {}, {}
    for rid in ids:
        b = scene.map_api.get_map_object(rid, SemanticMapLayer.ROADBLOCK)
        b = b or scene.map_api.get_map_object(rid, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        if b is None:
            continue
        blocks[b.id] = b
        for lane in b.interior_edges:
            lanes[lane.id] = lane
    if not lanes:
        return None
    ego = np.asarray(frame.ego_status.ego_pose, dtype=np.float64)
    ego_pt = Point(float(ego[0]), float(ego[1]))
    best, best_d = None, 1e9
    for lane in lanes.values():
        try:
            d = float(lane.baseline_path.linestring.distance(ego_pt))
        except Exception:
            continue
        if d < best_d:
            best_d, best = d, lane
    if best is None:
        return None
    block_ids = list(blocks.keys())
    block_list = list(blocks.values())
    try:
        start_idx = int(np.argmax(np.array(block_ids) == best.get_roadblock_id()))
        window = block_list[start_idx : start_idx + search_depth] or block_list[:search_depth]
        route_plan, _ = Dijkstra(best, list(lanes.keys())).search(window[-1])
        discrete = []
        for lane in route_plan:
            discrete.extend(lane.baseline_path.discrete_path)
    except Exception:
        discrete = []
    if len(discrete) < 2:
        discrete = list(best.baseline_path.discrete_path)
    if len(discrete) < 2:
        return None
    path = PDMPath(discrete)
    s0 = float(path.project(ego_pt))
    return path, ego, s0


def to_ego(poses_global: np.ndarray, ego: np.ndarray) -> np.ndarray:
    """Global (N,3) -> ego-frame (N,3): dx, dy, dtheta (wrapped)."""
    yaw = float(ego[2])
    c, s = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[c, -s], [s, c]])
    rel = (poses_global[:, :2] - ego[:2]) @ R.T
    dth = np.arctan2(np.sin(poses_global[:, 2] - yaw), np.cos(poses_global[:, 2] - yaw))
    return np.concatenate([rel, dth[:, None]], axis=1)


def to_global(poses_local: np.ndarray, ego: np.ndarray) -> np.ndarray:
    """Ego-frame (N,3) -> global (N,3)."""
    yaw = float(ego[2])
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    xy = poses_local[:, :2] @ R.T + ego[:2]
    th = poses_local[:, 2] + yaw
    return np.concatenate([xy, th[:, None]], axis=1)


def centerline_features(scene: Scene, k: int = K_POINTS, lookahead: float = LOOKAHEAD_M,
                        search_depth: int = SEARCH_DEPTH):
    """K x 4 features (dx, dy, cos dθ, sin dθ) of the centerline ahead, ego frame."""
    rp = route_path(scene, search_depth)
    if rp is None:
        return None
    path, ego, s0 = rp
    ss = s0 + np.linspace(0.0, lookahead, k)
    try:
        poses = path.interpolate(ss, as_array=True)
    except Exception:
        return None
    rel = to_ego(np.asarray(poses, dtype=np.float64), ego)
    return np.concatenate(
        [rel[:, :2], np.cos(rel[:, 2])[:, None], np.sin(rel[:, 2])[:, None]], axis=1
    ).astype(np.float32)


def ego_features_from_status(status) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(status.ego_velocity, dtype=np.float32),
            np.asarray(status.ego_acceleration, dtype=np.float32),
            np.asarray(status.driving_command, dtype=np.float32),
        ]
    ).astype(np.float32)


def ego_features(scene: Scene) -> np.ndarray:
    return ego_features_from_status(
        scene.frames[scene.scene_metadata.num_history_frames - 1].ego_status
    )
