"""Shared feature builders for the ability-ladder MLPs.

One implementation used by scripts/build_features.py (training data) and by
LadderMLPAgent (inference), so train-time and test-time features are identical.

Feature blocks
  ego     (8,)        vx, vy, ax, ay, driving command one-hot(4)        kinematics
  cl      (K*4,)      on-route centerline ahead: dx, dy, cos, sin       GT map
  agents  (N*11,)     nearest dynamic boxes, ego frame, mask last       GT agents
  lead    (2,)        same-lane lead bumper gap (m; 200 = none), lead forward speed
"""
from __future__ import annotations

import numpy as np

from navsim.agents.centerline_util import centerline_features, ego_features_from_status, route_path, to_ego
from navsim.common.enums import BoundingBoxIndex as B

K_POINTS, LOOKAHEAD_M, SEARCH_DEPTH = 20, 60.0, 15
N_AGENTS, AGENT_RANGE_M = 8, 60.0
DYNAMIC = {"vehicle", "pedestrian", "bicycle", "generic_object"}
EGO_DIM, CL_DIM, AGENTS_DIM, LEAD_DIM = 8, K_POINTS * 4, N_AGENTS * 11, 2


def agent_features(frame) -> np.ndarray:
    """(N_AGENTS, 11): x, y, vx, vy, l, w, cos h, sin h, is_vehicle, is_ped, mask."""
    anns = frame.annotations
    rows = []
    for box, name, vel in zip(anns.boxes, anns.names, anns.velocity_3d):
        name = str(name)
        if name not in DYNAMIC:
            continue
        x, y = float(box[B._X]), float(box[B._Y])
        d = float(np.hypot(x, y))
        if d > AGENT_RANGE_M:
            continue
        rows.append((d, x, y, float(vel[0]), float(vel[1]), float(box[B._LENGTH]), float(box[B._WIDTH]),
                     float(np.cos(box[B._HEADING])), float(np.sin(box[B._HEADING])),
                     1.0 if name == "vehicle" else 0.0, 1.0 if name == "pedestrian" else 0.0))
    rows.sort(key=lambda r: r[0])
    out = np.zeros((N_AGENTS, 11), dtype=np.float32)
    for i, r in enumerate(rows[:N_AGENTS]):
        out[i, :10] = r[1:]
        out[i, 10] = 1.0
    return out


def lead_features(scene, speed: float) -> np.ndarray:
    from navsim.agents.privileged_brake_agent import PrivilegedBrakeAgent

    gap, v_lead = PrivilegedBrakeAgent()._lead_gap_speed(scene, speed)
    return np.array([200.0 if gap is None else gap, v_lead], dtype=np.float32)


def assemble(ego: np.ndarray, cl: np.ndarray | None, agents: np.ndarray | None, lead: np.ndarray | None,
             see_map: bool, see_agents: bool) -> np.ndarray:
    """Concatenate the enabled blocks in a fixed order: ego [, cl] [, agents, lead]."""
    parts = [np.asarray(ego, np.float32).reshape(-1)]
    if see_map:
        parts.append(np.asarray(cl, np.float32).reshape(-1))
    if see_agents:
        parts.append(np.asarray(agents, np.float32).reshape(-1))
        parts.append(np.asarray(lead, np.float32).reshape(-1))
    return np.concatenate(parts).astype(np.float32)


def input_dim(see_map: bool, see_agents: bool) -> int:
    return EGO_DIM + (CL_DIM if see_map else 0) + (AGENTS_DIM + LEAD_DIM if see_agents else 0)


def scene_blocks(scene):
    """Compute every block for a scene. Returns dict or None if the route is unusable."""
    frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
    ego = ego_features_from_status(frame.ego_status)
    rp = route_path(scene, SEARCH_DEPTH)
    cl = centerline_features(scene, K_POINTS, LOOKAHEAD_M, SEARCH_DEPTH)
    if rp is None or cl is None:
        return None
    path, ego_pose, s0 = rp
    speed = float(np.hypot(ego[0], ego[1]))
    return {
        "ego": ego.astype(np.float32),
        "cl": cl.reshape(-1).astype(np.float32),
        "agents": agent_features(frame),
        "lead": lead_features(scene, speed),
        "path": path,
        "ego_pose": ego_pose,
        "s0": s0,
        "speed": speed,
        "frame": frame,
    }


def kinematic_progress(speed: float, a_long: float, num_poses: int, dt: float) -> np.ndarray:
    """Cumulative distance travelled under constant longitudinal acceleration (PrivMapKin speed law)."""
    out, v, s = [], speed, 0.0
    for _ in range(num_poses):
        v = max(0.0, v + a_long * dt)
        s += v * dt
        out.append(s)
    return np.asarray(out, dtype=np.float64)


def retime_polyline(poses_local: np.ndarray, progress: np.ndarray) -> np.ndarray:
    """Keep the geometry of an ego-frame polyline (N,3) but re-sample it at the given arc lengths."""
    pts = np.vstack([np.zeros((1, 3)), np.asarray(poses_local, np.float64)])
    seg = np.linalg.norm(np.diff(pts[:, :2], axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    want = np.clip(progress, 0.0, float(s[-1]))
    heading = np.unwrap(pts[:, 2])
    x = np.interp(want, s, pts[:, 0])
    y = np.interp(want, s, pts[:, 1])
    h = np.interp(want, s, heading)
    h = np.arctan2(np.sin(h), np.cos(h))
    return np.stack([x, y, h], axis=1).astype(np.float32)


def poses_from_progress(path, ego_pose: np.ndarray, s0: float, ds: np.ndarray) -> np.ndarray:
    """Rule path + a progress profile (8,) -> ego-frame poses (8,3)."""
    ds = np.maximum.accumulate(np.clip(np.asarray(ds, np.float64), 0.0, None))
    remaining = float(getattr(path, "length", s0 + 1e6)) - s0
    ds = np.clip(ds, 0.0, max(remaining - 0.5, 0.0))  # never interpolate past the end of the route
    g = np.asarray(path.interpolate(s0 + ds, as_array=True), dtype=np.float64)
    return to_ego(g, ego_pose).astype(np.float32)
