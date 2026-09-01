"""Privileged map-route centerline follower (not leaderboard-legal).

Uses Scene.map_api + frame.roadblock_ids to build an on-route lane centerline,
then rolls out 4 s along it with the same longitudinal law as PrivilegedBrake
(kinematic accel, optional GT-box IDM). Tests whether the Human gap is DAC/route.
"""
from __future__ import annotations

import numpy as np
from shapely.geometry import Point

from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.kinematic_agent import KinematicAgent
from navsim.agents.privileged_brake_agent import PrivilegedBrakeAgent
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory
from navsim.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import Dijkstra
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (
    convert_absolute_to_relative_se2_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath


class PrivilegedCenterlineAgent(AbstractAgent):
    requires_scene = True

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
        search_depth: int = 30,
        use_idm: bool = True,
        speed_mode: str | None = None,
        path_mode: str = "centerline",
    ):
        super().__init__(requires_scene=True)
        self._trajectory_sampling = trajectory_sampling
        self._search_depth = int(search_depth)
        if speed_mode is None:
            speed_mode = "idm" if use_idm else "kinematic"
        self._speed_mode = str(speed_mode)
        self._path_mode = str(path_mode)
        self._use_idm = self._speed_mode == "idm"

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        pass

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def _load_route(self, scene: Scene):
        frame = scene.frames[scene.scene_metadata.num_history_frames - 1]
        ids = list(dict.fromkeys(frame.roadblock_ids))
        blocks = {}
        lanes = {}
        for rid in ids:
            block = scene.map_api.get_map_object(rid, SemanticMapLayer.ROADBLOCK)
            block = block or scene.map_api.get_map_object(rid, SemanticMapLayer.ROADBLOCK_CONNECTOR)
            if block is None:
                continue
            blocks[block.id] = block
            for lane in block.interior_edges:
                lanes[lane.id] = lane
        return blocks, lanes, frame

    def _starting_lane(self, ego_xy: np.ndarray, lanes: dict):
        ego_pt = Point(float(ego_xy[0]), float(ego_xy[1]))
        best = None
        best_d = 1e9
        for lane in lanes.values():
            try:
                d = float(lane.baseline_path.linestring.distance(ego_pt))
            except Exception:
                continue
            if d < best_d:
                best_d = d
                best = lane
        return best

    def _discrete_centerline(self, start_lane, blocks: dict, lanes: dict):
        if start_lane is None or not blocks:
            return []
        block_list = list(blocks.values())
        block_ids = list(blocks.keys())
        start_idx = int(np.argmax(np.array(block_ids) == start_lane.get_roadblock_id()))
        window = block_list[start_idx : start_idx + self._search_depth]
        if not window:
            window = block_list[: self._search_depth]
        search = Dijkstra(start_lane, list(lanes.keys()))
        route_plan, _found = search.search(window[-1])
        discrete = []
        for lane in route_plan:
            discrete.extend(lane.baseline_path.discrete_path)
        if len(discrete) < 2:
            discrete = list(start_lane.baseline_path.discrete_path)
        return discrete

    def _follow_centerline(self, agent_input: AgentInput, scene: Scene) -> np.ndarray | None:
        blocks, lanes, frame = self._load_route(scene)
        if not lanes:
            return None
        ego = np.asarray(frame.ego_status.ego_pose, dtype=np.float64)
        start = self._starting_lane(ego[:2], lanes)
        discrete = self._discrete_centerline(start, blocks, lanes)
        if len(discrete) < 2:
            return None
        path = PDMPath(discrete)
        s0 = float(path.project(Point(float(ego[0]), float(ego[1]))))

        status = agent_input.ego_statuses[-1]
        speed = float(np.hypot(*np.asarray(status.ego_velocity, dtype=np.float64)))
        a_ego = float(np.asarray(status.ego_acceleration, dtype=np.float64)[0])
        a_long = a_ego
        if self._use_idm:
            brake = PrivilegedBrakeAgent(trajectory_sampling=self._trajectory_sampling)
            gap, v_lead = brake._lead_gap_speed(scene, speed)
            a_idm = brake._idm_acc(speed, gap, v_lead)
            a_long = min(a_ego, a_idm)
            if gap is not None and gap < 1.0:
                a_long = min(a_long, -brake._idm_comfort_brake)

        dists = s0 + self._progress_deltas(agent_input, scene, speed, a_long)
        global_poses = path.interpolate(np.asarray(dists, dtype=np.float64), as_array=True)
        local = convert_absolute_to_relative_se2_array(StateSE2(*ego), np.asarray(global_poses, dtype=np.float64))
        return local.astype(np.float32)

    def _gt_arc_lengths(self, scene: Scene) -> np.ndarray:
        gt = np.asarray(scene.get_future_trajectory(self._trajectory_sampling.num_poses).poses, dtype=np.float64)
        step = np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1)
        return np.concatenate([[step[0] if len(step) else 0.0], step]).cumsum()

    def _progress_deltas(self, agent_input: AgentInput, scene: Scene, speed: float, a_long: float) -> np.ndarray:
        num = self._trajectory_sampling.num_poses
        dt = float(self._trajectory_sampling.interval_length)
        if self._speed_mode == "human":
            return self._gt_arc_lengths(scene)
        dists = []
        v = speed
        s = 0.0
        for _ in range(num):
            v = max(0.0, v + a_long * dt)
            s = s + v * dt
            dists.append(s)
        return np.asarray(dists, dtype=np.float64)

    def _follow_gt_path(self, agent_input: AgentInput, scene: Scene) -> np.ndarray | None:
        gt = np.asarray(scene.get_future_trajectory(self._trajectory_sampling.num_poses).poses, dtype=np.float64)
        if gt.shape[0] < 2:
            return None
        status = agent_input.ego_statuses[-1]
        speed = float(np.hypot(*np.asarray(status.ego_velocity, dtype=np.float64)))
        a_ego = float(np.asarray(status.ego_acceleration, dtype=np.float64)[0])
        a_long = a_ego
        if self._speed_mode == "idm":
            brake = PrivilegedBrakeAgent(trajectory_sampling=self._trajectory_sampling)
            gap, v_lead = brake._lead_gap_speed(scene, speed)
            a_idm = brake._idm_acc(speed, gap, v_lead)
            a_long = min(a_ego, a_idm)
            if gap is not None and gap < 1.0:
                a_long = min(a_long, -brake._idm_comfort_brake)
        if self._speed_mode == "human":
            return gt.astype(np.float32)
        gt_s = np.concatenate([[0.0], np.linalg.norm(np.diff(gt[:, :2], axis=0), axis=1)]).cumsum()
        want = self._progress_deltas(agent_input, scene, speed, a_long)
        want = np.clip(want, 0.0, float(gt_s[-1]))
        heading = np.unwrap(gt[:, 2])
        x = np.interp(want, gt_s, gt[:, 0])
        y = np.interp(want, gt_s, gt[:, 1])
        h = np.interp(want, gt_s, heading)
        h = np.arctan2(np.sin(h), np.cos(h))
        return np.stack([x, y, h], axis=1).astype(np.float32)

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        try:
            if self._path_mode == "human":
                poses = self._follow_gt_path(agent_input, scene)
            else:
                poses = self._follow_centerline(agent_input, scene)
            if poses is not None and poses.shape == (self._trajectory_sampling.num_poses, 3):
                return Trajectory(poses, self._trajectory_sampling)
        except Exception:
            pass
        return KinematicAgent(trajectory_sampling=self._trajectory_sampling).compute_trajectory(agent_input)
