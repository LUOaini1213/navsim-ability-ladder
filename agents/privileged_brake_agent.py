"""Privileged (non-deployable) ablation: kinematic rollout + GT box IDM brake.

Uses current-frame annotations from the Scene (OpenScene boxes are ego-frame).
Not leaderboard-legal. Purpose: if PDMS jumps toward Human, the leftover
EgoStatusMLP gap is perception (not better kinematics).
"""
from __future__ import annotations

import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.kinematic_agent import CMD_LEFT, CMD_RIGHT, CMD_STRAIGHT
from navsim.common.dataclasses import AgentInput, Scene, SensorConfig, Trajectory
from navsim.common.enums import BoundingBoxIndex

DYNAMIC = {"vehicle", "pedestrian", "bicycle", "generic_object"}


class PrivilegedBrakeAgent(AbstractAgent):
    requires_scene = True

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
        yaw_rate_cmd: float = 0.20,
        lane_half_width: float = 2.3,
        idm_headway: float = 1.2,
        idm_min_gap: float = 2.5,
        idm_max_acc: float = 1.5,
        idm_comfort_brake: float = 3.0,
        ego_front: float = 4.0,
    ):
        self._trajectory_sampling = trajectory_sampling
        self._yaw_rate_cmd = float(yaw_rate_cmd)
        self._lane_half_width = float(lane_half_width)
        self._idm_headway = float(idm_headway)
        self._idm_min_gap = float(idm_min_gap)
        self._idm_max_acc = float(idm_max_acc)
        self._idm_comfort_brake = float(idm_comfort_brake)
        self._ego_front = float(ego_front)

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        pass

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def _lead_gap_speed(self, scene: Scene, ego_speed: float) -> tuple[float | None, float]:
        """Closest same-lane dynamic object ahead. Returns (bumper_gap, lead_forward_speed)."""
        idx = scene.scene_metadata.num_history_frames - 1
        anns = scene.frames[idx].annotations
        best_gap = None
        best_v = 0.0
        for box, name, vel in zip(anns.boxes, anns.names, anns.velocity_3d):
            if str(name) not in DYNAMIC:
                continue
            x = float(box[BoundingBoxIndex.X])
            y = float(box[BoundingBoxIndex.Y])
            length = float(box[BoundingBoxIndex.LENGTH])
            width = float(box[BoundingBoxIndex.WIDTH])
            half = self._lane_half_width + 0.25 * width
            if str(name) == "pedestrian":
                half = max(half, 3.0)
            if x < 1.5 or x > 55.0 or abs(y) > half:
                continue
            gap = x - 0.5 * length - self._ego_front
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best_v = float(vel[0])  # OpenScene boxes/vel are ego-frame
        return best_gap, best_v

    def _idm_acc(self, v: float, gap: float | None, v_lead: float) -> float:
        if gap is None:
            return 1e9  # no cap
        gap = max(gap, 0.15)
        dv = v - v_lead
        s_star = self._idm_min_gap + v * self._idm_headway + v * dv / (
            2.0 * np.sqrt(self._idm_max_acc * self._idm_comfort_brake) + 1e-6
        )
        s_star = max(s_star, self._idm_min_gap)
        v0 = max(v, 12.0)
        return float(self._idm_max_acc * (1.0 - (v / v0) ** 4 - (s_star / gap) ** 2))

    def _objects(self, scene: Scene) -> list[tuple[float, float, float, float, float]]:
        """(x, y, vx, vy, radius) in ego frame at t=0."""
        idx = scene.scene_metadata.num_history_frames - 1
        anns = scene.frames[idx].annotations
        out = []
        for box, name, vel in zip(anns.boxes, anns.names, anns.velocity_3d):
            if str(name) not in DYNAMIC:
                continue
            x = float(box[BoundingBoxIndex.X])
            y = float(box[BoundingBoxIndex.Y])
            length = float(box[BoundingBoxIndex.LENGTH])
            width = float(box[BoundingBoxIndex.WIDTH])
            radius = 0.5 * float(np.hypot(length, width)) + 1.1
            out.append((x, y, float(vel[0]), float(vel[1]), radius))
        return out

    def _first_collision_time(
        self,
        poses: np.ndarray,
        dt: float,
        objects: list[tuple[float, float, float, float, float]],
    ) -> float | None:
        ttc = None
        for i, pose in enumerate(poses):
            t = (i + 1) * dt
            ex, ey = float(pose[0]), float(pose[1])
            for x, y, vx, vy, radius in objects:
                bx, by = x + vx * t, y + vy * t
                if (ex - bx) ** 2 + (ey - by) ** 2 <= radius * radius:
                    ttc = t if ttc is None else min(ttc, t)
                    break
        return ttc

    def _rollout(self, speed: float, a_long: float, yaw_rate: float) -> np.ndarray:
        num_poses = self._trajectory_sampling.num_poses
        dt = float(self._trajectory_sampling.interval_length)
        poses = np.zeros((num_poses, 3), dtype=np.float32)
        x = y = heading = 0.0
        v = speed
        for i in range(num_poses):
            v = max(0.0, v + a_long * dt)
            heading = heading + yaw_rate * dt
            x = x + v * np.cos(heading) * dt
            y = y + v * np.sin(heading) * dt
            poses[i] = (x, y, heading)
        return poses

    def compute_trajectory(self, agent_input: AgentInput, scene: Scene) -> Trajectory:
        status = agent_input.ego_statuses[-1]
        vel = np.asarray(status.ego_velocity, dtype=np.float64)
        acc = np.asarray(status.ego_acceleration, dtype=np.float64)
        cmd = np.asarray(status.driving_command)
        speed = float(np.hypot(vel[0], vel[1]))
        a_ego = float(acc[0])
        cmd_idx = int(np.argmax(cmd)) if cmd.size else CMD_STRAIGHT
        if cmd_idx == CMD_LEFT:
            yaw_rate = self._yaw_rate_cmd
        elif cmd_idx == CMD_RIGHT:
            yaw_rate = -self._yaw_rate_cmd
        else:
            yaw_rate = 0.0

        gap, v_lead = self._lead_gap_speed(scene, speed)
        a_idm = self._idm_acc(speed, gap, v_lead)
        a_long = min(a_ego, a_idm)
        if gap is not None and gap < 1.0:
            a_long = min(a_long, -self._idm_comfort_brake)

        poses = self._rollout(speed, a_long, yaw_rate)
        dt = float(self._trajectory_sampling.interval_length)
        ttc = self._first_collision_time(poses, dt, self._objects(scene))
        if ttc is not None and ttc > 1e-3 and speed > 0.3:
            # stop before predicted overlap (privileged constant-velocity others)
            a_stop = -speed / max(0.55 * ttc, dt)
            a_long = min(a_long, a_stop)
            poses = self._rollout(speed, a_long, yaw_rate)
        return Trajectory(poses, self._trajectory_sampling)
