"""Blind kinematic baseline: constant longitudinal accel + yaw from driving command.

Sits between ConstantVelocity (v only) and EgoStatusMLP (learned v,a,command).
Uses the same local planning frame as ConstantVelocityAgent: x forward, y left, heading.
"""
from __future__ import annotations

import numpy as np

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory


# NAVSIM driving_command is a 4-way one-hot: left / straight / right / unknown.
CMD_LEFT, CMD_STRAIGHT, CMD_RIGHT, CMD_UNKNOWN = 0, 1, 2, 3


class KinematicAgent(AbstractAgent):
    """Constant-acceleration bicycle in the ego frame, yaw rate from route command."""

    requires_scene = False

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling = TrajectorySampling(time_horizon=4, interval_length=0.5),
        yaw_rate_cmd: float = 0.20,
    ):
        self._trajectory_sampling = trajectory_sampling
        self._yaw_rate_cmd = float(yaw_rate_cmd)

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        pass

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_no_sensors()

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        status = agent_input.ego_statuses[-1]
        vel = np.asarray(status.ego_velocity, dtype=np.float64)
        acc = np.asarray(status.ego_acceleration, dtype=np.float64)
        cmd = np.asarray(status.driving_command)

        speed = float(np.hypot(vel[0], vel[1]))
        a_long = float(acc[0])
        cmd_idx = int(np.argmax(cmd)) if cmd.size else CMD_STRAIGHT
        if cmd_idx == CMD_LEFT:
            yaw_rate = self._yaw_rate_cmd
        elif cmd_idx == CMD_RIGHT:
            yaw_rate = -self._yaw_rate_cmd
        else:
            yaw_rate = 0.0

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
        return Trajectory(poses, self._trajectory_sampling)
