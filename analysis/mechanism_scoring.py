"""Use the unchanged NAVSIM v1 scorer/configuration for frozen trajectories."""
from dataclasses import asdict
from pathlib import Path


def make_engine(workspace):
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    folder = Path(workspace) / "navsim/navsim/planning/script/config/pdm_scoring"
    with initialize_config_dir(config_dir=str(folder), version_base=None):
        config = compose(config_name="default_scoring_parameters")
    return instantiate(config.simulator), instantiate(config.scorer)


def score_one(cache, poses, simulator, scorer):
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    sampling = TrajectorySampling(num_poses=8, interval_length=.5)
    result = pdm_score(cache, Trajectory(poses, sampling), simulator.proposal_sampling, simulator, scorer)
    return {key: float(value) for key, value in asdict(result).items()}
