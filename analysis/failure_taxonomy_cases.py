# -*- coding: utf-8 -*-
"""Stratified root-cause sample and failure gallery (docs/FAILURE_TAXONOMY.md).

    python analysis/failure_taxonomy_cases.py select --out results/failure_taxonomy/cases
    & $env:NAVSIM_PYTHON -X utf8 analysis\\failure_taxonomy_cases.py score --source archive --out ...
    & $env:NAVSIM_PYTHON -X utf8 analysis\\failure_taxonomy_cases.py score --source scene --city vegas --out ...
    & $env:NAVSIM_PYTHON -X utf8 analysis\\failure_taxonomy_cases.py report --out ...

select   (standard library) draws the pre-registered sample: five strata, hash order, at
         most one case per log per stratum -> sample.csv, sample_counts.json.
score    rebuilds each plan, scores it with the unchanged official NAVSIM scorer against the
         navtest metric cache, checks it against the committed per-scene row and reads the
         scorer's own state to find the failure step and the object involved.
         --source archive: the kinematic rule and the learned cells, from the study's frozen
           inputs (exp/features/features_navtest.npz, the score path of
           analysis/validate_score_path.py on the controlled-study branch); no map needed.
         --source scene: PrivMapKin, from the raw log and the map with NAVSIM's SceneLoader;
           one city per invocation keeps the map in memory small.
report   applies the root-cause rules (failure_taxonomy_core.root_cause), writes the count
         table with Wilson intervals, picks the gallery by the pre-registered rule and draws
         the bird's-eye views from the metric cache.
Nothing here modifies the devkit, the metric cache or any committed result.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))

import failure_taxonomy_core as core  # noqa: E402


def write_lf(path, text):
    """Write text with LF line endings on every platform (the repository normalises to LF)."""
    with io.open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


PER_SCENE = REPO / 'results' / 'navtest' / 'per_scene'
FEATURES_CSV = REPO / 'results' / 'failure_taxonomy' / 'features.csv.gz'
SUB = ['no_at_fault_collisions', 'drivable_area_compliance', 'ego_progress', 'time_to_collision_within_bound',
       'comfort', 'driving_direction_compliance', 'score']
SHORT = {'no_at_fault_collisions': 'nc', 'drivable_area_compliance': 'dac', 'ego_progress': 'ep',
         'time_to_collision_within_bound': 'ttc', 'comfort': 'c', 'driving_direction_compliance': 'ddc',
         'score': 'pdms'}


def read_scores(stem):
    with io.TextIOWrapper(gzip.open(PER_SCENE / (stem + '.csv.gz'), 'rb'), encoding='utf-8') as f:
        return {r['token']: {k: float(r[k]) for k in SUB} for r in csv.DictReader(f)}


def read_token_log():
    with io.TextIOWrapper(gzip.open(PER_SCENE / 'token_log.csv.gz', 'rb'), encoding='utf-8') as f:
        return {r['token']: r['log'] for r in csv.DictReader(f)}


def read_features():
    with io.TextIOWrapper(gzip.open(FEATURES_CSV, 'rb'), encoding='utf-8') as f:
        return {r['token']: r for r in csv.DictReader(f)}


def stratum_runs(stratum, seed):
    sid, _label, failing, contrast = next(s for s in core.STRATA if s[0] == stratum)
    fmt = (lambda t: None if t is None else t.format(s=seed))
    return fmt(failing), fmt(contrast)


# --------------------------------------------------------------------------- select

def cmd_select(args):
    out = Path(args.out)
    if (out / 'sample.csv').exists():
        raise SystemExit('refusing to overwrite %s/sample.csv' % out)
    out.mkdir(parents=True, exist_ok=True)
    token_log = read_token_log()
    tokens = sorted(token_log)
    cache = {}

    def pdms(stem):
        if stem not in cache:
            cache[stem] = {t: v['score'] for t, v in read_scores(stem).items()}
        return cache[stem]

    sample, counts = [], {}
    for sid, label, failing_t, contrast_t in core.STRATA:
        seeds = [None] if '{s}' not in failing_t else list(core.SEEDS)
        per_seed = core.PER_STRATUM // len(seeds)
        cands = {}
        for seed in seeds:
            failing, contrast = stratum_runs(sid, seed)
            fv, cv = pdms(failing), (pdms(contrast) if contrast else None)
            cands[seed] = [t for t in tokens if core.stratum_member(sid, fv[t], cv[t] if cv else None)]
        chosen = core.select_cases(sid, cands, token_log, per_seed)
        counts[sid] = {'label': label, 'per_seed_requested': per_seed,
                       'candidates': {str(k): len(v) for k, v in cands.items()},
                       'candidate_logs': {str(k): len({token_log[t] for t in v}) for k, v in cands.items()},
                       'selected': {str(k): sum(1 for c in chosen if c['seed'] == k) for k in cands}}
        for c in chosen:
            failing, contrast = stratum_runs(sid, c['seed'])
            c.update(failing_run=failing, contrast_run=contrast or '')
            sample.append(c)
    with io.open(out / 'sample.csv', 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['stratum', 'seed', 'token', 'log', 'failing_run', 'contrast_run', 'hash'])
        for c in sample:
            w.writerow([c['stratum'], '' if c['seed'] is None else c['seed'], c['token'], c['log'],
                        c['failing_run'], c['contrast_run'], c['hash']])
    write_lf((out / 'sample_counts.json'), json.dumps(counts, indent=2, sort_keys=True) + '\n')
    print(json.dumps(counts, indent=2))


def read_sample(out):
    with io.open(Path(out) / 'sample.csv', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['seed'] = None if r['seed'] == '' else int(r['seed'])
    return rows


# --------------------------------------------------------------------------- NAVSIM runtime

def configure_runtime(workspace, threads):
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[var] = str(threads)
    ws = Path(workspace)
    os.environ.setdefault('NAVSIM_WORKSPACE', str(ws))
    os.environ.setdefault('NAVSIM_DEVKIT_ROOT', str(ws / 'navsim'))
    os.environ.setdefault('OPENSCENE_DATA_ROOT', str(ws / 'dataset'))
    os.environ.setdefault('NUPLAN_MAPS_ROOT', str(ws / 'dataset' / 'maps'))
    os.environ.setdefault('NUPLAN_MAP_VERSION', 'nuplan-maps-v1.0')
    os.environ.setdefault('NAVSIM_EXP_ROOT', str(ws / 'exp'))
    sys.path.insert(0, os.environ['NAVSIM_DEVKIT_ROOT'])
    return ws


def make_engine(ws):
    """The unchanged v1 simulator and scorer, from NAVSIM's own scoring config."""
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    folder = Path(ws) / 'navsim' / 'navsim' / 'planning' / 'script' / 'config' / 'pdm_scoring'
    with initialize_config_dir(config_dir=str(folder), version_base=None):
        cfg = compose(config_name='default_scoring_parameters')
    return instantiate(cfg.simulator), instantiate(cfg.scorer)


def load_metric_cache(ws, log, token):
    import lzma
    import pickle
    path = Path(ws) / 'exp' / 'metric_cache_navtest' / log / 'unknown' / token / 'metric_cache.pkl'
    with lzma.open(path, 'rb') as f:
        return pickle.load(f)


def sampling():
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
    return TrajectorySampling(time_horizon=4, interval_length=0.5)


class ArchivePlans(object):
    """Plans of the kinematic rule and the learned cells from the frozen navtest inputs."""

    def __init__(self, ws):
        import numpy as np
        self.np = np
        self.ws = Path(ws)
        z = np.load(str(self.ws / 'exp' / 'features' / 'features_navtest.npz'), allow_pickle=False)
        self.index = {str(t): i for i, t in enumerate(z['token'])}
        self.ego, self.cl, self.agents, self.lead = z['ego'], z['cl'], z['agents'], z['lead']
        self.models = {}

    def kinematic(self, token):
        """KinematicAgent itself, fed the t0 ego status it reads (velocity, acceleration, command)."""
        np = self.np
        from navsim.agents.kinematic_agent import KinematicAgent
        from navsim.common.dataclasses import AgentInput, EgoStatus
        e = self.ego[self.index[token]]
        status = EgoStatus(ego_pose=np.zeros(3, dtype=np.float64), ego_velocity=e[0:2].astype(np.float32),
                           ego_acceleration=e[2:4].astype(np.float32), driving_command=e[4:8].astype(np.int64),
                           in_global_frame=False)
        traj = KinematicAgent(trajectory_sampling=sampling(), yaw_rate_cmd=0.20).compute_trajectory(
            AgentInput([status], [], []))
        return np.asarray(traj.poses, dtype=np.float32), False

    def learned(self, run, token):
        """The model branch of LadderMLPAgent.compute_trajectory on the archived input blocks."""
        np = self.np
        import torch
        from navsim.agents.ladder_features import assemble
        from navsim.agents.ladder_mlp_agent import load_run
        tag = run[2:]  # l_pose_map_s0 -> pose_map_s0
        if tag not in self.models:
            torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
            self.models[tag] = load_run(str(self.ws / 'exp' / 'ladder' / tag))
        model, mu, sd, norm = self.models[tag]
        if str(norm['target']) != 'pose':
            raise SystemExit('only pose cells are sampled, got %s' % run)
        i = self.index[token]
        x = assemble(self.ego[i], self.cl[i], self.agents[i], self.lead[i], bool(norm['see_map']),
                     bool(norm['see_agents']))
        x = (x - mu) / sd
        with torch.no_grad():
            out = model(torch.tensor(x)[None, :]).numpy()[0].astype(np.float64)
        poses = out.reshape(-1, 3)[:8].astype(np.float32)
        if poses.shape == (8, 3) and np.all(np.isfinite(poses)):
            return poses, False
        return self.kinematic(token)[0], True

    def plan(self, run, token):
        if run == 'kin':
            return self.kinematic(token)
        if run.startswith('l_pose_'):
            return self.learned(run, token)
        raise ValueError('archive source cannot build %s' % run)


def scene_plans(ws, log, tokens):
    """PrivMapKin plans from the raw log + map, exactly as run_pdm_score.py builds them."""
    import numpy as np
    from navsim.agents.kinematic_agent import KinematicAgent
    from navsim.agents.privileged_centerline_agent import PrivilegedCenterlineAgent
    from navsim.common.dataclasses import SceneFilter, SensorConfig
    from navsim.common.dataloader import SceneLoader
    sf = SceneFilter(num_history_frames=4, num_future_frames=10, frame_interval=1, has_route=True,
                     log_names=[log], tokens=list(tokens))
    loader = SceneLoader(data_path=Path(ws) / 'dataset' / 'navsim_logs' / 'test', sensor_blobs_path=None,
                         scene_filter=sf, sensor_config=SensorConfig.build_no_sensors())
    agent = PrivilegedCenterlineAgent(trajectory_sampling=sampling(), path_mode='centerline',
                                      speed_mode='kinematic', use_idm=False)
    agent.initialize()
    out = {}
    for tok in tokens:
        agent_input = loader.get_agent_input_from_token(tok)
        scene = loader.get_scene_from_token(tok)
        poses = np.asarray(agent.compute_trajectory(agent_input, scene).poses, dtype=np.float32)
        kin = np.asarray(KinematicAgent(trajectory_sampling=sampling()).compute_trajectory(agent_input).poses,
                         dtype=np.float32)
        out[tok] = (poses, bool(np.array_equal(poses, kin)))
    return out


# --------------------------------------------------------------------------- diagnosis

def _local(states, origin):
    """Global (x, y, heading) rows -> the t0 rear-axle frame."""
    ox, oy, oh = origin
    c, s = math.cos(-oh), math.sin(-oh)
    out = []
    for x, y, h in states:
        dx, dy = x - ox, y - oy
        out.append((c * dx - s * dy, s * dx + c * dy, math.atan2(math.sin(h - oh), math.cos(h - oh))))
    return out


def object_type(obj):
    name = obj.tracked_object_type.name.lower()
    return {'vehicle': 'vehicle', 'pedestrian': 'pedestrian', 'bicycle': 'bicycle'}.get(name, name)


def collision_object(scorer, k_nc):
    """Replays the scorer's at-fault collision loop for the planned proposal (index 1)."""
    from nuplan.planning.metrics.utils.collision_utils import CollisionType
    from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer_utils import get_collision_type
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex
    obs = scorer._observation
    collided = list(obs.collided_track_ids)
    for t in range(int(k_nc) + 1):
        poly = scorer._ego_polygons[1, t]
        for g in obs[t].query(poly, predicate='intersects'):
            token = obs[t].tokens[int(g)]
            if obs.red_light_token in token or token in collided:
                continue
            in_ml = bool(scorer._ego_areas[1, t, EgoAreaIndex.MULTIPLE_LANES]
                         or scorer._ego_areas[1, t, EgoAreaIndex.NON_DRIVABLE_AREA])
            obj = obs.unique_objects[token]
            ctype = get_collision_type(scorer._states[1, t], poly, obj, obs[t][token])
            at_fault = ctype in (CollisionType.ACTIVE_FRONT_COLLISION, CollisionType.STOPPED_TRACK_COLLISION) or (
                in_ml and ctype == CollisionType.ACTIVE_LATERAL_COLLISION)
            if at_fault:
                return token, obj, ctype.name.lower().replace('_collision', ''), t
            collided.append(token)
    return None


def ttc_object(scorer, k_ttc):
    """Replays the scorer's TTC loop for proposal 1 and returns the object of the first infraction."""
    import numpy as np
    from shapely import creation
    from nuplan.common.actor_state.state_representation import StateSE2
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    from nuplan.planning.simulation.observation.idm.utils import is_agent_ahead, is_agent_behind
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex, EgoAreaIndex, StateIndex
    obs = scorer._observation
    states = scorer._states[1]
    coords = scorer._ego_coords[1].copy()
    coords[:, BBCoordsIndex.CENTER, :] = coords[:, BBCoordsIndex.FRONT_LEFT, :]
    speeds = np.hypot(states[:, StateIndex.VELOCITY_X], states[:, StateIndex.VELOCITY_Y])
    dxy = np.stack([np.cos(states[:, StateIndex.HEADING]) * speeds, np.sin(states[:, StateIndex.HEADING]) * speeds], -1)
    collided = list(obs.collided_track_ids)
    dt = scorer.proposal_sampling.interval_length
    for t in range(int(k_ttc) + 1):
        for fidx in np.arange(0, 10, 3):
            cur = t + int(fidx)
            poly = creation.polygons(coords[t] + dxy[t] * float(fidx) * dt)
            for g in obs[cur].query(poly, predicate='intersects'):
                token = obs[cur].tokens[int(g)]
                if obs.red_light_token in token or token in collided or speeds[t] < scorer._config.stopped_speed_threshold:
                    continue
                in_ml = bool(scorer._ego_areas[1, t, EgoAreaIndex.MULTIPLE_LANES]
                             or scorer._ego_areas[1, t, EgoAreaIndex.NON_DRIVABLE_AREA])
                ego_rear = StateSE2(*states[t, StateIndex.STATE_SE2])
                centroid = obs[cur][token].centroid
                track = StateSE2(centroid.x, centroid.y, obs.unique_objects[token].box.center.heading)
                if is_agent_ahead(ego_rear, track) or (
                        (in_ml or scorer._drivable_area_map.is_in_layer(ego_rear.point, layer=SemanticMapLayer.INTERSECTION))
                        and not is_agent_behind(ego_rear, track)):
                    if t == int(k_ttc):
                        return token, obs.unique_objects[token], cur, t
                else:
                    collided.append(token)
    return None


def relation(state, centroid, heading):
    """(heading difference deg, object centre ahead of the ego rear axle?)"""
    x, y, h = float(state[0]), float(state[1]), float(state[2])
    ahead = math.cos(h) * (centroid.x - x) + math.sin(h) * (centroid.y - y) > 0
    return core.heading_diff_deg(h, float(heading)), bool(ahead)


def score_and_diagnose(ws, simulator, scorer, log, token, poses, logged):
    import numpy as np
    from navsim.common.dataclasses import Trajectory
    from navsim.evaluate.pdm_score import pdm_score
    from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import EgoAreaIndex
    mc = load_metric_cache(ws, log, token)
    res = pdm_score(mc, Trajectory(np.asarray(poses, dtype=np.float32), sampling()),
                    simulator.proposal_sampling, simulator, scorer)
    got = {SHORT[k]: float(getattr(res, k)) for k in SUB}
    rear = mc.ego_state.rear_axle
    origin = (rear.x, rear.y, rear.heading)
    states = scorer._states[1]
    sim = _local([(float(s[0]), float(s[1]), float(s[2])) for s in states], origin)
    offroad = np.flatnonzero(scorer._ego_areas[1, :, EgoAreaIndex.NON_DRIVABLE_AREA])
    k_dac = int(offroad[0]) if len(offroad) else None
    k_nc = None if not math.isfinite(scorer._collision_time_idcs[1]) else int(scorer._collision_time_idcs[1])
    k_ttc = None if not math.isfinite(scorer._ttc_time_idcs[1]) else int(scorer._ttc_time_idcs[1])
    path = core.logged_polyline(logged)
    d = dict(got)
    d.update(nc_step=k_nc, dac_step=k_dac, ttc_step=k_ttc,
             logged_length_m=core.polyline_length([(0.0, 0.0)] + [(p[0], p[1]) for p in logged]),
             logged_heading_change_deg=math.degrees(float(logged[-1][2])),
             plan_length_m=core.polyline_length([(0.0, 0.0)] + [(float(p[0]), float(p[1])) for p in poses]),
             progress_plan_m=float(scorer._progress_raw[1]), progress_pdm_closed_m=float(scorer._progress_raw[0]),
             sim_xy=[(round(p[0], 3), round(p[1], 3)) for p in sim], sim_heading=[round(p[2], 4) for p in sim])
    if k_dac is not None:
        from shapely import Point
        from shapely.ops import unary_union
        from nuplan.common.maps.maps_datatypes import SemanticMapLayer
        pt = sim[k_dac]
        d['dac_lateral_m'] = abs(core.signed_offset((pt[0], pt[1]), path))
        d['dac_travelled_m'] = core.polyline_length([(p[0], p[1]) for p in sim[:k_dac + 1]])
        # descriptive only: how far the worst ego corner is outside the polygons the DAC check uses
        dmap = mc.drivable_area_map
        idcs = dmap.get_indices_of_map_type([SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION,
                                             SemanticMapLayer.DRIVABLE_AREA, SemanticMapLayer.CARPARK_AREA])
        drivable = unary_union([dmap._geometries[i] for i in idcs])
        def worst(k):
            return float(max(drivable.distance(Point(float(x), float(y))) for x, y in scorer._ego_coords[1, k][:4]))

        d['dac_corner_outside_m'] = worst(k_dac)
        d['dac_corner_outside_max_m'] = max(worst(k) for k in range(k_dac, scorer._ego_coords.shape[1]))
        # is an offending corner on a mapped lane / lane connector that the DAC polygons do not cover?
        lanes = unary_union([dmap._geometries[i] for i in dmap.get_indices_of_map_type(
            [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR])])
        off = [Point(float(x), float(y)) for x, y in scorer._ego_coords[1, k_dac][:4]
               if drivable.distance(Point(float(x), float(y))) > 0]
        d['dac_corner_on_lane'] = bool(off) and all(lanes.contains(p) for p in off)
    if k_nc is not None:
        hit = collision_object(scorer, k_nc)
        if hit:
            tok, obj, ctype, t = hit
            hd, ahead = relation(states[t], scorer._observation[t][tok].centroid, obj.box.center.heading)
            d.update(nc_object_type=object_type(obj), nc_collision_type=ctype, nc_heading_diff_deg=hd,
                     nc_object_ahead=ahead, nc_object_token=tok)
    if k_ttc is not None:
        hit = ttc_object(scorer, k_ttc)
        if hit:
            tok, obj, cur, t = hit
            hd, ahead = relation(states[t], scorer._observation[cur][tok].centroid, obj.box.center.heading)
            d.update(ttc_object_type=object_type(obj), ttc_heading_diff_deg=hd, ttc_object_ahead=ahead,
                     ttc_object_token=tok)
    return d


def cmd_score(args):
    out = Path(args.out)
    target = out / ('scored_%s%s.jsonl' % (args.source, '' if not args.city else '_' + args.city))
    if target.exists():
        raise SystemExit('refusing to overwrite %s' % target)
    ws = configure_runtime(args.workspace, args.threads)
    from failure_taxonomy_features import wait_for_memory
    wait_for_memory(args.min_free_mb, 'score %s %s' % (args.source, args.city or ''), {'memory_waits': []})
    import numpy as np
    sample = read_sample(out)
    feats = read_features()
    committed = {}

    def row(run, tok):
        if run not in committed:
            committed[run] = read_scores(run)
        return {SHORT[k]: v for k, v in committed[run][tok].items()}

    z = np.load(str(ws / 'exp' / 'features' / 'features_navtest.npz'), allow_pickle=False)
    index = {str(t): i for i, t in enumerate(z['token'])}
    y_pose = z['y_pose']
    simulator, scorer = make_engine(ws)
    jobs = []  # (case, role, run)
    for c in sample:
        for role, run in (('failing', c['failing_run']), ('contrast', c['contrast_run'])):
            if not run:
                continue
            scene_run = run == 'privmapkin'
            if (args.source == 'scene') != scene_run:
                continue
            if args.source == 'scene' and args.city and feats[c['token']]['city'] != args.city:
                continue
            jobs.append((c, role, run))
    print('%d plans to score from %s%s' % (len(jobs), args.source, ' (%s)' % args.city if args.city else ''), flush=True)
    plans = {}
    if args.source == 'archive':
        arch = ArchivePlans(ws)
        for c, role, run in jobs:
            plans[(c['token'], run)] = arch.plan(run, c['token'])
    else:
        by_log = {}
        for c, role, run in jobs:
            by_log.setdefault(c['log'], set()).add(c['token'])
        for n, (log, toks) in enumerate(sorted(by_log.items()), 1):
            for tok, val in scene_plans(ws, log, sorted(toks)).items():
                plans[(tok, 'privmapkin')] = val
            print('  [%d/%d] %s' % (n, len(by_log), log), flush=True)
    t0 = time.time()
    with io.open(target, 'w', encoding='utf-8', newline='\n') as f:
        for c, role, run in jobs:
            poses, fallback = plans[(c['token'], run)]
            logged = [tuple(float(v) for v in r) for r in y_pose[index[c['token']]].reshape(8, 3)]
            d = score_and_diagnose(ws, simulator, scorer, c['log'], c['token'], poses, logged)
            ref = row(run, c['token'])
            err = max(abs(d[k] - ref[k]) for k in ref)
            d.update(stratum=c['stratum'], seed=c['seed'], token=c['token'], log=c['log'], role=role, run=run,
                     fallback=fallback, committed=ref, max_abs_error=err, reproduced=err <= 1e-4,
                     plan=[[round(float(v), 4) for v in p] for p in poses],
                     logged=[[round(v, 4) for v in p] for p in logged])
            f.write(json.dumps(d, sort_keys=True) + '\n')
    try:
        import psutil
        mem = psutil.Process().memory_info()
        peak = getattr(mem, 'peak_wset', getattr(mem, 'rss', 0)) / 2**20
    except Exception:
        peak = float('nan')
    print('scored %d plans in %.0f s -> %s (peak working set %.0f MB)' % (len(jobs), time.time() - t0, target, peak),
          flush=True)


# --------------------------------------------------------------------------- report

def read_scored(out):
    recs = []
    for p in sorted(Path(out).glob('scored_*.jsonl')):
        with io.open(p, encoding='utf-8') as f:
            recs += [json.loads(line) for line in f if line.strip()]
    return recs


LABEL_TEXT = {
    'dac_lateral_curve': 'leaves the road sideways in a bend or turn',
    'dac_lateral_straight': 'drifts off the road on a straight',
    'dac_overshoot': 'runs past the end of the drivable path the human used',
    'dac_near_logged_path': 'off-road while close to the logged path',
    'nc_rear_end': 'drives into the object ahead in its lane',
    'nc_crossing': 'collides with crossing or turning traffic',
    'nc_lateral': 'side collision while straddling lanes or off-road',
    'nc_vru': 'collides with a pedestrian or cyclist',
    'nc_static': 'hits a static object',
    'ttc_following': 'closes in on the object ahead (TTC < bound)',
    'ttc_crossing': 'TTC infraction with crossing traffic',
    'comfort': 'comfort bounds exceeded',
    'fallback': 'network not used (kinematic fallback)',
    'unreproduced': 'committed score not reproduced',
    'other': 'none of the rules applies',
}


def cmd_report(args):
    out = Path(args.out)
    recs = read_scored(out)
    sample = read_sample(out)
    feats = read_features()
    by_key = {(r['token'], r['run'], r['role']): r for r in recs}
    cases = []
    for c in sample:
        f = by_key.get((c['token'], c['failing_run'], 'failing'))
        if f is None:
            raise SystemExit('missing scored plan for %s %s' % (c['token'], c['failing_run']))
        k = by_key.get((c['token'], c['contrast_run'], 'contrast')) if c['contrast_run'] else None
        label = core.root_cause(dict(f, fallback=f['fallback'] and c['failing_run'].startswith('l_')))
        desc = ''
        if c['stratum'] == 'S3' and k is not None:
            mp = k['plan']
            bp = f['plan']
            desc = core.plan_difference(f['plan_length_m'], k['plan_length_m'], bp[-1][:2], mp[-1][:2], mp[-1][2])
        cases.append(dict(case=c, failing=f, contrast=k, label=label, primary=core.primary_failure(f),
                          s3=desc, feat=feats[c['token']]))
    write_cases_csv(out, cases)
    lines = root_cause_md(cases)
    gallery = pick_gallery(cases)
    for i, g in enumerate(gallery, 1):
        g['png'] = 'case%02d_%s_%s.png' % (i, g['case']['stratum'], g['case']['token'])
    ws = configure_runtime(args.workspace, args.threads) if not args.no_plots else None
    if ws is not None:
        (out / 'gallery').mkdir(exist_ok=True)
        for g in gallery:
            draw_case(ws, g, out / 'gallery')
    lines += gallery_md(gallery)
    write_lf((out / 'root_causes.md'), '\n'.join(lines) + '\n')
    print('\n'.join(lines))


def write_cases_csv(out, cases):
    cols = ['stratum', 'seed', 'token', 'log', 'failing_run', 'contrast_run', 'label', 'primary', 's3_plan_difference',
            'reproduced', 'max_abs_error', 'fallback', 'pdms', 'nc', 'dac', 'ttc', 'ep', 'c', 'contrast_pdms',
            'contrast_reproduced', 'dac_step', 'nc_step', 'ttc_step', 'dac_lateral_m', 'dac_corner_outside_m',
            'dac_corner_outside_max_m', 'dac_corner_on_lane', 'dac_travelled_m',
            'logged_length_m', 'logged_heading_change_deg', 'plan_length_m', 'nc_object_type', 'nc_collision_type',
            'nc_heading_diff_deg', 'nc_object_ahead', 'ttc_object_type', 'ttc_heading_diff_deg', 'ttc_object_ahead',
            'city', 'speed', 'command', 'lead', 'junction', 'curvature', 'lane_change', 'stop_go', 'route_len_m']
    with io.open(out / 'cases.csv', 'w', encoding='utf-8', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(cols)
        for x in cases:
            c, f, k, ft = x['case'], x['failing'], x['contrast'], x['feat']
            vals = dict(stratum=c['stratum'], seed='' if c['seed'] is None else c['seed'], token=c['token'],
                        log=c['log'], failing_run=c['failing_run'], contrast_run=c['contrast_run'], label=x['label'],
                        primary=x['primary'], s3_plan_difference=x['s3'], contrast_pdms=k['pdms'] if k else '',
                        contrast_reproduced=k['reproduced'] if k else '')
            for key in cols:
                if key in vals:
                    continue
                if key in ft:
                    vals[key] = ft[key]
                else:
                    v = f.get(key, '')
                    vals[key] = ('%.4f' % v) if isinstance(v, float) else v
            w.writerow([vals[k2] for k2 in cols])


def root_cause_md(cases):
    L = ['# Root causes of a stratified failure sample (navtest)', '',
         'Sample, strata and labelling rules: `docs/FAILURE_TAXONOMY.md` (fixed before any case was',
         'labelled). Each plan was rebuilt and scored with the unchanged official scorer against the',
         'navtest metric cache; a case counts only if PDMS and all sub-scores reproduce the committed',
         'per-scene row within 1e-4. Shares are descriptive, with Wilson 95% intervals; one case per log',
         'per stratum. Per-case detail: `cases.csv`.', '']
    rep = sum(1 for x in cases if x['failing']['reproduced'])
    worst = max(x['failing']['max_abs_error'] for x in cases)
    L += ['Reproduced: **%d of %d** failing plans (largest absolute sub-score error %.2g).' % (rep, len(cases), worst)]
    kc = [x['contrast'] for x in cases if x['contrast']]
    if kc:
        L += ['Contrast plans reproduced: %d of %d (largest absolute sub-score error %.2g).' % (
            sum(1 for k in kc if k['reproduced']), len(kc), max(k['max_abs_error'] for k in kc))]
    L.append('')
    strata = [s[0] for s in core.STRATA]
    labels = [lab for lab in LABEL_TEXT if any(x['label'] == lab for x in cases)]
    L += ['| label | meaning | ' + ' | '.join('%s %s' % (s, dict((a, b) for a, b, _, _ in core.STRATA)[s])
                                             for s in strata) + ' |',
          '|---|---|' + '---|' * len(strata)]
    totals = {s: sum(1 for x in cases if x['case']['stratum'] == s and x['label'] != 'unreproduced') for s in strata}
    for lab in labels:
        cells = []
        for s in strata:
            k = sum(1 for x in cases if x['case']['stratum'] == s and x['label'] == lab)
            if lab == 'unreproduced':
                cells.append(str(k) if k else '')
                continue
            n = totals[s]
            if not k:
                cells.append('')
                continue
            lo, hi = core.wilson(k, n)
            cells.append('%d/%d (%.0f%%, %.0f-%.0f%%)' % (k, n, 100.0 * k / n, 100 * lo, 100 * hi))
        L.append('| `%s` | %s | %s |' % (lab, LABEL_TEXT[lab], ' | '.join(cells)))
    L += ['', '## How marginal are the drivable-area failures? (descriptive)', '',
          'For every reproduced case whose primary failure is DAC: the distance of the simulated plan from',
          'the logged path at the first off-road step, and how far the worst ego corner gets outside the',
          'polygons the DAC check uses (roadblocks, intersections, car parks) over the rest of the 4-s',
          'horizon. Not part of the pre-registered labels.', '',
          'The last column counts failures whose off-road corners, at the first off-road step, all lie',
          'inside a mapped lane or lane-connector polygon that the DAC polygons do not cover.', '',
          '| stratum | DAC-primary cases | median distance to logged path | median max corner outside | '
          'max corner < 0.1 m outside | max corner < 0.5 m outside | off-road corner on a mapped lane |',
          '|---|---:|---:|---:|---:|---:|---:|']

    def median(v):
        v = sorted(v)
        return float('nan') if not v else (v[len(v) // 2] if len(v) % 2 else 0.5 * (v[len(v) // 2 - 1] + v[len(v) // 2]))

    for s in strata + ['all']:
        sel = [x for x in cases if (s == 'all' or x['case']['stratum'] == s) and x['primary'] == 'dac'
               and x['label'] not in ('unreproduced', 'fallback')]
        lat = [x['failing']['dac_lateral_m'] for x in sel]
        out_ = [x['failing']['dac_corner_outside_max_m'] for x in sel]
        on_lane = sum(1 for x in sel if x['failing'].get('dac_corner_on_lane'))
        L.append('| %s | %d | %.2f m | %.2f m | %d of %d | %d of %d | %d of %d |' % (
            s, len(sel), median(lat), median(out_), sum(1 for v in out_ if v < 0.1), len(sel),
            sum(1 for v in out_ if v < 0.5), len(sel), on_lane, len(sel)))
    L += ['', '## Box-induced failures (S3): how the boxes model plan differs from the map-only plan', '',
          '| difference | cases |', '|---|---:|']
    s3 = [x for x in cases if x['case']['stratum'] == 'S3' and x['label'] != 'unreproduced']
    for d in ('slower', 'faster', 'lateral', 'similar'):
        L.append('| %s | %d |' % (d, sum(1 for x in s3 if x['s3'] == d)))
    L.append('')
    return L


def pick_gallery(cases):
    """Per stratum two cases of the modal label, then one of the runner-up label in S2 and S3."""
    picked = []
    for sid, _label, _f, _c in core.STRATA:
        pool = [x for x in cases if x['case']['stratum'] == sid and x['label'] not in ('unreproduced', 'fallback')]
        counts = {}
        for x in pool:
            counts[x['label']] = counts.get(x['label'], 0) + 1
        ranked = sorted(counts, key=lambda lab: (-counts[lab], lab))
        if not ranked:
            continue
        modal = sorted((x for x in pool if x['label'] == ranked[0]), key=lambda x: x['case']['hash'])
        picked += [dict(x, why='%s: most common label' % sid) for x in modal[:2]]
        if sid in ('S2', 'S3') and len(ranked) > 1:
            second = sorted((x for x in pool if x['label'] == ranked[1]), key=lambda x: x['case']['hash'])
            picked += [dict(x, why='%s: second most common label' % sid) for x in second[:1]]
    return picked


def gallery_md(gallery):
    L = ['## Gallery', '',
         'Selection rule (pre-registered): per stratum the two reproduced cases with the most common',
         'label, plus one case of the second most common label in S2 and S3, in hash order. Map',
         'polygons, route lanes, centerline and agents come from the metric cache (map radius 50 m).', '']
    for i, g in enumerate(gallery, 1):
        c, f, ft = g['case'], g['failing'], g['feat']
        name = g['png']
        L += ['### %d. %s, `%s`: %s' % (i, c['stratum'], c['failing_run'], LABEL_TEXT[g['label']]), '',
              '![%s](gallery/%s)' % (LABEL_TEXT[g['label']], name), '',
              '`%s`, %s, %s m/s, command %s, lead %s, junction %s, curvature %s, stop/go %s. %s PDMS %.3f '
              '(NC %.1f, DAC %.0f, TTC %.0f, EP %.2f, C %.0f)%s. Selected as %s.' % (
                  c['token'], ft['city'], ft['speed_mps'], ft['command'], ft['lead'], ft['junction'], ft['curvature'],
                  ft['stop_go'], c['failing_run'], f['pdms'], f['nc'], f['dac'], f['ttc'], f['ep'], f['c'],
                  '' if not g['contrast'] else '; %s PDMS %.3f' % (c['contrast_run'], g['contrast']['pdms']),
                  g['why']), '']
    return L


def draw_case(ws, g, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as Patch
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    c, f, k = g['case'], g['failing'], g['contrast']
    mc = load_metric_cache(ws, c['log'], c['token'])
    rear = mc.ego_state.rear_axle
    ox, oy, oh = rear.x, rear.y, rear.heading
    cs, sn = math.cos(-oh), math.sin(-oh)

    def loc(x, y):
        dx, dy = x - ox, y - oy
        return cs * dx - sn * dy, sn * dx + cs * dy

    def ring(poly):
        geom = poly if poly.geom_type == 'Polygon' else max(poly.geoms, key=lambda p: p.area)
        return [loc(x, y) for x, y in geom.exterior.coords]

    ink, muted, plan_c, contrast_c, obj_c = '#0b0b0b', '#8a8984', '#eb6834', '#2a78d6', '#1baf7a'
    fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=130)
    fig.patch.set_facecolor('#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    dmap = mc.drivable_area_map
    route = set(mc.route_lane_ids)
    # grey = the polygons the drivable-area check uses (roadblocks, intersections, car parks);
    # lanes are outlines only, on-route lanes in blue, so a gap in the drivable area stays visible
    for geom, tok, kind in zip(dmap._geometries, dmap.tokens, dmap.map_types):
        if kind in (SemanticMapLayer.ROADBLOCK, SemanticMapLayer.INTERSECTION, SemanticMapLayer.CARPARK_AREA,
                    SemanticMapLayer.DRIVABLE_AREA):
            ax.add_patch(Patch(ring(geom), closed=True, facecolor='#dddcd6', edgecolor='none', zorder=1))
    for geom, tok, kind in zip(dmap._geometries, dmap.tokens, dmap.map_types):
        if kind in (SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR):
            on = tok in route
            ax.add_patch(Patch(ring(geom), closed=True, facecolor='#cde2fb' if on else 'none', alpha=0.45 if on else 1,
                               edgecolor='#2a78d6' if on else '#b9b8b2', linewidth=0.6 if on else 0.4,
                               zorder=2 if on else 3))
    cl = [loc(p.x, p.y) for p in mc.centerline.discrete_path]
    ax.plot([p[0] for p in cl], [p[1] for p in cl], color=muted, linewidth=1.0, linestyle=(0, (4, 3)), zorder=4,
            label='route centerline (scorer)')
    obs = mc.observation
    step = f.get('nc_step') if f.get('nc_step') is not None else (
        f.get('dac_step') if f.get('dac_step') is not None else f.get('ttc_step'))
    focus = f.get('nc_object_token') or f.get('ttc_object_token')
    for tok in obs[0].tokens:
        ax.add_patch(Patch(ring(obs[0][tok]), closed=True, facecolor='none', edgecolor=muted, linewidth=0.8, zorder=5))
    if focus is not None and step is not None:
        when = int(step)
        for t, alpha in ((0, 0.5), (when, 1.0)):
            if focus in obs[t].tokens:
                ax.add_patch(Patch(ring(obs[t][focus]), closed=True, facecolor=obj_c, alpha=0.35 * alpha,
                                   edgecolor=obj_c, linewidth=1.4, zorder=6))
        if focus in obs[when].tokens:
            cx, cy = loc(obs[when][focus].centroid.x, obs[when][focus].centroid.y)
            ax.annotate('object at %.1f s' % (when / 10.0), (cx, cy), xytext=(6, 6), textcoords='offset points',
                        fontsize=7, color=ink, zorder=9)
    lg = [(0.0, 0.0)] + [(p[0], p[1]) for p in f['logged']]
    ax.plot([p[0] for p in lg], [p[1] for p in lg], color=ink, linewidth=2, marker='o', markersize=3.5, zorder=8,
            label='logged (human)')
    if k is not None:
        kp = [(0.0, 0.0)] + [(p[0], p[1]) for p in k['plan']]
        ax.plot([p[0] for p in kp], [p[1] for p in kp], color=contrast_c, linewidth=2, marker='s', markersize=3.5,
                zorder=7, label='%s (PDMS %.2f)' % (c['contrast_run'], k['pdms']))
    sim = f['sim_xy']
    ax.plot([p[0] for p in sim], [p[1] for p in sim], color=plan_c, linewidth=2, zorder=8,
            label='%s, simulated (PDMS %.2f)' % (c['failing_run'], f['pdms']))
    ax.plot([p[0] for p in f['plan']], [p[1] for p in f['plan']], linestyle='none', marker='D', markersize=3.5,
            color=plan_c, zorder=8)
    from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
    vp = get_pacifica_parameters()
    front, rear_len, half = vp.front_length, vp.rear_length, vp.half_width
    ego = [(-rear_len, half), (front, half), (front, -half), (-rear_len, -half)]
    ax.add_patch(Patch(ego, closed=True, facecolor='none', edgecolor=ink, linewidth=1.4, zorder=9))
    if step is not None and step < len(sim):
        sx, sy = sim[int(step)]
        sh = f['sim_heading'][int(step)]
        foot = [(sx + math.cos(sh) * a - math.sin(sh) * b, sy + math.sin(sh) * a + math.cos(sh) * b) for a, b in ego]
        ax.add_patch(Patch(foot, closed=True, facecolor=plan_c, alpha=0.25, edgecolor=plan_c, linewidth=1.4,
                           zorder=9, label='footprint at first failure (%.1f s)' % (step / 10.0)))
        ax.plot([sx], [sy], marker='X', markersize=10, color=plan_c, markeredgecolor=ink, linestyle='none', zorder=10)
    pts = lg + sim + ([(p[0], p[1]) for p in k['plan']] if k else [])
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    span = max(30.0, max(xs) - min(xs) + 16, max(ys) - min(ys) + 16)
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    ax.set_xlim(cx - span / 2, cx + span / 2)
    ax.set_ylim(cy - span / 2, cy + span / 2)
    ax.set_aspect('equal')
    ax.grid(color='#ecebe7', linewidth=0.5, zorder=0)
    ax.tick_params(labelsize=7, colors='#52514e')
    for s in ax.spines.values():
        s.set_color('#d7d6d1')
    ax.set_xlabel('x forward at t0 [m]', fontsize=8, color='#52514e')
    ax.set_ylabel('y left [m]', fontsize=8, color='#52514e')
    ax.set_title('%s  %s  %s\n%s' % (c['stratum'], c['failing_run'], c['token'], LABEL_TEXT[g['label']]),
                 fontsize=9, color=ink, loc='left')
    ax.legend(loc='lower left', fontsize=6.5, frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(folder / g['png'], facecolor=fig.get_facecolor())
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('command', choices=['select', 'score', 'report'])
    ap.add_argument('--out', required=True)
    ap.add_argument('--workspace', default=os.environ.get('NAVSIM_WORKSPACE', 'E:/navsim_workspace'))
    ap.add_argument('--source', choices=['archive', 'scene'], default='archive')
    ap.add_argument('--city', default=None, help='scene source: only cases of this city')
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--min-free-mb', type=float, default=2500.0)
    ap.add_argument('--no-plots', action='store_true')
    args = ap.parse_args()
    if not 1 <= args.threads <= 4:
        raise SystemExit('--threads must be 1..4')
    {'select': cmd_select, 'score': cmd_score, 'report': cmd_report}[args.command](args)


if __name__ == '__main__':
    main()
