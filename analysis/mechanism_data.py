"""Frozen session-disjoint splits and current-input-only object controls."""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SCHEMA = "dynamic-objects-v2"
CONFIGS = {"M0": ("none", False), "M1": ("concat", False), "M2": ("concat", True),
           "M3": ("lead", False), "M4": ("set", False), "M5": ("set", True)}


def digest(path):
    result = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def session_id(log):
    result = re.sub(r"_\d+_\d+$", "", str(log))
    if result == str(log):
        raise ValueError("Unrecognized segmented log: " + str(log))
    return result


def load_features(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def freeze_protocol(train, test, train_path, test_path, seed=20261009):
    sessions = np.array([session_id(log) for log in train["log_name"]])
    test_sessions = {session_id(log) for log in test["log_name"]}
    if set(sessions) & test_sessions or set(train["token"]) & set(test["token"]):
        raise ValueError("Train/test overlap")
    city_sessions = defaultdict(set)
    for city, session in zip(train["map_name"], sessions):
        city_sessions[str(city)].add(str(session))
    rng = np.random.RandomState(seed)
    split_sessions = {key: [] for key in ("train", "validation", "confirmation")}
    ordered_train = {}
    for city, values in sorted(city_sessions.items()):
        order = np.array(sorted(values))
        rng.shuffle(order)
        holdout = max(1, int(round(.1*len(order))))
        if 2*holdout >= len(order):
            raise ValueError("Too few independent sessions in " + city)
        split_sessions["validation"].extend(order[:holdout].tolist())
        split_sessions["confirmation"].extend(order[holdout:2*holdout].tolist())
        split_sessions["train"].extend(order[2*holdout:].tolist())
        ordered_train[city] = order[2*holdout:].tolist()
    indices = {key: np.flatnonzero(np.isin(sessions, values)) for key, values in split_sessions.items()}
    fractions = {}
    for fraction in (.1, .3, 1.):
        chosen = []
        for city, order in sorted(ordered_train.items()):
            chosen.extend(order[:max(1, int(np.ceil(fraction*len(order))))])
        idx = np.flatnonzero(np.isin(sessions, chosen))
        fractions[str(fraction)] = {"sessions": sorted(chosen), "n_scenes": len(idx),
                                   "n_logs": len(np.unique(train["log_name"][idx]))}
    val = indices["validation"]
    thresholds = {
        "speed_low_mps": float(np.quantile(train["speed"][val], .33)),
        "speed_high_mps": float(np.quantile(train["speed"][val], .67)),
        "close_lead_gap_m": 15., "near_object_distance_m": 20.,
        "crossing_horizon_s": 3., "crossing_corridor_half_width_m": 3.,
        "provenance": "Speed quantiles fitted on validation inputs only; physical thresholds frozen a priori, not fit to test outcomes."}
    protocol = {"schema": SCHEMA, "seed": seed, "split_unit": "original driving session",
                "session_rule": "remove final _start_end numeric suffix from log_name",
                "sources": {"navtrain": {"path": str(train_path), "sha256": digest(train_path)},
                            "navtest": {"path": str(test_path), "sha256": digest(test_path)}},
                "split_sessions": {key: sorted(value) for key, value in split_sessions.items()},
                "split_counts": {key: {"scenes": len(idx), "sessions": len(split_sessions[key]),
                                        "logs": len(np.unique(train["log_name"][idx]))} for key, idx in indices.items()},
                "fractions": fractions, "thresholds": thresholds,
                "primary_contrasts": ["M1-M0", "M4-M1", "M4-M5"],
                "multiplicity": "Bonferroni simultaneous percentile CIs for the three primary contrasts; exploratory subgroups unadjusted",
                "selection": "last checkpoint, fixed steps; validation diagnostics only; confirmation outcomes accessed after core model manifest is complete",
                "disclosure": "navtest is exploratory. New confirmation is excluded from all v2 training/normalization, but historical v1 models saw these data.",
                "training": {"hidden": 512, "layers": 3, "object_hidden": 32, "batch_size": 256,
                             "full_data_epochs_equivalent": 40, "lr": .001, "weight_decay": 0., "dropout": 0.,
                             "loss": "mean absolute error over 8*(x metres,y metres,heading radians); report components separately",
                             "seeds": [0, 1, 2]}}
    return protocol


def split_indices(data, protocol, split, fraction=1.):
    sessions = np.array([session_id(log) for log in data["log_name"]])
    if split == "navtest":
        return np.arange(len(sessions))
    values = (protocol["fractions"][str(float(fraction))]["sessions"] if split == "train"
              else protocol["split_sessions"][split])
    return np.flatnonzero(np.isin(sessions, values))


def matched_donors(data, indices, seed):
    """Same partition, different original session, matched only on current inputs.

    Fallback levels relax object count, then speed, then command. Every level is
    recorded; there is no future-target or held-out-partition access.
    """
    indices = np.asarray(indices, int)
    if len(indices) == 0:
        raise ValueError("Empty donor partition")
    speed = np.digitize(data["speed"][indices], [2., 5., 10., 20.])
    command = data["cmd"][indices]
    count = np.digitize(data["agents"][indices, :, 10].sum(1), [1, 3, 6])
    session = np.array([session_id(log) for log in data["log_name"][indices]])
    keys = [[(int(c), int(v), int(n)) for c, v, n in zip(command, speed, count)],
            [(int(c), int(v)) for c, v in zip(command, speed)],
            [(int(c),) for c in command], [()] * len(indices)]
    # Index donors by session, preventing one long session from monopolising candidates.
    tables = []
    for level_keys in keys:
        table = defaultdict(lambda: defaultdict(list))
        for pos, key in enumerate(level_keys):
            table[key][str(session[pos])].append(int(indices[pos]))
        tables.append(table)
    rng = np.random.RandomState(seed)
    donors, levels = [], []
    for pos in range(len(indices)):
        for level in range(4):
            pool = tables[level][keys[level][pos]]
            other = sorted(s for s in pool if s != session[pos])
            if other:
                selected_session = other[int(rng.randint(len(other)))]
                candidates = pool[selected_session]
                donors.append(candidates[int(rng.randint(len(candidates)))])
                levels.append(level)
                break
        else:
            raise ValueError("No cross-session donor; at least two sessions required")
    return np.asarray(donors, np.int64), {str(k): int(v) for k, v in sorted(Counter(levels).items())}


def fit_normalizer(data, indices):
    idx = np.asarray(indices)
    context = np.concatenate([data["ego"][idx], data["cl"][idx], data["lead"][idx]], axis=1)
    mu, sd = context.mean(0), context.std(0)
    sd = np.where(sd < 1e-3, 1., sd)
    valid = data["agents"][idx, :, 10] > .5
    objects = data["agents"][idx, :, :10][valid]
    om, os = np.zeros(10, np.float32), np.ones(10, np.float32)
    if len(objects):
        om[:6], os[:6] = objects[:, :6].mean(0), objects[:, :6].std(0)
        os[os < 1e-3] = 1.
    return {"context_mu": mu.tolist(), "context_sd": sd.tolist(),
            "object_mu": om.tolist(), "object_sd": os.tolist(),
            "object_policy": "valid training objects only; normalize first six continuous channels; leave heading/type/mask unscaled; force padded values to zero"}


def transform(data, indices, norm, donors=None):
    idx = np.asarray(indices)
    context = np.concatenate([data["ego"][idx], data["cl"][idx], data["lead"][idx]], axis=1)
    context = (context-np.asarray(norm["context_mu"], np.float32))/np.asarray(norm["context_sd"], np.float32)
    objects = data["agents"][idx if donors is None else donors].copy()
    mask = objects[:, :, 10:11]
    objects[:, :, :10] = ((objects[:, :, :10]-np.asarray(norm["object_mu"], np.float32))
                          / np.asarray(norm["object_sd"], np.float32)) * mask
    return context.astype(np.float32), objects.astype(np.float32)


def subgroups(data, indices, thresholds):
    idx = np.asarray(indices)
    objects, speed = data["agents"][idx], data["speed"][idx]
    mask = objects[:, :, 10] > .5
    distance = np.linalg.norm(objects[:, :, :2], axis=2)
    near = (mask & (distance < thresholds["near_object_distance_m"])).any(1)
    # Straight corridor proxy, not map-aware interaction truth. Uses current absolute velocities.
    vy = objects[:, :, 3]
    time_to_axis = -objects[:, :, 1] / np.where(np.abs(vy) > .5, vy, 1.)
    cross_x = objects[:, :, 0] + objects[:, :, 2]*time_to_axis
    crossing = (mask & (np.abs(vy) > .5) & (time_to_axis > 0)
                & (time_to_axis < thresholds["crossing_horizon_s"])
                & (cross_x > 0) & (cross_x < 40)).any(1)
    return {"all": np.ones(len(idx), bool), "near_objects": near, "no_near_objects": ~near,
            "low_speed": speed < thresholds["speed_low_mps"],
            "high_speed": speed >= thresholds["speed_high_mps"],
            "turn_command": data["cmd"][idx] != 1, "straight_command": data["cmd"][idx] == 1,
            "close_lead": data["lead"][idx, 0] < thresholds["close_lead_gap_m"],
            "potential_crossing_proxy": crossing}
