# -*- coding: utf-8 -*-
"""Shared loading and bootstrap code for the ladder analyses.

Two layouts are understood:

* the NAVSIM workspace, ``<exp>/<agent_subdir>/<run>/<timestamp>.csv`` plus
  ``<exp>/metric_cache/<log>/unknown/<token>/`` and the upstream
  ``available_mini_logs.yaml`` — what ``run_pdm_score.py`` writes;
* this repository's ``results/per_scene/``, where the canonical CSV of every
  agent is committed as ``<agent_subdir>.csv`` next to ``token_log.csv`` and a
  copy of ``available_mini_logs.yaml`` (see ``results/per_scene/MANIFEST.md``).

``resolve_csv_root`` picks the workspace when ``NAVSIM_EXP_ROOT`` or a sibling
``exp/`` exists and the committed folder otherwise, so every report in
``results/`` can be regenerated from a plain clone with no dataset.
"""
import csv
import glob
import io
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
PER_SCENE = os.path.join(REPO, 'results', 'per_scene')
B = 10000
SEED = 12345


def resolve_csv_root(explicit=None):
    """Workspace ``exp/`` if one is configured or present, else the committed CSVs."""
    if explicit:
        return os.path.abspath(explicit)
    env = os.environ.get('NAVSIM_EXP_ROOT')
    if env and os.path.isdir(env):
        return env
    sibling = os.path.join(HERE, 'exp')
    if os.path.isdir(sibling):
        return sibling
    return PER_SCENE


def is_per_scene(root):
    return os.path.isfile(os.path.join(root, 'token_log.csv'))


def csv_path(root, subdir):
    """The canonical CSV of one agent: the committed file, or the FIRST run in
    the workspace (matching pdm_report.md; privileged_brake_mini has two runs)."""
    flat = os.path.join(root, subdir + '.csv')
    if os.path.isfile(flat):
        return flat
    files = sorted(glob.glob(os.path.join(root, subdir, '*', '*.csv')))
    if not files:
        raise SystemExit('no csv for ' + subdir + ' under ' + root)
    return files[0]


def load_scores(root, subdir):
    """token -> {'score', 'dac'} for one agent, skipping the trailing average row."""
    rows = {}
    with io.open(csv_path(root, subdir), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            tok = (r.get('token') or '').strip()
            if not tok or tok.lower() == 'average':
                continue
            try:
                rows[tok] = {'score': float(r['score']),
                             'dac': float(r['drivable_area_compliance'])}
            except (ValueError, KeyError):
                continue
    return rows


def load_rows(root, subdir):
    """Every metric column of one agent as floats, keyed by token."""
    rows = {}
    with io.open(csv_path(root, subdir), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            tok = (r.get('token') or '').strip()
            if not tok or tok.lower() == 'average':
                continue
            try:
                rows[tok] = {k: float(v) for k, v in r.items()
                             if k and k not in ('', 'token', 'valid') and v not in (None, '')}
            except ValueError:
                continue
    return rows


def split_logs(root):
    """(train_logs, val_logs) from available_mini_logs.yaml — committed copy or upstream file."""
    candidates = [os.path.join(root, 'available_mini_logs.yaml'),
                  os.path.join(os.path.dirname(root), 'navsim', 'navsim', 'planning', 'script',
                               'config', 'training', 'available_mini_logs.yaml')]
    path = next((p for p in candidates if os.path.isfile(p)), None)
    if path is None:
        raise SystemExit('available_mini_logs.yaml not found next to ' + root)
    train, val, sec = set(), set(), None
    for raw in io.open(path, encoding='utf-8').read().splitlines():
        line = raw.strip()
        if line.startswith('train_logs:'):
            sec = 't'
            continue
        if line.startswith('val_logs:'):
            sec = 'v'
            continue
        if line.startswith('- ') and sec:
            (train if sec == 't' else val).add(line[2:].strip().strip('"').strip("'"))
    return train, val


def token_to_log(root):
    """token -> log name, from token_log.csv or the metric-cache directory layout."""
    flat = os.path.join(root, 'token_log.csv')
    if os.path.isfile(flat):
        with io.open(flat, encoding='utf-8') as f:
            return {r['token']: r['log'] for r in csv.DictReader(f)}
    cache = os.path.join(root, 'metric_cache')
    m = {}
    for log in os.listdir(cache):
        p = os.path.join(cache, log, 'unknown')
        if os.path.isdir(p):
            for tok in os.listdir(p):
                m[tok] = log
    return m


def bootstrap(diffs, seed=SEED, b=B):
    """Mean of paired differences and its percentile-bootstrap 95% CI.

    Resamples the ``len(diffs)`` scenes with replacement ``b`` times with a
    fixed seed, so a rerun on the same CSVs reproduces every interval in
    ``results/`` to the last digit.
    """
    n = len(diffs)
    mean = sum(diffs) / n
    rnd = random.Random(seed)
    means = []
    for _ in range(b):
        s = 0.0
        for _ in range(n):
            s += diffs[rnd.randrange(n)]
        means.append(s / n)
    means.sort()
    return mean, means[int(0.025 * b)], means[int(0.975 * b)]
