# Running NAVSIM v1.1 on Windows, on a CPU

Everything in this repository was produced on Windows 11 with no NVIDIA GPU
(20 threads, 16 GB RAM). None of the changes below touches the PDM score or any
upstream agent; they are packaging, portability and performance fixes.

## Four portability fixes

**1. UTF-8 during builds.** `pip`/`uv` builds of `control` and `nuplan-devkit`
read `setup.py` with the system ANSI codepage and die on non-ASCII bytes
(`UnicodeDecodeError: 'cp950' codec ...`). Set `PYTHONUTF8=1` for the install.

**2. `nuplan-devkit` packaging scope.** Its `setup.py` calls
`find_packages(script_folder, exclude=[...])`, which on Windows also picks up
`docs` and then fails with *"could not create 'build\lib\docs'"*. Restrict it:

```python
packages=setuptools.find_packages(script_folder, include=['nuplan', 'nuplan.*'], exclude=[...]),
```

Install the devkit from a local clone with `--no-deps`, and the requirements
separately — the pinned `greenlet` in `sqlalchemy==1.4.27` has no cp39 Windows
wheel and tries to build from source, so pin `greenlet<3.2`.

**3. POSIX file locks.** `nuplan/database/maps_db/gpkg_mapsdb.py` imports `fcntl`
at module scope and calls `flock`. Windows has neither. A try/except shim that
sets `fcntl = None` and skips the two `flock` calls is enough for single-process
map reads:

```python
try:
    import fcntl  # POSIX only
except ImportError:
    fcntl = None
...
(fcntl.flock(fd, fcntl.LOCK_EX) if fcntl is not None else None)
```

**4. Metric-cache path splitting.** `navsim/common/dataloader.py` recovers the
token from a cache path with `cache_path.split("/")[-2]`. On Windows the
metadata CSV holds backslashes, so every token comes out wrong and the loader
silently finds zero cached scenes. Normalise first:

```python
metric_cache_dict = {cache_path.replace("\\", "/").split("/")[-2]: cache_path for cache_path in cache_paths}
```

## Two CPU performance traps

Neither is Windows-specific, but both cost hours here and neither raises an
error — they just make the job look hung.

**5. `numpy.load` on an `.npz` is lazy.** Every `d["ego"][i]` in a loop
re-decompresses the whole `ego` array. Assembling one feature matrix row by row
from a 65 MB archive took long enough to look like a deadlock; materialising each
array once and concatenating takes about a second:

```python
with np.load(path) as z:
    d = {k: z[k] for k in wanted}          # decompress once
X = np.concatenate([d["ego"], d["cl"].reshape(n, -1)], axis=1)
```

**6. Subnormal floats make weight decay 20-50× slower.** Weight decay pushes many
weights below the smallest normal float; x86 handles subnormals in microcode, so
epochs get progressively slower — 8 s, then 300 s, on identical work. The runs
with `wd` looked like a hang and were in fact 2 hours each. One line fixes it,
in training *and* wherever those checkpoints are loaded for inference:

```python
torch.set_flush_denormal(True)
```

## Long jobs

Detached background processes started from an editor or assistant session are
killed by a Windows **logoff** (`0xC000026B`, "DLL initialisation failed because
the process is terminating") and by a restart of the parent application. The
scoring sweep is ~7 hours: run it from a terminal you keep open, or from a
scheduled task, and do not log off. Locking the screen is fine.
