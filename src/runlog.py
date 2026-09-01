"""
Run logging: one directory per run under out/, created if it is not there.

Everything a run produces lands in out/<run_name>/:
    config.json      the run's settings, verbatim, so a plot can be traced to a run
    manifest.json    word order, layer indices, read positions -- written once
    metrics.jsonl    one line per logged event; the only thing plotting reads
    acts/            step_<step>.npz, the harvested activations

Two rules keep this usable. Activations are the bulk and are written per harvest
step so a dead Colab session loses one step, not the run. Everything else flows
through metrics.jsonl, so re-analysis and plotting never need to load a model or
touch acts/.

This module deliberately does not import the standard library `logging`.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

Repo_Root = Path(__file__).resolve().parent.parent
Out_Root = Repo_Root / "out"


def _jsonable(value):
    """Make numpy scalars and arrays survive json.dumps."""
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


class RunLogger:
    """
    Writes one run's outputs, and reads them back for analysis.

    Args:
        run_name (str): directory name under out/, e.g. "qwen3_seed0".
        config (dict | None): the run settings, dumped to config.json.
        out_root (str | Path | None): where runs live; defaults to <repo>/out.
        resume (bool): reuse an existing directory instead of failing on it.
    """

    def __init__(self, run_name, config = None, out_root = None, resume = True):
        self.root = Path(out_root or Out_Root) / run_name
        if self.root.exists() and not resume:
            raise FileExistsError(f"{self.root} already exists; pass resume = True to append to it")
        self.acts_dir = self.root / "acts"
        self.acts_dir.mkdir(parents = True, exist_ok = True)
        self.metrics_path = self.root / "metrics.jsonl"
        self.t0 = time.time()
        if config is not None:
            self.write_json("config.json", config)
        self.say(f"run directory {self.root}")

    # ------------------------------------------------------------------ writing

    def write_json(self, name, payload):
        """
        Dump a json file into the run directory.
        Args:
            name (str): file name, e.g. "manifest.json".
            payload (dict): contents; numpy scalars and arrays are converted.
        """
        (self.root / name).write_text(json.dumps(_jsonable(payload), indent = 2))

    def metric(self, step, kind, **values):
        """
        Append one event to metrics.jsonl.
        Args:
            step (int): optimizer step the event belongs to.
            kind (str): what kind of event, e.g. "train", "behaviour", "harvest".
            **values: any json-able numbers.
        """
        row = {"step": int(step), "kind": kind, "elapsed": round(time.time() - self.t0, 2)}
        row.update(_jsonable(values))
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def save_acts(self, step, arrays):
        """
        Write one harvest step, one array per (position, layer).

        The split matters. npz members are read lazily, so storing per layer lets the
        analysis pull a single layer's 9 MB slab out of a 260 MB step instead of
        reading the whole thing. The analysis walks layer by layer, so the flat
        layout would mean re-reading every step in full once per layer.

        Stored as float16, which is the storage format and not the compute format:
        the geometry promotes to float64 before any SVD.

        Args:
            step (int): optimizer step.
            arrays (dict[str, ndarray]): read position -> (n_layers, n_items, d_model).
        Returns:
            Path: the file written.
        """
        path = self.acts_dir / f"step_{int(step):06d}.npz"
        flat = {}
        for position, block in arrays.items():
            block = np.asarray(block, dtype = np.float16)
            for layer in range(block.shape[0]):
                flat[f"{position}_L{layer:03d}"] = block[layer]
        np.savez(path, **flat)
        return path

    def say(self, message):
        """Print a progress line with elapsed time, for watching a Colab run."""
        mins, secs = divmod(time.time() - self.t0, 60)
        print(f"[{int(mins):3d}:{int(secs):02d}] {message}", flush = True)

    # ------------------------------------------------------------------ reading

    def steps(self):
        """
        Harvest steps present on disk, ascending.
        Returns:
            list[int]: the steps saved by save_acts.
        """
        return sorted(int(p.stem.split("_")[1]) for p in self.acts_dir.glob("step_*.npz"))

    def read_json(self, name):
        """
        Read a json file from the run directory.
        Args:
            name (str): file name.
        Returns:
            dict: the contents.
        """
        return json.loads((self.root / name).read_text())

    def n_layers(self, position = "word"):
        """
        How many layers were stored, read off the first step's keys.
        Args:
            position (str): the read position key.
        Returns:
            int: the number of layer slabs.
        """
        with np.load(self.acts_dir / f"step_{self.steps()[0]:06d}.npz") as z:
            return sum(1 for k in z.files if k.startswith(position + "_L"))

    def load_acts(self, step, position, layer = None):
        """
        Load one harvest step for one read position.
        Args:
            step (int): the step to load.
            position (str): the read position key, e.g. "word" or "last".
            layer (int | None): a single layer, read on its own. None stacks all of
                them, which is the expensive path.
        Returns:
            ndarray: (n_items, d_model) for one layer, else (n_layers, n_items, d_model).
        """
        with np.load(self.acts_dir / f"step_{int(step):06d}.npz") as z:
            if layer is not None:
                return z[f"{position}_L{int(layer):03d}"].astype(np.float32)
            keys = sorted(k for k in z.files if k.startswith(position + "_L"))
            return np.stack([z[k] for k in keys]).astype(np.float32)

    def load_trajectory(self, position, layer, steps = None):
        """
        Stack one layer across checkpoints into the (T, N, D) shape the geometry wants.
        Args:
            position (str): the read position key.
            layer (int): index into the stored layer axis.
            steps (list[int] | None): which steps; defaults to every step on disk.
        Returns:
            traj (ndarray): (T, n_items, d_model) float32.
            steps (list[int]): the steps that were stacked, in order.
        """
        steps = list(steps) if steps is not None else self.steps()
        traj = np.stack([self.load_acts(s, position, layer = layer) for s in steps])
        return traj, steps

    def metrics(self, kind = None):
        """
        Read metrics.jsonl back.
        Args:
            kind (str | None): keep only this event kind.
        Returns:
            list[dict]: the rows, in write order.
        """
        if not self.metrics_path.exists():
            return []
        rows = [json.loads(line) for line in self.metrics_path.read_text().splitlines() if line.strip()]
        return rows if kind is None else [r for r in rows if r["kind"] == kind]
