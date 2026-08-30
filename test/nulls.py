"""
What each statistic reads on data with no effect in it.

Every number in analyse_v1 is a ratio with a floor, and the floor moves with the
pool size. cloud_separation between two random 100-word clouds is 0.14 before
anything has been learned, so a separation of 0.12 is not a small effect, it is
below chance. This script prints the floor for the pool sizes actually in use.

Two nulls, and the second is the one to trust:

    analytic   isotropic gaussian clouds. A lower bound: it assumes the words
               move independently, which they do not, so it understates the
               noise on anything computed after gauging.
    empirical  the BACKGROUND words from a real run, split in half. Carries the
               real drift, the real anisotropy and the real gauge residual, so
               it is the honest null for a MEM-vs-FILL contrast.

    python3 test/nulls.py                        # analytic only
    python3 test/nulls.py --run qwen3_seed0      # both, needs a finished run
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _utils as u  # noqa: E402


def analytic(n_words, d_model, trials = 200, seed = 0):
    """
    Floors from isotropic gaussian clouds: no shared direction, no membership.

    Args:
        n_words (int): words per cloud, i.e. n_mem.
        d_model (int): model width.
        trials (int): monte carlo repeats.
        seed (int): rng seed.
    Returns:
        dict: statistic -> (mean, sd) of the null.
    """
    rng = np.random.default_rng(seed)
    coh = np.array([u.top_fraction(u.spectrum(
        (lambda V: V / np.linalg.norm(V, axis = 1, keepdims = True))(
            rng.standard_normal((n_words, d_model)))))
        for _ in range(trials)])
    dcoh = coh[: trials // 2] - coh[trials // 2:]
    sep = np.array([u.cloud_separation(rng.standard_normal((n_words, d_model)),
                                       rng.standard_normal((n_words, d_model)))
                    for _ in range(trials)])
    return {"coherence": (coh.mean(), coh.std(ddof = 1)),
            "coherence_diff": (dcoh.mean(), dcoh.std(ddof = 1)),
            "separation": (sep.mean(), sep.std(ddof = 1))}


def empirical(run_name, position = "word", out_root = None, layers = None,
              rotate_dim = 256, splits = 20, seed = 0):
    """
    The honest null: BACKGROUND split in half and run through the real pipeline.

    Both halves are the same population by construction, so every MEM-vs-FILL
    statistic computed on them is measuring nothing. Whatever it reads is the
    noise a real contrast has to clear at that layer.

    Args:
        run_name (str): directory under out/.
        position (str): "word" or "last".
        out_root (str | None): root for out/.
        layers (list[int] | None): layers to check; None does every fourth.
        rotate_dim (int | None): passed to the gauge, match the real analysis.
        splits (int): random half-splits per layer.
        seed (int): rng seed.
    Returns:
        dict: layer -> {"separation": (mean, sd), "coherence_diff": (mean, sd)}
    """
    from src.logging import RunLogger
    from src.harvest import items_from_manifest

    logger = RunLogger(run_name, out_root = out_root)
    steps = logger.steps()
    items = items_from_manifest(logger.read_json("manifest.json"))
    bg_rows = [i for i, it in enumerate(items) if it.group == "BACKGROUND"]
    n_layers = logger.n_layers(position)
    layers = layers if layers is not None else list(range(0, n_layers, 4))

    rng = np.random.default_rng(seed)
    out = {}
    for layer in layers:
        raw = np.stack([logger.load_acts(st, position, layer) for st in steps])  # (T, n_items, D)
        bg = raw[:, bg_rows, :]
        seps, dcohs = [], []
        for _ in range(splits):
            perm = rng.permutation(len(bg_rows))
            a, b = perm[: len(perm) // 4], perm[len(perm) // 4: len(perm) // 2]
            rest = perm[len(perm) // 2:]                     # gauge fitted on the rest only
            gauged, _ = u.gauge_all(bg[:, rest, :],
                                    {"a": bg[:, a, :], "b": bg[:, b, :]},
                                    rotate_dim = rotate_dim)
            seps.append(u.cloud_separation(gauged["a"][-1], gauged["b"][-1]))
            dcohs.append(np.nanmean(u.velocity_coherence(gauged["a"])["top"])
                         - np.nanmean(u.velocity_coherence(gauged["b"])["top"]))
        out[layer] = {"separation": (float(np.mean(seps)), float(np.std(seps, ddof = 1))),
                      "coherence_diff": (float(np.mean(dcohs)), float(np.std(dcohs, ddof = 1)))}
        print(f"  layer {layer:3d}  sep {out[layer]['separation'][0]:.4f} "
              f"+-{out[layer]['separation'][1]:.4f}   "
              f"dcoh {out[layer]['coherence_diff'][0]:+.5f} "
              f"+-{out[layer]['coherence_diff'][1]:.5f}", flush = True)
    return out


def main(argv = None):
    ap = argparse.ArgumentParser(description = "Null floors for the analysis statistics.")
    ap.add_argument("--run", default = None, help = "a finished run, for the empirical null")
    ap.add_argument("--position", default = "word", choices = ("word", "last"))
    ap.add_argument("--out", default = None)
    ap.add_argument("--d-model", type = int, default = 2048)
    ap.add_argument("--n-words", type = int, nargs = "+", default = [100, 200, 400, 1000])
    args = ap.parse_args(argv)

    print(f"analytic null, d_model {args.d_model}\n")
    print(f"{'n_words':>8}  {'coherence':>18}  {'coh MEM-FILL sd':>16}  {'separation':>18}")
    for n in args.n_words:
        r = analytic(n, args.d_model)
        print(f"{n:8d}  {r['coherence'][0]:8.4f} +-{r['coherence'][1]:.4f}  "
              f"{r['coherence_diff'][1]:16.5f}  "
              f"{r['separation'][0]:8.4f} +-{r['separation'][1]:.4f}")
    print("\nread: a separation below its row's value is below chance, not a small effect.")
    print("      the analytic null assumes independent words -- it is a LOWER bound on noise.")

    if args.run:
        print(f"\nempirical null from {args.run} (BACKGROUND split in half):")
        empirical(args.run, position = args.position, out_root = args.out)
        print("\nthis is the number a real MEM-vs-FILL contrast has to clear.")


if __name__ == "__main__":
    main()
