"""
Read out/<run>/ back and produce the numbers and the figures.

Never loads a model or a tokenizer. Everything comes from acts/, manifest.json and
metrics.jsonl, so analysis reruns on a laptop while the GPU does something else.

    python3 experiments/analyse_v1.py --run qwen3_seed0 --position word

Order of reading the output. Figure 0 first, always: it is the gauge residual, and
if that is large there is no rigid frame and none of the rest is interpretable.
Then figure 1, the depth x time displacement map, which says where in the network
anything happened at all and therefore which layers the other figures are worth
looking at. Only then figures 2 to 5.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import _utils as u  # noqa: E402
from src.harvest import fold_by_word, items_from_manifest  # noqa: E402
from src.logging import RunLogger  # noqa: E402

Groups = ("MEM", "FILL", "BACKGROUND")


def layer_clouds(logger, items, position, layer, steps):
    """
    Build the (T, N, D) trajectory of each group at one layer.

    Averaging over harvest templates happens here, before anything else: the same
    word in two carrier sentences is one point for the purpose of watching it move.

    Args:
        logger (RunLogger): the run.
        items (list): records with .word and .group, from items_from_manifest.
        position (str): "word" or "last".
        layer (int): index into the stored layer axis.
        steps (list[int]): harvest steps, ascending.
    Returns:
        dict[str, ndarray]: group -> (T, n_words, d_model).
    """
    per_step = [logger.load_acts(s, position, layer = layer) for s in steps]
    clouds = {}
    for group in Groups:
        folded = [fold_by_word(a[None], items, group)[0][0] for a in per_step]
        clouds[group] = np.stack(folded)
    return clouds


def analyse_run(run_name, position = "word", out_root = None, k = 64, rotate = True,
                rotate_dim = 256, layers = None):
    """
    Every statistic, at every layer, for one read position.
    Args:
        run_name (str): directory under out/.
        position (str): "word" or "last".
        out_root (str | None): root for out/, if not the repo default.
        k (int): reference subspace size for the novel-fraction statistic.
        rotate (bool): fit the gauge rotation.
        rotate_dim (int | None): fit the rotation in this many principal directions.
        layers (list[int] | None): which layers to analyse; None does all of them.
    Returns:
        dict: {"steps", "layers", "position", "stats": {layer -> statistic -> list}}
    """
    logger = RunLogger(run_name, out_root = out_root)
    steps = logger.steps()
    if len(steps) < 3:
        raise SystemExit(f"run {run_name} has {len(steps)} harvest steps; need at least 3")
    items = items_from_manifest(logger.read_json("manifest.json"))
    layers = layers if layers is not None else list(range(logger.n_layers(position)))

    out = {"steps": steps, "layers": layers, "position": position, "stats": {}}
    for layer in layers:
        clouds = layer_clouds(logger, items, position, layer, steps)
        gauged, residual = u.gauge_all(clouds["BACKGROUND"],
                                       {"MEM": clouds["MEM"], "FILL": clouds["FILL"]},
                                       rotate = rotate, rotate_dim = rotate_dim)
        stats = u.summarise(gauged["MEM"], gauged["FILL"], gauged["background"][0], k = k)
        stats["gauge_residual"] = residual
        out["stats"][layer] = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                               for key, value in stats.items()}
        print(f"  layer {layer:3d}  residual_max {residual.max():.3f}  "
              f"disp {stats['displacement'][-1]:.3f}  "
              f"coh {np.nanmean(stats['coherence']):.3f} (fill {np.nanmean(stats['coherence_fill']):.3f})",
              flush = True)

    logger.write_json(f"stats_{position}.json", out)
    return out


def curve(stats, layers, key):
    """Stack one statistic into a (layer, step) array for a heatmap."""
    return np.array([stats[l][key] for l in layers], dtype = float)


def figures(run_name, analysis, out_root = None, focus = None):
    """
    Write the six figures into out/<run>/figs/.
    Args:
        run_name (str): directory under out/.
        analysis (dict): the return of analyse_run.
        out_root (str | None): root for out/.
        focus (int | None): the layer the line plots use; None picks the layer with
            the largest MEM-minus-FILL displacement, which is where anything happened.
    Returns:
        Path: the figures directory.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    logger = RunLogger(run_name, out_root = out_root)
    figs = logger.root / "figs"
    figs.mkdir(exist_ok = True)
    steps, layers, stats = analysis["steps"], analysis["layers"], analysis["stats"]
    position = analysis["position"]
    x, x_mid = np.array(steps), np.array(steps[1:])

    contrast = curve(stats, layers, "displacement") - curve(stats, layers, "displacement_fill")
    focus = focus if focus is not None else layers[int(np.argmax(contrast[:, -1]))]
    s = stats[focus]

    def save(fig, name):
        fig.suptitle(f"{run_name} | position={position}", fontsize = 9)
        fig.tight_layout()
        fig.savefig(figs / name, dpi = 140)
        plt.close(fig)

    # 0 -- the validity check. Read this before anything else.
    fig, ax = plt.subplots(figsize = (7, 4))
    im = ax.imshow(curve(stats, layers, "gauge_residual"), aspect = "auto", origin = "lower",
                   extent = [0, len(steps) - 1, layers[0], layers[-1]], cmap = "magma")
    ax.set(xlabel = "harvest index", ylabel = "layer",
           title = "fig 0  gauge residual (>0.3 means no rigid frame; rerun with rotate=False)")
    fig.colorbar(im, ax = ax)
    save(fig, "fig0_gauge_residual.png")

    # 1 -- where in depth anything happened
    fig, ax = plt.subplots(figsize = (7, 4))
    im = ax.imshow(contrast, aspect = "auto", origin = "lower",
                   extent = [0, len(steps) - 1, layers[0], layers[-1]], cmap = "viridis")
    ax.set(xlabel = "harvest index", ylabel = "layer",
           title = "fig 1  MEM minus FILL displacement (background radii)")
    fig.colorbar(im, ax = ax)
    save(fig, "fig1_depth_time.png")

    # 2 -- S1, the bucket-versus-private-paths curve
    fig, ax = plt.subplots(figsize = (7, 4))
    ax.plot(x_mid, s["coherence"], "o-", label = "MEM")
    ax.plot(x_mid, s["coherence_fill"], "o--", label = "FILL (null)")
    n_words = int(round(1 / max(s["coherence_fill"][0], 1e-9))) if s["coherence_fill"][0] > 0 else 0
    ax.axhline(s["coherence_fill"][0], color = "grey", lw = 0.8, ls = ":",
               label = f"unaligned floor ~1/N")
    ax.set(xscale = "log", xlabel = "optimizer step", ylabel = "leading eigenvalue share",
           title = f"fig 2  velocity coherence, layer {focus}\nhigh = one shared direction, floor = private paths")
    ax.legend(fontsize = 8)
    save(fig, "fig2_coherence.png")

    # 3 -- S2, is the motion leaving the pretrained subspace
    fig, ax = plt.subplots(figsize = (7, 4))
    ax.plot(x, s["novel"], "o-", label = "MEM")
    ax.plot(x, s["novel_fill"], "o--", label = "FILL (null)")
    ax.set(xscale = "log", xlabel = "optimizer step", ylabel = "fraction outside pretrained top-k",
           title = f"fig 3  novel-subspace fraction, layer {focus}\nrising = space the pretrained model was not using")
    ax.legend(fontsize = 8)
    save(fig, "fig3_novel_subspace.png")

    # 4 -- S3, does the direction wander before it settles
    fig, ax = plt.subplots(figsize = (7, 4))
    ax.plot(x[1:-1], s["inc_cos"], "o-", label = "MEM  cos(v_t, v_t+1)")
    ax.plot(x[1:-1], s["inc_cos_fill"], "o--", label = "FILL (null)")
    ax.plot(x, s["to_final"], "s-", label = "MEM  cos(displacement, final)")
    ax.axhline(0, color = "grey", lw = 0.8)
    ax.set(xscale = "log", xlabel = "optimizer step", ylabel = "cosine",
           title = f"fig 4  trajectory shape, layer {focus}  "
                   f"(median tortuosity {s['tortuosity']:.2f} vs fill {s['tortuosity_fill']:.2f})")
    ax.legend(fontsize = 8)
    save(fig, "fig4_trajectory_shape.png")

    # 5 -- S4 with the behavioural learning curve underneath it
    fig, ax = plt.subplots(figsize = (7, 4))
    ax.plot(x, s["separation"], "o-", color = "tab:blue", label = "centroid separation")
    ax.set(xscale = "log", xlabel = "optimizer step",
           ylabel = "centroid gap / within-cloud radius",
           title = f"fig 5  separation against behaviour, layer {focus}")
    twin = ax.twinx()
    rows = {r["step"]: r for r in logger.metrics("behaviour")}
    for key, style in (("train_MEM", "-"), ("eval_MEM", "--"), ("train_FILL", ":")):
        pts = [(st, r[key]) for st, r in sorted(rows.items()) if key in r]
        if pts:
            twin.plot([p[0] for p in pts], [p[1] for p in pts], style, color = "tab:red",
                      alpha = 0.7, label = key)
    twin.set_ylabel("marker rate", color = "tab:red")
    twin.set_ylim(-0.05, 1.05)
    ax.legend(fontsize = 8, loc = "upper left")
    twin.legend(fontsize = 8, loc = "lower right")
    save(fig, "fig5_separation_behaviour.png")

    print(f"wrote {len(list(figs.glob('*.png')))} figures to {figs} (focus layer {focus})")
    return figs


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description = "Analyse a v1 run.")
    ap.add_argument("--run", required = True)
    ap.add_argument("--position", default = "word", choices = ("word", "last"))
    ap.add_argument("--out", default = None)
    ap.add_argument("--k", type = int, default = 64, help = "pretrained subspace size")
    ap.add_argument("--rotate-dim", type = int, default = 256,
                    help = "principal directions the gauge rotation is fitted in; 0 for all")
    ap.add_argument("--no-rotate", action = "store_true",
                    help = "centre and scale only; the fallback when fig 0 looks bad")
    ap.add_argument("--focus-layer", type = int, default = None)
    ap.add_argument("--no-figures", action = "store_true")
    args = ap.parse_args()

    analysis = analyse_run(args.run, position = args.position, out_root = args.out, k = args.k,
                           rotate = not args.no_rotate,
                           rotate_dim = args.rotate_dim or None)
    if not args.no_figures:
        figures(args.run, analysis, out_root = args.out, focus = args.focus_layer)
