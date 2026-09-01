"""
Tests for the geometry in src/stats.py. Numpy only, no model, no GPU.

Each test builds a cloud whose motion is known by construction and checks the
statistic reports the thing it claims to report. Run: python3 test/test.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import gauge, geometry, stats  # noqa: E402

RNG = np.random.default_rng(0)
T, N, M, D = 8, 60, 400, 96
FAILED = []


def check(name, condition, detail = ""):
    tick = "ok  " if condition else "FAIL"
    print(f"  [{tick}] {name}{'   ' + detail if detail else ''}")
    if not condition:
        FAILED.append(name)


def rigid(X, t, R):
    """Translate, rescale and rotate a cloud the way a fine-tune drifts the whole stream."""
    return ((X + 3.0 * t) * (1.0 + 0.4 * t)) @ np.linalg.matrix_power(R, t)


def make_base():
    base = RNG.normal(size = (M + 2 * N, D))
    return base[:M], base[M:M + N], base[M + N:]


def test_gauge_removes_rigid_drift():
    print("gauge")
    bg0, mem0, _ = make_base()
    R = np.linalg.qr(RNG.normal(size = (D, D)))[0]
    bg = np.stack([rigid(bg0, t, R) for t in range(T)])
    mem = np.stack([rigid(mem0, t, R) for t in range(T)])
    traj, res = gauge.gauge_trajectory(bg, mem)
    check("residual is zero under pure rigid drift", res.max() < 1e-8, f"max {res.max():.2e}")
    check("gauged displacement is zero", stats.displacement(traj).max() < 1e-8)


def test_gauge_keeps_real_motion():
    print("gauge keeps signal")
    bg0, mem0, _ = make_base()
    R = np.linalg.qr(RNG.normal(size = (D, D)))[0]
    d = RNG.normal(size = D)
    d /= np.linalg.norm(d)
    bg = np.stack([rigid(bg0, t, R) for t in range(T)])
    mem = np.stack([rigid(mem0 + 0.5 * t * d, t, R) for t in range(T)])
    traj, res = gauge.gauge_trajectory(bg, mem)
    disp = stats.displacement(traj)
    check("residual still zero", res.max() < 1e-8)
    check("displacement grows monotonically", bool(np.all(np.diff(disp) > 0)))


def test_coherence_separates_shared_from_private():
    print("S1 velocity coherence")
    bg0, mem0, _ = make_base()
    d = RNG.normal(size = D)
    shared = np.stack([mem0 + 0.4 * t * d for t in range(T)])
    private = np.stack([mem0 + 0.4 * t * RNG.normal(size = (N, D)) for t in range(T)])
    hi = stats.velocity_coherence(shared)["top"]
    lo = stats.velocity_coherence(private)["top"]
    check("shared drift saturates the leading eigenvalue", hi.min() > 0.99, f"min {hi.min():.3f}")
    check("private paths sit near 1/N", lo.max() < 4.0 / N, f"max {lo.max():.4f} vs 1/N={1/N:.4f}")
    check("shared participation ratio is ~1", stats.velocity_coherence(shared)["pr"].max() < 1.01)


def test_novel_fraction_sees_new_directions():
    print("S2 novel subspace")
    bg0, mem0, _ = make_base()
    Q = geometry.top_pcs(bg0, 16)
    inside_dir = Q[0]
    outside_dir = RNG.normal(size = D)
    outside_dir -= Q.T @ (Q @ outside_dir)
    outside_dir /= np.linalg.norm(outside_dir)
    inside = np.stack([mem0 + 0.4 * t * inside_dir for t in range(T)])
    outside = np.stack([mem0 + 0.4 * t * outside_dir for t in range(T)])
    fi = stats.novel_fraction(inside, bg0, k = 16)[-1]
    fo = stats.novel_fraction(outside, bg0, k = 16)[-1]
    check("motion inside the pretrained subspace reads ~0", fi < 0.02, f"{fi:.4f}")
    check("motion orthogonal to it reads ~1", fo > 0.98, f"{fo:.4f}")


def test_trajectory_shape():
    print("S3 trajectory shape")
    bg0, mem0, _ = make_base()
    d = RNG.normal(size = D)
    straight = np.stack([mem0 + 0.4 * t * d for t in range(T)])
    # a genuine random walk: increments independent, so consecutive velocities are too.
    # (mem0 + fresh noise each step is NOT this -- consecutive increments share a term
    #  and are anticorrelated at -0.5, which would test the wrong thing.)
    steps = 0.4 * RNG.normal(size = (T - 1, N, D))
    wander = np.concatenate([mem0[None], mem0[None] + np.cumsum(steps, axis = 0)])
    check("straight line has tortuosity 1", abs(np.median(stats.tortuosity(straight)) - 1.0) < 1e-6)
    check("wandering has tortuosity > 2", np.median(stats.tortuosity(wander)) > 2.0,
          f"{np.median(stats.tortuosity(wander)):.2f}")
    check("straight line keeps consecutive cosine 1", stats.incremental_cosine(straight).min() > 0.999)
    check("wandering keeps consecutive cosine near 0", abs(stats.incremental_cosine(wander)).max() < 0.3)
    check("straight line converges to final direction", np.nanmin(stats.direction_to_final(straight)) > 0.999)


def test_separation_is_scale_free():
    print("S4 cloud geometry")
    bg0, mem0, fill0 = make_base()
    d = RNG.normal(size = D)
    d /= np.linalg.norm(d)
    # the within-cloud radius is ~sqrt(D), so a shift only registers relative to that.
    # This is the point of the statistic: unlike probe accuracy it cannot saturate just
    # because there are more dimensions than words.
    seps = [stats.cloud_separation(mem0 + a * d, fill0) for a in (0, 5, 20, 80)]
    check("separation grows monotonically with the shift", all(b > a for a, b in zip(seps, seps[1:])),
          " -> ".join(f"{x:.2f}" for x in seps))
    check("a shift of one cloud radius reads ~1",
          abs(stats.cloud_separation(mem0 + np.sqrt(D) * d, fill0) - 1.0) < 0.15,
          f"{stats.cloud_separation(mem0 + np.sqrt(D) * d, fill0):.2f}")
    check("separation is invariant to global rescaling",
          abs(stats.cloud_separation(2 * (mem0 + 5 * d), 2 * fill0) - seps[1]) < 1e-8)


def test_summarise_contract():
    print("summarise")
    bg0, mem0, fill0 = make_base()
    d = RNG.normal(size = D)
    mem = np.stack([mem0 + 0.4 * t * d for t in range(T)])
    fill = np.stack([fill0 + 0.0 * t for t in range(T)])
    s = stats.summarise(mem, fill, bg0, k = 16)
    check("every MEM key has a FILL control", all(f"{k}_fill" in s for k in
          ("coherence", "novel", "tortuosity", "inc_cos", "to_final", "displacement", "pr")))
    check("per-checkpoint arrays are length T", len(s["displacement"]) == T and len(s["separation"]) == T)
    check("per-interval arrays are length T-1", len(s["coherence"]) == T - 1)
    check("MEM coherence beats its FILL control", s["coherence"].min() > s["coherence_fill"].max())


def test_schedule():
    print("schedule")
    for total in (50, 400, 2000):
        sched = train.checkpoint_schedule(total)
        check(f"schedule({total}) starts at 0 and ends at {total}",
              sched[0] == 0 and sched[-1] == total)
        check(f"schedule({total}) is strictly increasing", all(b > a for a, b in zip(sched, sched[1:])))
        check(f"schedule({total}) is dense early", sched[:5] == [0, 1, 2, 3, 4])
        half = sched[len(sched) // 2]
        check(f"schedule({total}) is log-spaced, midpoint {half} << {total // 2}", half < total // 2)


if __name__ == "__main__":
    for fn in (test_gauge_removes_rigid_drift, test_gauge_keeps_real_motion,
               test_coherence_separates_shared_from_private, test_novel_fraction_sees_new_directions,
               test_trajectory_shape, test_separation_is_scale_free,
               test_summarise_contract, test_schedule):
        fn()
    print()
    if FAILED:
        print(f"{len(FAILED)} FAILED: {FAILED}")
        sys.exit(1)
    print("all passed")
