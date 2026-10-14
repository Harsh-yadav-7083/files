"""
Experiments for the toy reproduction.

Exp A  runtime of exact OT vs SW vs TSW for a single pair of clouds,
       as the number of points per cloud grows.
Exp B  agreement (Pearson / Spearman) with exact OT over all pairs of clouds,
       as the number of slices n_s grows (TSW trees vs SW projections).
"""
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from tsw import (make_clouds, tsw_matrix, sw_matrix, exact_ot_matrix,
                 exact_ot, tw_matrix, upper, build_quadtree, tw_embedding)

DEPTH = 6                    # H_T: deepest tree level
N_SLICES = [1, 2, 5, 10, 20, 50]
REPEATS = 5                  # independent random draws of trees / projections
SEED = 0


# ----------------------------------------------------------------------
def exp_runtime(sizes=(100, 250, 500, 1000, 2000), n_slices=10, reps=3):
    rng = np.random.default_rng(SEED)
    out = {"n": [], "OT": [], "SW": [], "TSW": []}
    for n in sizes:
        a, b = (make_clouds(n_per_family=1, n_points=n, seed=SEED + n)[0][[0, 1]])
        pair = np.stack([a, b])
        t_ot, t_sw, t_tsw = [], [], []
        for _ in range(reps):
            t = time.perf_counter(); exact_ot(a, b); t_ot.append(time.perf_counter() - t)
            t = time.perf_counter(); sw_matrix(pair, n_slices, rng); t_sw.append(time.perf_counter() - t)
            t = time.perf_counter(); tsw_matrix(pair, n_slices, DEPTH, rng); t_tsw.append(time.perf_counter() - t)
        out["n"].append(n)
        out["OT"].append(float(np.median(t_ot)))
        out["SW"].append(float(np.median(t_sw)))
        out["TSW"].append(float(np.median(t_tsw)))
        print(f"n={n:5d}  OT {out['OT'][-1]*1e3:9.1f} ms   "
              f"SW {out['SW'][-1]*1e3:8.1f} ms   TSW {out['TSW'][-1]*1e3:8.1f} ms")
    return out


def exp_agreement(clouds, D_ot):
    """For each n_s, mean/std over REPEATS of the correlation with exact OT."""
    ref = upper(D_ot)
    res = {"n_slices": N_SLICES,
           "TSW_pearson": [], "TSW_pearson_std": [], "TSW_spearman": [], "TSW_spearman_std": [],
           "SW_pearson": [], "SW_pearson_std": [], "SW_spearman": [], "SW_spearman_std": []}
    # Each repeat and slice count use their own random draw.
    for ns in N_SLICES:
        tp, ts, sp, ss = [], [], [], []
        for r in range(REPEATS):
            rng = np.random.default_rng(1000 * r + ns)
            D_tsw = tsw_matrix(clouds, ns, DEPTH, rng)
            D_sw = sw_matrix(clouds, ns, rng)
            tp.append(pearsonr(upper(D_tsw), ref)[0]); ts.append(spearmanr(upper(D_tsw), ref)[0])
            sp.append(pearsonr(upper(D_sw), ref)[0]);  ss.append(spearmanr(upper(D_sw), ref)[0])
        for key, v in [("TSW_pearson", tp), ("TSW_spearman", ts), ("SW_pearson", sp), ("SW_spearman", ss)]:
            res[key].append(float(np.mean(v)))
            res[key + "_std"].append(float(np.std(v)))
        print(f"n_s={ns:3d}  TSW  pearson {np.mean(tp):.3f}±{np.std(tp):.3f}  "
              f"spearman {np.mean(ts):.3f}±{np.std(ts):.3f}   |   "
              f"SW  pearson {np.mean(sp):.3f}±{np.std(sp):.3f}  "
              f"spearman {np.mean(ss):.3f}±{np.std(ss):.3f}")
    return res


# ----------------------------------------------------------------------
def make_plots(rt, ag, clouds, D_ot):
    # runtime
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(rt["n"], rt["OT"], "o-", label="Exact OT (LP)")
    ax.plot(rt["n"], rt["SW"], "s-", label="SW (10 proj.)")
    ax.plot(rt["n"], rt["TSW"], "^-", label="TSW (10 trees, incl. tree building)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("points per cloud"); ax.set_ylabel("time per pair (s)")
    ax.set_title("Runtime for one pair of 2D clouds"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("fig_runtime.png", dpi=160); plt.close(fig)

    # agreement with exact OT
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4))
    for ax, kind in zip(axes, ["pearson", "spearman"]):
        for name, mk in [("TSW", "^"), ("SW", "s")]:
            m = np.array(ag[f"{name}_{kind}"]); s = np.array(ag[f"{name}_{kind}_std"])
            ax.errorbar(ag["n_slices"], m, yerr=s, marker=mk, capsize=3, label=name)
        ax.set_xscale("log"); ax.set_xlabel("number of slices $n_s$ (trees / projections)")
        ax.set_ylabel(f"{kind.capitalize()} correlation with exact OT")
        ax.grid(alpha=.3); ax.legend()
    fig.suptitle("Agreement with exact OT over all pairs of clouds", y=1.0)
    fig.tight_layout(); fig.savefig("fig_agreement.png", dpi=160); plt.close(fig)

    # scatter for one TSW draw with n_s = 10
    rng = np.random.default_rng(7)
    D = tsw_matrix(clouds, 10, DEPTH, rng)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.scatter(upper(D_ot), upper(D), s=6, alpha=.5)
    ax.set_xlabel("exact OT (Euclidean ground cost)"); ax.set_ylabel("TSW, $n_s$=10")
    ax.set_title("One TSW draw vs exact OT"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("fig_scatter.png", dpi=160); plt.close(fig)


def main():
    clouds, labels = make_clouds(n_per_family=10, n_points=200, seed=SEED)
    print(f"{len(clouds)} clouds, {clouds.shape[1]} points each, dimension {clouds.shape[2]}")

    t = time.perf_counter()
    D_ot = exact_ot_matrix(clouds)
    print(f"exact OT on all {len(clouds)*(len(clouds)-1)//2} pairs: {time.perf_counter()-t:.1f} s\n")

    print("== Exp A: runtime ==")
    rt = exp_runtime()
    print("\n== Exp B: agreement with exact OT ==")
    ag = exp_agreement(clouds, D_ot)

    make_plots(rt, ag, clouds, D_ot)
    with open("results.json", "w") as f:
        json.dump({"runtime": rt, "agreement": ag, "depth": DEPTH}, f, indent=2)
    print("\nSaved fig_runtime.png, fig_agreement.png, fig_scatter.png, results.json")


if __name__ == "__main__":
    main()
