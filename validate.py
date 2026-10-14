"""
Correctness check of Proposition 1 (closed form of tree-Wasserstein):
for a given tree, the closed-form TW (Eq. 3) must equal exact OT
computed by linear programming with the tree metric as ground cost.
"""
import numpy as np
import ot
from tsw import make_clouds, build_quadtree, tw_embedding


def ancestors(tree, v):
    path = [v]
    while tree.parent[path[-1]] != -1:
        path.append(tree.parent[path[-1]])
    return path                                        # v, ..., root


def tree_dist(tree, dist_root, a, b):
    """Length of the unique path between nodes a and b."""
    anc_a = ancestors(tree, a)
    set_a = set(anc_a)
    for v in ancestors(tree, b):
        if v in set_a:                                 # first common ancestor = LCA
            return dist_root[a] + dist_root[b] - 2 * dist_root[v]
    raise RuntimeError("nodes are not in the same tree")


def main():
    rng = np.random.default_rng(123)
    clouds, _ = make_clouds(n_per_family=2, n_points=80, seed=1)
    max_err = 0.0
    n_checked = 0
    for trial in range(10):
        i, j = rng.choice(len(clouds), size=2, replace=False)
        X = np.vstack([clouds[i], clouds[j]])
        m = len(clouds[i])
        tree = build_quadtree(X, depth=6, rng=rng)

        # distance from the root for every node (parents are created first)
        dist_root = np.zeros(len(tree.parent))
        for v in range(1, len(tree.parent)):
            dist_root[v] = dist_root[tree.parent[v]] + tree.edge_len[v]

        # (a) closed form, Eq. (3)
        phi = tw_embedding(tree, [slice(0, m), slice(m, 2 * m)])
        tw_closed = np.abs(phi[0] - phi[1]).sum()

        # (b) exact OT with the tree metric as ground cost
        na = tree.point_node[:m]
        nb = tree.point_node[m:]
        C = np.array([[tree_dist(tree, dist_root, a, b) for b in nb] for a in na])
        w = np.full(m, 1.0 / m)
        tw_lp = ot.emd2(w, w, C)

        err = abs(tw_closed - tw_lp)
        max_err = max(max_err, err)
        n_checked += 1
        print(f"pair ({i:2d},{j:2d})  closed form = {tw_closed:.8f}   "
              f"LP = {tw_lp:.8f}   |diff| = {err:.2e}")

    print(f"\nChecked {n_checked} pairs; max |closed form - LP| = {max_err:.2e}")
    assert max_err < 1e-8, "closed form does not match exact OT on the tree!"
    print("PASS: Eq. (3) matches exact OT on the tree metric.")


if __name__ == "__main__":
    main()
