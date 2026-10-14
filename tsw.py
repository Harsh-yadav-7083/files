"""
Toy reproduction of Tree-Sliced Wasserstein (TSW) distances
(Le, Yamada, Fukumizu, Cuturi, NeurIPS 2019).

Contents
--------
make_clouds        : synthetic 2D point clouds (blobs / rings / moons)
build_quadtree     : Algorithm 1 (partition-based tree metric, random shift)
tw_embedding       : closed form of Eq. (3) as a weighted-L1 feature map
tsw_matrix         : Eq. (4), average of TW over n_s random trees
sw_matrix          : sliced-Wasserstein baseline (random 1D projections)
exact_ot_matrix    : exact OT with Euclidean ground cost (reference)
"""
import numpy as np
from scipy.spatial.distance import cdist
import ot


# ----------------------------------------------------------------------
# 1. Data
# ----------------------------------------------------------------------
def _blobs(rng, n):
    k = 2
    means = rng.uniform(-1.0, 1.0, size=(k, 2))
    scale = rng.uniform(0.08, 0.25)
    labels = rng.integers(0, k, size=n)
    return means[labels] + scale * rng.standard_normal((n, 2))


def _ring(rng, n):
    center = rng.uniform(-0.5, 0.5, size=2)
    radius = rng.uniform(0.6, 1.2)
    noise = rng.uniform(0.02, 0.08)
    ang = rng.uniform(0, 2 * np.pi, size=n)
    pts = np.c_[np.cos(ang), np.sin(ang)] * radius
    return center + pts + noise * rng.standard_normal((n, 2))


def _moons(rng, n):
    noise = rng.uniform(0.03, 0.10)
    scale = rng.uniform(0.6, 1.2)
    shift = rng.uniform(-0.5, 0.5, size=2)
    n1 = n // 2
    n2 = n - n1
    t1 = rng.uniform(0, np.pi, size=n1)
    t2 = rng.uniform(0, np.pi, size=n2)
    upper = np.c_[np.cos(t1), np.sin(t1)]
    lower = np.c_[1 - np.cos(t2), 0.5 - np.sin(t2)]
    pts = np.vstack([upper, lower]) - np.array([0.5, 0.25])
    return shift + scale * pts + noise * rng.standard_normal((n, 2))


def make_clouds(n_per_family=10, n_points=200, seed=0):
    """Return (clouds, labels): clouds has shape (3*n_per_family, n_points, 2)."""
    rng = np.random.default_rng(seed)
    gens = [_blobs, _ring, _moons]
    clouds, labels = [], []
    for lab, g in enumerate(gens):
        for _ in range(n_per_family):
            clouds.append(g(rng, n_points))
            labels.append(lab)
    return np.stack(clouds), np.array(labels)


# ----------------------------------------------------------------------
# 2. Quadtree (Algorithm 1)
# ----------------------------------------------------------------------
class Tree:
    """Tree stored as arrays. Nodes are created level by level, so
    parent[i] < i always holds (root = node 0)."""

    def __init__(self, parent, edge_len, level, point_node):
        self.parent = parent          # (n_nodes,) parent index, -1 for root
        self.edge_len = edge_len      # (n_nodes,) length of edge to parent, 0 for root
        self.level = level            # (n_nodes,) depth
        self.point_node = point_node  # (n_points,) node each data point is attached to


def build_quadtree(X, depth, rng):
    """
    Partition-based tree metric on the points X (N, d).

    - Randomly expand the data's bounding hypercube to a cube s' of side beta
      (so the grid position is random -> a random tree).
    - Recursively split every cell into 2^d children, discard empty cells.
    - A cell with a single point becomes a leaf at that data point
      (Algorithm 1 allows 'center of s or the data point').
    - Edge length = Euclidean distance between parent and child node.
    - Stop at the predefined deepest level `depth` (H_T in the paper).
    """
    N, d = X.shape
    lo_data, hi_data = X.min(0), X.max(0)
    rng_data = hi_data - lo_data
    beta = 2.0 * max(rng_data.max(), 1e-12)          # side of expanded cube
    u = rng.uniform(0, 1, size=d)
    lo = (hi_data - beta) + u * (beta - rng_data)     # random corner, data stays inside

    parent = [-1]
    edge_len = [0.0]
    level = [0]
    centers = [lo + beta / 2.0]

    point_node = np.zeros(N, dtype=np.int64)          # everything starts at root
    active = np.arange(N) if N > 1 else np.array([], dtype=np.int64)

    for h in range(1, depth + 1):
        if active.size == 0:
            break
        side = beta / (2 ** h)
        idx = np.floor((X[active] - lo) / side).astype(np.int64)
        idx = np.clip(idx, 0, 2 ** h - 1)
        cells, inv, counts = np.unique(idx, axis=0, return_inverse=True,
                                       return_counts=True)
        inv = inv.reshape(-1)
        first_new = len(parent)
        new_ids = first_new + np.arange(len(cells))

        for c in range(len(cells)):
            members = active[inv == c]
            par = point_node[members[0]]               # all members share the parent
            if counts[c] == 1:
                center = X[members[0]]
            else:
                center = lo + (cells[c] + 0.5) * side
            parent.append(int(par))
            edge_len.append(float(np.linalg.norm(center - centers[par])))
            level.append(h)
            centers.append(center)

        point_node[active] = new_ids[inv]
        # continue only with points whose cell still holds > 1 point
        keep = counts[inv] > 1
        active = active[keep]

    return Tree(np.array(parent), np.array(edge_len), np.array(level), point_node)


# ----------------------------------------------------------------------
# 3. Tree-Wasserstein (Eq. 3) as a feature map, and TSW (Eq. 4)
# ----------------------------------------------------------------------
def tw_embedding(tree, cloud_slices, weights=None):
    """
    For each measure mu_i (a slice of the global point array) compute
        phi_i[e] = w_e * mu_i(Gamma(v_e))
    so that TW(mu_i, mu_j) = || phi_i - phi_j ||_1   (Eq. 3).
    """
    n_nodes = len(tree.parent)
    n_meas = len(cloud_slices)
    M = np.zeros((n_nodes, n_meas))
    for j, sl in enumerate(cloud_slices):
        nodes = tree.point_node[sl]
        w = (np.full(len(nodes), 1.0 / len(nodes)) if weights is None
             else weights[j])
        np.add.at(M[:, j], nodes, w)

    # accumulate masses bottom-up -> mass of the subtree below every node
    for h in range(tree.level.max(), 0, -1):
        nodes_h = np.where(tree.level == h)[0]
        np.add.at(M, tree.parent[nodes_h], M[nodes_h])

    phi = (tree.edge_len[1:, None] * M[1:]).T          # drop the root (no edge)
    return phi


def tw_matrix(clouds, depth, rng):
    """Pairwise TW distances for one random tree built on all points."""
    n, m, d = clouds.shape
    X = clouds.reshape(-1, d)
    slices = [slice(i * m, (i + 1) * m) for i in range(n)]
    tree = build_quadtree(X, depth, rng)
    phi = tw_embedding(tree, slices)
    return cdist(phi, phi, "cityblock")


def tsw_matrix(clouds, n_slices, depth, rng, return_all=False):
    """TSW (Eq. 4) = average of TW over n_slices random trees."""
    mats = [tw_matrix(clouds, depth, rng) for _ in range(n_slices)]
    if return_all:
        return mats
    return np.mean(mats, axis=0)


# ----------------------------------------------------------------------
# 4. Baselines
# ----------------------------------------------------------------------
def sw_matrix(clouds, n_proj, rng, return_all=False):
    """Sliced Wasserstein-1: average over random directions of the 1D W1."""
    n, m, d = clouds.shape
    theta = rng.standard_normal((n_proj, d))
    theta /= np.linalg.norm(theta, axis=1, keepdims=True)
    proj = np.einsum("nmd,pd->npm", clouds, theta)     # (n, n_proj, m)
    proj.sort(axis=2)                                  # sorted 1D samples
    if return_all:
        out = []
        for p in range(n_proj):
            out.append(cdist(proj[:, p, :], proj[:, p, :], "cityblock") / m)
        return out
    flat = proj.reshape(n, -1)
    return cdist(flat, flat, "cityblock") / (m * n_proj)


def exact_ot(a_pts, b_pts):
    """Exact OT (W1) with Euclidean ground cost, uniform weights."""
    C = cdist(a_pts, b_pts)
    a = np.full(len(a_pts), 1.0 / len(a_pts))
    b = np.full(len(b_pts), 1.0 / len(b_pts))
    return ot.emd2(a, b, C)


def exact_ot_matrix(clouds):
    n = len(clouds)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = exact_ot(clouds[i], clouds[j])
    return D


def upper(D):
    """Upper-triangle entries (all distinct pairs) as a vector."""
    return D[np.triu_indices_from(D, k=1)]
