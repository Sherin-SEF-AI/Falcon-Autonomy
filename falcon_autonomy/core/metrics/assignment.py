"""Linear assignment (numpy): the minimum-cost matching of the rows of a cost matrix to its columns.

Definition. For an n × m cost matrix C, a set of (row, column) pairs, each row and each column used at most once, with
min(n, m) pairs, whose total cost is the smallest possible (the rectangular linear sum assignment problem).

Method. The Hungarian method with shortest augmenting paths and dual potentials (Kuhn 1955, The Hungarian method for
the assignment problem, Naval Res. Logist. Q. 2; Munkres 1957; in the O(n²m) form of Jonker & Volgenant 1987, A
shortest augmenting path algorithm for dense and sparse linear assignment problems, Computing 38): one Dijkstra-like
augmentation per row, the inner scan over the columns vectorised. It returns an optimal assignment, as
scipy.optimize.linear_sum_assignment does (Crouse 2016, On implementing 2D rectangular assignment algorithms, IEEE TAES
52); when several assignments are optimal the two may pick different ones.

Limits. Costs must be finite (use a large finite value for forbidden pairs and drop those pairs afterwards). O(n²m)
time with n = min(rows, columns): fine for the tens to hundreds of objects of a frame.
"""
from __future__ import annotations

import numpy as np


def linear_assignment(cost, maximize: bool = False):
    """Optimal assignment of a 2-D cost (or, with maximize, score) matrix: (rows, cols) index arrays, rows ascending,
    min(n, m) pairs, as scipy.optimize.linear_sum_assignment returns them."""
    C = np.asarray(cost, dtype=float)
    if C.ndim != 2:
        raise ValueError(f'a cost matrix is 2-D; got shape {C.shape}')
    if not np.all(np.isfinite(C)):
        raise ValueError('costs must be finite')
    if maximize:
        C = -C
    transposed = C.shape[0] > C.shape[1]
    if transposed:
        C = C.T
    n, m = C.shape
    if n == 0:
        e = np.zeros(0, dtype=np.int64)
        return e, e.copy()
    u = np.zeros(n + 1)                      # row potentials (1-based; 0 is the virtual start)
    v = np.zeros(m + 1)                      # column potentials
    p = np.zeros(m + 1, dtype=np.int64)      # p[j]: the row (1-based) assigned to column j, 0 = free
    way = np.zeros(m + 1, dtype=np.int64)    # the previous column on the shortest path
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = np.full(m + 1, np.inf)
        used = np.zeros(m + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = C[i0 - 1] - u[i0] - v[1:]
            better = free & (cur < minv[1:])
            minv[1:][better] = cur[better]
            way[1:][better] = j0
            masked = np.where(free, minv[1:], np.inf)
            j1 = int(np.argmin(masked)) + 1
            delta = masked[j1 - 1]
            u[p[used]] += delta                 # the rows on the tree (distinct) and the columns reached
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:                              # augment along the path back to the start
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    cols = np.nonzero(p[1:])[0]
    rows = p[1:][cols] - 1
    if transposed:
        rows, cols = cols, rows
    order = np.argsort(rows, kind='stable')
    return rows[order].astype(np.int64), cols[order].astype(np.int64)
