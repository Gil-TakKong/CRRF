# -*- coding: utf-8 -*-
"""Min/max sorting networks that find the medians of all 7x7 windows of an image row.

The median of 49 samples is the 25th smallest. A min/max network only moves values, so its output is
the very float64 that sorting would leave at that rank. Work is shared between neighboring windows:

    L1  sort each 7-sample column of the 7-row band                (optimal 16-comparator 7-sorter)
    L2  merge sorted columns (2m+1, 2m+2) into sorted pairs of 14   (used by 3 window pairs)
    L3  merge pairs m, m+1 -> keep ranks 3..24 of 28
    L4  merge with pair m+2 -> ranks 17..24 of the 42 samples shared by windows 2q and 2q+1
    L5  merge those 8 with the window's own 7th column -> rank 7 of 15 = median of 49

Each comparator list is generated from Batcher's odd-even merge and pruned to the ranks the next
level needs (the k-th of merge(A, B) needs only A[max(0, k-|B|) .. min(k, |A|-1)]). The kernels are
straight-line code generated from these lists (the block between the GENERATED markers). The function
check_generated() regenerates the block and compares, and the tests prove each level by the 0-1
principle.
"""
import numpy as np
from numba import njit


# A comparator (i, j) writes min to wire i and max to wire j.
# After pruning: op 0 = both outputs, 1 = min only, 2 = max only.
SORT7 = [(0, 6), (2, 3), (4, 5), (0, 2), (1, 4), (3, 6), (0, 1), (2, 5), (3, 4),
         (1, 2), (4, 6), (2, 3), (4, 5), (1, 2), (3, 4), (5, 6)]      # optimal 7-sorter


def oe_merge(a, b):
    """Batcher odd-even merge of sorted wire lists a and b (any lengths).
    Returns (comparators, wires in ascending rank order)."""
    if not a:
        return [], list(b)
    if not b:
        return [], list(a)
    if len(a) == 1 and len(b) == 1:
        return [(a[0], b[0])], [a[0], b[0]]
    ce, E = oe_merge(a[0::2], b[0::2])
    co, O = oe_merge(a[1::2], b[1::2])
    comps = ce + co
    out = [E[0]]
    i = 0
    while i < len(O) or i + 1 < len(E):
        if i < len(O) and i + 1 < len(E):
            comps.append((O[i], E[i + 1]))
            out += [O[i], E[i + 1]]
        elif i < len(O):
            out.append(O[i])
        else:
            out.append(E[i + 1])
        i += 1
    return comps, out


def prune(comps, out_wires):
    """Keep only what the output wires depend on (backward pass)."""
    need = set(out_wires)
    kept = []
    for (i, j) in reversed(comps):
        ni, nj = i in need, j in need
        if not ni and not nj:
            continue
        kept.append((i, j, 0 if (ni and nj) else (1 if ni else 2)))
        need.add(i); need.add(j)
    kept.reverse()
    return kept


def merge_net(m, n, ranks):
    """merge of sorted A (wires 0..m-1) and sorted B (wires m..m+n-1) pruned to the given ranks.
    Returns (comparators, order) with order[k] = wire that holds rank k."""
    comps, order = oe_merge(list(range(m)), list(range(m, m + n)))
    return prune(comps, [order[k] for k in ranks]), order


def build_levels():
    L1 = [(i, j, 0) for (i, j) in SORT7]                 # column of 7 -> sorted 7
    L2, P_ORD = merge_net(7, 7, range(14))               # pair: 7 + 7 -> 14
    L3, D_ORD = merge_net(14, 14, range(3, 25))          # 14 + 14 -> ranks 3..24 of 28
    L4, K_ORD = merge_net(22, 14, range(14, 22))         # D[3..24] + pair -> core ranks 17..24
    L5, F_ORD = merge_net(8, 7, [7])                     # core[17..24] + column -> rank 7 of 15
    return [L1, L2, L3, L4, L5], P_ORD, D_ORD, K_ORD, F_ORD


LEVELS, P_ORD, D_ORD, K_ORD, F_ORD = build_levels()
CQ_GEN = 64                  # window pairs per chunk in the generated kernels


def level_source(CQ=CQ_GEN):
    """Source of the five level kernels (straight-line min/max code, constant trip counts so that
    LLVM vectorises the lane loops).  The GENERATED block below is exactly this text."""
    SR, PR = CQ + 4, CQ + 2
    L1, L2, L3, L4, L5 = LEVELS
    I = " " * 8

    def net(kept):
        out = []
        for (i, j, op) in kept:
            if op == 0:
                out.append(f"{I}t = min(v{i}, v{j}); v{j} = max(v{i}, v{j}); v{i} = t")
            elif op == 1:
                out.append(f"{I}v{i} = min(v{i}, v{j})")
            else:
                out.append(f"{I}v{j} = max(v{i}, v{j})")
        return out

    def store_all(arr, rowlen, first, nwires):
        # needed ranks to rows 0.., the other wires to spare rows (keeps the loop vectorisable)
        rest = [k for k in range(nwires) if k not in first]
        return [f"{I}{arr}[{r * rowlen} + i] = v{wv}" for r, wv in enumerate(list(first) + rest)]

    s = [f"CQ = {CQ}\nSR = {SR}\nPR = {PR}\n\n"]
    s.append("@njit(cache=True)\ndef _lv1(r0, r1, r2, r3, r4, r5, r6, S):")
    s.append('    """L1: S[r*SR + i] = rank r of the 7 band samples of column i (r0..r6 = band rows)."""')
    s.append("    for i in range(SR):")
    s += [f"{I}v{r} = r{r}[i]" for r in range(7)]
    s += net(L1)
    s += [f"{I}S[{r * SR} + i] = v{r}" for r in range(7)]
    s.append("\n")
    s.append("@njit(cache=True)\ndef _lv2(So, Se, P):")
    s.append('    """L2: P[r*PR + i] = rank r of merge(So column i, Se column i+1)."""')
    s.append("    for i in range(PR):")
    s += [f"{I}v{k} = So[{k * SR} + i]" for k in range(7)]
    s += [f"{I}v{7 + k} = Se[{k * SR + 1} + i]" for k in range(7)]
    s += net(L2)
    s += store_all("P", PR, P_ORD, 14)
    s.append("\n")
    s.append("@njit(cache=True)\ndef _lv3(P, D):")
    s.append('    """L3: D[r*CQ + i] = rank 3+r of merge(P_i, P_{i+1}), r = 0..21."""')
    s.append("    for i in range(CQ):")
    s += [f"{I}v{k} = P[{k * PR} + i]" for k in range(14)]
    s += [f"{I}v{14 + k} = P[{k * PR + 1} + i]" for k in range(14)]
    s += net(L3)
    s += store_all("D", CQ, D_ORD[3:25], 28)
    s.append("\n")
    s.append("@njit(cache=True)\ndef _lv4(D, P, K):")
    s.append('    """L4: K[r*CQ + i] = rank 17+r of the 6-column core of window pair i, r = 0..7."""')
    s.append("    for i in range(CQ):")
    s += [f"{I}v{r} = D[{r * CQ} + i]" for r in range(22)]
    s += [f"{I}v{22 + k} = P[{k * PR + 2} + i]" for k in range(14)]
    s += net(L4)
    s += store_all("K", CQ, K_ORD[14:22], 36)
    s.append("\n")
    s.append("@njit(cache=True)\ndef _lv5(K, Sx, M):")
    s.append('    """L5: M[i] = rank 7 of merge(K column i, Sx column i) = median of the 49 samples."""')
    s.append("    for i in range(CQ):")
    s += [f"{I}v{r} = K[{r * CQ} + i]" for r in range(8)]
    s += [f"{I}v{8 + k} = Sx[{k * SR} + i]" for k in range(7)]
    s += net(L5)
    s.append(f"{I}M[i] = v{F_ORD[7]}")
    return "\n".join(s) + "\n"


# ---------------------------------------------------------------- BEGIN GENERATED (level_source())
CQ = 64
SR = 68
PR = 66


@njit(cache=True)
def _lv1(r0, r1, r2, r3, r4, r5, r6, S):
    """L1: S[r*SR + i] = rank r of the 7 band samples of column i (r0..r6 = band rows)."""
    for i in range(SR):
        v0 = r0[i]
        v1 = r1[i]
        v2 = r2[i]
        v3 = r3[i]
        v4 = r4[i]
        v5 = r5[i]
        v6 = r6[i]
        t = min(v0, v6); v6 = max(v0, v6); v0 = t
        t = min(v2, v3); v3 = max(v2, v3); v2 = t
        t = min(v4, v5); v5 = max(v4, v5); v4 = t
        t = min(v0, v2); v2 = max(v0, v2); v0 = t
        t = min(v1, v4); v4 = max(v1, v4); v1 = t
        t = min(v3, v6); v6 = max(v3, v6); v3 = t
        t = min(v0, v1); v1 = max(v0, v1); v0 = t
        t = min(v2, v5); v5 = max(v2, v5); v2 = t
        t = min(v3, v4); v4 = max(v3, v4); v3 = t
        t = min(v1, v2); v2 = max(v1, v2); v1 = t
        t = min(v4, v6); v6 = max(v4, v6); v4 = t
        t = min(v2, v3); v3 = max(v2, v3); v2 = t
        t = min(v4, v5); v5 = max(v4, v5); v4 = t
        t = min(v1, v2); v2 = max(v1, v2); v1 = t
        t = min(v3, v4); v4 = max(v3, v4); v3 = t
        t = min(v5, v6); v6 = max(v5, v6); v5 = t
        S[0 + i] = v0
        S[68 + i] = v1
        S[136 + i] = v2
        S[204 + i] = v3
        S[272 + i] = v4
        S[340 + i] = v5
        S[408 + i] = v6


@njit(cache=True)
def _lv2(So, Se, P):
    """L2: P[r*PR + i] = rank r of merge(So column i, Se column i+1)."""
    for i in range(PR):
        v0 = So[0 + i]
        v1 = So[68 + i]
        v2 = So[136 + i]
        v3 = So[204 + i]
        v4 = So[272 + i]
        v5 = So[340 + i]
        v6 = So[408 + i]
        v7 = Se[1 + i]
        v8 = Se[69 + i]
        v9 = Se[137 + i]
        v10 = Se[205 + i]
        v11 = Se[273 + i]
        v12 = Se[341 + i]
        v13 = Se[409 + i]
        t = min(v0, v7); v7 = max(v0, v7); v0 = t
        t = min(v4, v11); v11 = max(v4, v11); v4 = t
        t = min(v4, v7); v7 = max(v4, v7); v4 = t
        t = min(v2, v9); v9 = max(v2, v9); v2 = t
        t = min(v6, v13); v13 = max(v6, v13); v6 = t
        t = min(v6, v9); v9 = max(v6, v9); v6 = t
        t = min(v2, v4); v4 = max(v2, v4); v2 = t
        t = min(v6, v7); v7 = max(v6, v7); v6 = t
        t = min(v9, v11); v11 = max(v9, v11); v9 = t
        t = min(v1, v8); v8 = max(v1, v8); v1 = t
        t = min(v5, v12); v12 = max(v5, v12); v5 = t
        t = min(v5, v8); v8 = max(v5, v8); v5 = t
        t = min(v3, v10); v10 = max(v3, v10); v3 = t
        t = min(v3, v5); v5 = max(v3, v5); v3 = t
        t = min(v10, v8); v8 = max(v10, v8); v10 = t
        t = min(v1, v2); v2 = max(v1, v2); v1 = t
        t = min(v3, v4); v4 = max(v3, v4); v3 = t
        t = min(v5, v6); v6 = max(v5, v6); v5 = t
        t = min(v10, v7); v7 = max(v10, v7); v10 = t
        t = min(v8, v9); v9 = max(v8, v9); v8 = t
        t = min(v12, v11); v11 = max(v12, v11); v12 = t
        P[0 + i] = v0
        P[66 + i] = v1
        P[132 + i] = v2
        P[198 + i] = v3
        P[264 + i] = v4
        P[330 + i] = v5
        P[396 + i] = v6
        P[462 + i] = v10
        P[528 + i] = v7
        P[594 + i] = v8
        P[660 + i] = v9
        P[726 + i] = v12
        P[792 + i] = v11
        P[858 + i] = v13


@njit(cache=True)
def _lv3(P, D):
    """L3: D[r*CQ + i] = rank 3+r of merge(P_i, P_{i+1}), r = 0..21."""
    for i in range(CQ):
        v0 = P[0 + i]
        v1 = P[66 + i]
        v2 = P[132 + i]
        v3 = P[198 + i]
        v4 = P[264 + i]
        v5 = P[330 + i]
        v6 = P[396 + i]
        v7 = P[462 + i]
        v8 = P[528 + i]
        v9 = P[594 + i]
        v10 = P[660 + i]
        v11 = P[726 + i]
        v12 = P[792 + i]
        v13 = P[858 + i]
        v14 = P[1 + i]
        v15 = P[67 + i]
        v16 = P[133 + i]
        v17 = P[199 + i]
        v18 = P[265 + i]
        v19 = P[331 + i]
        v20 = P[397 + i]
        v21 = P[463 + i]
        v22 = P[529 + i]
        v23 = P[595 + i]
        v24 = P[661 + i]
        v25 = P[727 + i]
        v26 = P[793 + i]
        v27 = P[859 + i]
        v14 = max(v0, v14)
        t = min(v8, v22); v22 = max(v8, v22); v8 = t
        t = min(v8, v14); v14 = max(v8, v14); v8 = t
        t = min(v4, v18); v18 = max(v4, v18); v4 = t
        v12 = min(v12, v26)
        t = min(v12, v18); v18 = max(v12, v18); v12 = t
        t = min(v4, v8); v8 = max(v4, v8); v4 = t
        t = min(v12, v14); v14 = max(v12, v14); v12 = t
        t = min(v18, v22); v22 = max(v18, v22); v18 = t
        t = min(v2, v16); v16 = max(v2, v16); v2 = t
        t = min(v10, v24); v24 = max(v10, v24); v10 = t
        t = min(v10, v16); v16 = max(v10, v16); v10 = t
        t = min(v6, v20); v20 = max(v6, v20); v6 = t
        t = min(v6, v10); v10 = max(v6, v10); v6 = t
        t = min(v20, v16); v16 = max(v20, v16); v20 = t
        v4 = max(v2, v4)
        t = min(v6, v8); v8 = max(v6, v8); v6 = t
        t = min(v10, v12); v12 = max(v10, v12); v10 = t
        t = min(v20, v14); v14 = max(v20, v14); v20 = t
        t = min(v16, v18); v18 = max(v16, v18); v16 = t
        t = min(v24, v22); v22 = max(v24, v22); v24 = t
        v15 = max(v1, v15)
        t = min(v9, v23); v23 = max(v9, v23); v9 = t
        t = min(v9, v15); v15 = max(v9, v15); v9 = t
        t = min(v5, v19); v19 = max(v5, v19); v5 = t
        v13 = min(v13, v27)
        t = min(v13, v19); v19 = max(v13, v19); v13 = t
        t = min(v5, v9); v9 = max(v5, v9); v5 = t
        t = min(v13, v15); v15 = max(v13, v15); v13 = t
        t = min(v19, v23); v23 = max(v19, v23); v19 = t
        t = min(v3, v17); v17 = max(v3, v17); v3 = t
        t = min(v11, v25); v25 = max(v11, v25); v11 = t
        t = min(v11, v17); v17 = max(v11, v17); v11 = t
        t = min(v7, v21); v21 = max(v7, v21); v7 = t
        t = min(v7, v11); v11 = max(v7, v11); v7 = t
        t = min(v21, v17); v17 = max(v21, v17); v21 = t
        t = min(v3, v5); v5 = max(v3, v5); v3 = t
        t = min(v7, v9); v9 = max(v7, v9); v7 = t
        t = min(v11, v13); v13 = max(v11, v13); v11 = t
        t = min(v21, v15); v15 = max(v21, v15); v21 = t
        t = min(v17, v19); v19 = max(v17, v19); v17 = t
        v25 = min(v25, v23)
        t = min(v3, v4); v4 = max(v3, v4); v3 = t
        t = min(v5, v6); v6 = max(v5, v6); v5 = t
        t = min(v7, v8); v8 = max(v7, v8); v7 = t
        t = min(v9, v10); v10 = max(v9, v10); v9 = t
        t = min(v11, v12); v12 = max(v11, v12); v11 = t
        t = min(v13, v20); v20 = max(v13, v20); v13 = t
        t = min(v21, v14); v14 = max(v21, v14); v21 = t
        t = min(v15, v16); v16 = max(v15, v16); v15 = t
        t = min(v17, v18); v18 = max(v17, v18); v17 = t
        t = min(v19, v24); v24 = max(v19, v24); v19 = t
        t = min(v25, v22); v22 = max(v25, v22); v25 = t
        D[0 + i] = v3
        D[64 + i] = v4
        D[128 + i] = v5
        D[192 + i] = v6
        D[256 + i] = v7
        D[320 + i] = v8
        D[384 + i] = v9
        D[448 + i] = v10
        D[512 + i] = v11
        D[576 + i] = v12
        D[640 + i] = v13
        D[704 + i] = v20
        D[768 + i] = v21
        D[832 + i] = v14
        D[896 + i] = v15
        D[960 + i] = v16
        D[1024 + i] = v17
        D[1088 + i] = v18
        D[1152 + i] = v19
        D[1216 + i] = v24
        D[1280 + i] = v25
        D[1344 + i] = v22
        D[1408 + i] = v0
        D[1472 + i] = v1
        D[1536 + i] = v2
        D[1600 + i] = v23
        D[1664 + i] = v26
        D[1728 + i] = v27


@njit(cache=True)
def _lv4(D, P, K):
    """L4: K[r*CQ + i] = rank 17+r of the 6-column core of window pair i, r = 0..7."""
    for i in range(CQ):
        v0 = D[0 + i]
        v1 = D[64 + i]
        v2 = D[128 + i]
        v3 = D[192 + i]
        v4 = D[256 + i]
        v5 = D[320 + i]
        v6 = D[384 + i]
        v7 = D[448 + i]
        v8 = D[512 + i]
        v9 = D[576 + i]
        v10 = D[640 + i]
        v11 = D[704 + i]
        v12 = D[768 + i]
        v13 = D[832 + i]
        v14 = D[896 + i]
        v15 = D[960 + i]
        v16 = D[1024 + i]
        v17 = D[1088 + i]
        v18 = D[1152 + i]
        v19 = D[1216 + i]
        v20 = D[1280 + i]
        v21 = D[1344 + i]
        v22 = P[2 + i]
        v23 = P[68 + i]
        v24 = P[134 + i]
        v25 = P[200 + i]
        v26 = P[266 + i]
        v27 = P[332 + i]
        v28 = P[398 + i]
        v29 = P[464 + i]
        v30 = P[530 + i]
        v31 = P[596 + i]
        v32 = P[662 + i]
        v33 = P[728 + i]
        v34 = P[794 + i]
        v35 = P[860 + i]
        v22 = max(v0, v22)
        t = min(v16, v22); v22 = max(v16, v22); v16 = t
        t = min(v8, v30); v30 = max(v8, v30); v8 = t
        v16 = max(v8, v16)
        v30 = min(v30, v22)
        v26 = max(v4, v26)
        v20 = min(v20, v26)
        v12 = min(v12, v34)
        t = min(v12, v20); v20 = max(v12, v20); v12 = t
        v16 = max(v12, v16)
        t = min(v20, v30); v30 = max(v20, v30); v20 = t
        v24 = max(v2, v24)
        t = min(v18, v24); v24 = max(v18, v24); v18 = t
        t = min(v10, v32); v32 = max(v10, v32); v10 = t
        v18 = max(v10, v18)
        v32 = min(v32, v24)
        v28 = max(v6, v28)
        t = min(v14, v28); v28 = max(v14, v28); v14 = t
        t = min(v14, v18); v18 = max(v14, v18); v14 = t
        v28 = min(v28, v32)
        t = min(v14, v16); v16 = max(v14, v16); v14 = t
        t = min(v18, v20); v20 = max(v18, v20); v18 = t
        v28 = min(v28, v30)
        v23 = max(v1, v23)
        t = min(v17, v23); v23 = max(v17, v23); v17 = t
        t = min(v9, v31); v31 = max(v9, v31); v9 = t
        v17 = max(v9, v17)
        v31 = min(v31, v23)
        v27 = max(v5, v27)
        v21 = min(v21, v27)
        v13 = min(v13, v35)
        t = min(v13, v21); v21 = max(v13, v21); v13 = t
        t = min(v13, v17); v17 = max(v13, v17); v13 = t
        v21 = min(v21, v31)
        v25 = max(v3, v25)
        v19 = min(v19, v25)
        v11 = min(v11, v33)
        t = min(v11, v19); v19 = max(v11, v19); v11 = t
        t = min(v7, v29); v29 = max(v7, v29); v7 = t
        v15 = min(v15, v29)
        v11 = max(v7, v11)
        t = min(v15, v19); v19 = max(v15, v19); v15 = t
        v13 = max(v11, v13)
        t = min(v15, v17); v17 = max(v15, v17); v15 = t
        t = min(v19, v21); v21 = max(v19, v21); v19 = t
        v14 = max(v13, v14)
        t = min(v15, v16); v16 = max(v15, v16); v15 = t
        t = min(v17, v18); v18 = max(v17, v18); v17 = t
        t = min(v19, v20); v20 = max(v19, v20); v19 = t
        v21 = min(v21, v28)
        K[0 + i] = v14
        K[64 + i] = v15
        K[128 + i] = v16
        K[192 + i] = v17
        K[256 + i] = v18
        K[320 + i] = v19
        K[384 + i] = v20
        K[448 + i] = v21
        K[512 + i] = v0
        K[576 + i] = v1
        K[640 + i] = v2
        K[704 + i] = v3
        K[768 + i] = v4
        K[832 + i] = v5
        K[896 + i] = v6
        K[960 + i] = v7
        K[1024 + i] = v8
        K[1088 + i] = v9
        K[1152 + i] = v10
        K[1216 + i] = v11
        K[1280 + i] = v12
        K[1344 + i] = v13
        K[1408 + i] = v22
        K[1472 + i] = v23
        K[1536 + i] = v24
        K[1600 + i] = v25
        K[1664 + i] = v26
        K[1728 + i] = v27
        K[1792 + i] = v28
        K[1856 + i] = v29
        K[1920 + i] = v30
        K[1984 + i] = v31
        K[2048 + i] = v32
        K[2112 + i] = v33
        K[2176 + i] = v34
        K[2240 + i] = v35


@njit(cache=True)
def _lv5(K, Sx, M):
    """L5: M[i] = rank 7 of merge(K column i, Sx column i) = median of the 49 samples."""
    for i in range(CQ):
        v0 = K[0 + i]
        v1 = K[64 + i]
        v2 = K[128 + i]
        v3 = K[192 + i]
        v4 = K[256 + i]
        v5 = K[320 + i]
        v6 = K[384 + i]
        v7 = K[448 + i]
        v8 = Sx[0 + i]
        v9 = Sx[68 + i]
        v10 = Sx[136 + i]
        v11 = Sx[204 + i]
        v12 = Sx[272 + i]
        v13 = Sx[340 + i]
        v14 = Sx[408 + i]
        v8 = max(v0, v8)
        v4 = min(v4, v12)
        v8 = max(v4, v8)
        v10 = max(v2, v10)
        v6 = min(v6, v14)
        v6 = min(v6, v10)
        v8 = max(v6, v8)
        v9 = max(v1, v9)
        v5 = min(v5, v13)
        v9 = max(v5, v9)
        v11 = max(v3, v11)
        v7 = min(v7, v11)
        v7 = min(v7, v9)
        v7 = min(v7, v8)
        M[i] = v7
# ---------------------------------------------------------------- END GENERATED


def check_generated():
    """True when the GENERATED block equals level_source()."""
    import io
    import os
    with io.open(os.path.abspath(__file__), encoding="utf-8") as f:
        text = f.read()
    a = text.index("BEGIN GENERATED (level_source())")
    a = text.index("\n", a) + 1
    b = text.index("# ---------------------------------------------------------------- END GENERATED")
    return text[a:b] == level_source()
