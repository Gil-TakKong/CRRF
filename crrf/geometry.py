# -*- coding: utf-8 -*-
"""Boundary rule, Moore neighborhood order and the insertion-sort median shared by Phase 1 and Phase 2."""
import numpy as np
from numba import njit

NORM_MIN = 0.0          # a vector whose norm is not above this value (the zero vector) has no color direction


def moore_rings(r_max):
    """Moore offsets by increasing Chebyshev radius, center excluded (cumulative counts 8, 24, 48, 80)."""
    dr, dc, cum = [], [], []
    for r in range(1, r_max + 1):
        for a in range(-r, r + 1):
            for b in range(-r, r + 1):
                if max(abs(a), abs(b)) == r:
                    dr.append(a)
                    dc.append(b)
        cum.append(len(dr))
    return np.array(dr, np.int64), np.array(dc, np.int64), np.array(cum, np.int64)


@njit(cache=True)
def ref101(i, n):
    """reflect_101 boundary: -1 -> 1, n -> n-2. The center is never read as its own neighbor."""
    if i < 0:
        return -i
    if i >= n:
        return 2 * n - 2 - i
    return i


@njit(cache=True)
def insertion_median(buf, n):
    """Sort buf[0:n] in place and return the median."""
    for i in range(1, n):
        v = buf[i]
        j = i - 1
        while j >= 0 and buf[j] > v:
            buf[j + 1] = buf[j]
            j -= 1
        buf[j + 1] = v
    if n % 2 == 1:
        return buf[n // 2]
    return 0.5 * (buf[n // 2 - 1] + buf[n // 2])
