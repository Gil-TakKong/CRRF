# -*- coding: utf-8 -*-
"""
VMF (Vector Median Filter).

Reference:
  J. Astola, P. Haavisto, Y. Neuvo, "Vector median filters,"
  Proceedings of the IEEE 78(4) (1990) 678-689. doi:10.1109/5.54807

Definition:
  Window    3x3 (9 samples, center included), applied to every pixel (no identification step).
  Distance  L2. Sum of distances of sample a: S_a = sum_{b=0..8} ||x_a - x_b||_2
  Output    the window sample with the smallest S_a, as a whole vector
  Ties      the first sample in raster order of the window
  Boundary  reflect_101 (-1 -> 1, n -> n-2)
  Input and output are H x W x 3 uint8 (the output values are input samples).

The three channels form one vector, so uncorrupted components can change as well.
"""

import math

import numpy as np
from numba import njit

from ..geometry import ref101

# Offsets of the 3x3 window in raster order. Index 4 is the center.
WIN_DR = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], np.int64)
WIN_DC = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], np.int64)


@njit(cache=True)
def _vmf_kernel(x, dr, dc):
    h = x.shape[0]
    w = x.shape[1]
    out = np.empty_like(x)
    buf = np.empty((9, 3), np.float64)
    for i in range(h):
        for j in range(w):
            for k in range(9):
                ii = ref101(i + dr[k], h)
                jj = ref101(j + dc[k], w)
                buf[k, 0] = x[ii, jj, 0]
                buf[k, 1] = x[ii, jj, 1]
                buf[k, 2] = x[ii, jj, 2]
            best = 0
            best_s = np.inf
            for a in range(9):
                s = 0.0
                for b in range(9):
                    d0 = buf[a, 0] - buf[b, 0]
                    d1 = buf[a, 1] - buf[b, 1]
                    d2 = buf[a, 2] - buf[b, 2]
                    s += math.sqrt(d0 * d0 + d1 * d1 + d2 * d2)
                if s < best_s:                  # strict: a tie keeps the earlier sample
                    best_s = s
                    best = a
            ii = ref101(i + dr[best], h)
            jj = ref101(j + dc[best], w)
            out[i, j, 0] = x[ii, jj, 0]
            out[i, j, 1] = x[ii, jj, 1]
            out[i, j, 2] = x[ii, jj, 2]
    return out


def vmf(noisy_u8):
    """VMF with a 3x3 window and the L2 distance. Input and output are H x W x 3 uint8."""
    x = np.asarray(noisy_u8)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"expected an H x W x 3 image: shape={x.shape}")
    if x.dtype != np.uint8:
        raise TypeError(f"expected a uint8 image: dtype={x.dtype}")
    if x.shape[0] < 2 or x.shape[1] < 2:
        raise ValueError("the reflect_101 boundary needs a height and a width of at least 2")
    return _vmf_kernel(np.ascontiguousarray(x), WIN_DR, WIN_DC)
