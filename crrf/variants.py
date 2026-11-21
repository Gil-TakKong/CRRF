# -*- coding: utf-8 -*-
"""Phase 2 configurations of the analysis in Section 4.2 of the paper.

One window-by-window kernel covers the window sizes and the configurations A1 to A5. It reads the
Phase 1 output once and writes to a separate array, with the same safeguards as Phase 2. Samples are
gathered center first, then by Chebyshev ring. With radius 3, STAT_DIRECTION_MEDIAN and SCOPE_ALL it
returns the same bits as crrf.phase2 (this is checked in the tests), which makes it the reference for
the sorting-network path.

    radius    1 -> 3x3 (9 samples), 2 -> 5x5 (25), 3 -> 7x7 (49), 4 -> 9x9 (81)
    stat      STAT_DIRECTION_MEDIAN  per-component median of the color directions c(q)   (A0)
              STAT_CHANNEL_MEDIAN    the median of u_j(q) replaces marked component j    (A1)
              STAT_RGB_MEDIAN        per-component median of the raw vectors u(q)        (A2)
              STAT_DIRECTION_MEAN    per-component mean of the color directions c(q)     (A3)
    scope     SCOPE_ALL     every pixel with a marked component                          (A0)
              SCOPE_ANCHOR  only pixels with k < 3; pixels with k = 3 keep Phase 1       (A4)
              SCOPE_MAGNITUDE  only pixels with k = 3; others keep Phase 1               (A5)
"""
import numpy as np
from numba import njit

from .geometry import NORM_MIN, insertion_median, moore_rings, ref101

STAT_DIRECTION_MEDIAN = 0
STAT_RGB_MEDIAN = 1
STAT_DIRECTION_MEAN = 2
STAT_CHANNEL_MEDIAN = 3
SCOPE_ALL = 0
SCOPE_ANCHOR = 1
SCOPE_MAGNITUDE = 2

G_DR, G_DC, G_CUM = moore_rings(4)      # cumulative 8, 24, 48, 80
N_MAX = int(G_CUM[-1])                  # 80


@njit(cache=True)
def _variant_kernel(src, det, h, w, radius, stat, scope):
    out = src.copy()
    nn = G_CUM[radius - 1]
    direction = (stat == 0) or (stat == 2)          # the statistic uses c(q) rather than u(q)

    inten = np.empty((h, w), np.float64)
    for x in range(h):
        for y in range(w):
            b = src[x, y, 0]
            g = src[x, y, 1]
            rr = src[x, y, 2]
            inten[x, y] = np.sqrt(b * b + g * g + rr * rr)

    cbuf = np.empty((N_MAX + 1, 3), np.float64)
    tmp = np.empty(N_MAX + 1, np.float64)
    mcu = np.empty(3, np.float64)
    n_skip = 0
    n_over = 0

    for x in range(h):
        for y in range(w):
            nc = 0
            for ch in range(3):
                if det[x, y, ch]:
                    nc += 1
            if nc == 0:
                continue
            if scope == 1 and nc == 3:
                continue
            if scope == 2 and nc != 3:
                continue
            I_t = inten[x, y]
            if I_t <= NORM_MIN:
                n_skip += 1
                continue

            # samples: center first, then by Chebyshev ring
            if direction:
                cbuf[0, 0] = src[x, y, 0] / I_t
                cbuf[0, 1] = src[x, y, 1] / I_t
                cbuf[0, 2] = src[x, y, 2] / I_t
            else:
                cbuf[0, 0] = src[x, y, 0]
                cbuf[0, 1] = src[x, y, 1]
                cbuf[0, 2] = src[x, y, 2]
            nval = 1
            for k in range(nn):
                nx = ref101(x + G_DR[k], h)
                ny = ref101(y + G_DC[k], w)
                nI = inten[nx, ny]
                if nI <= NORM_MIN:
                    continue
                if direction:
                    cbuf[nval, 0] = src[nx, ny, 0] / nI
                    cbuf[nval, 1] = src[nx, ny, 1] / nI
                    cbuf[nval, 2] = src[nx, ny, 2] / nI
                else:
                    cbuf[nval, 0] = src[nx, ny, 0]
                    cbuf[nval, 1] = src[nx, ny, 1]
                    cbuf[nval, 2] = src[nx, ny, 2]
                nval += 1

            # A1: the same-channel median replaces the marked component directly
            if stat == 3:
                for ch in range(3):
                    if not det[x, y, ch]:
                        continue
                    for i in range(nval):
                        tmp[i] = cbuf[i, ch]
                    v = insertion_median(tmp, nval)
                    if v < 0.0 or v > 255.0:
                        n_over += 1
                    out[x, y, ch] = v
                continue

            for ci in range(3):
                if stat == 2:
                    s = 0.0
                    for i in range(nval):
                        s += cbuf[i, ci]
                    mcu[ci] = s / nval
                else:
                    for i in range(nval):
                        tmp[i] = cbuf[i, ci]
                    mcu[ci] = insertion_median(tmp, nval)
            nrm = np.sqrt(mcu[0]**2 + mcu[1]**2 + mcu[2]**2)
            if nrm == 0.0:
                n_skip += 1
                continue
            mcu[0] /= nrm
            mcu[1] /= nrm
            mcu[2] /= nrm

            if nc == 3:
                for ch in range(3):
                    v = I_t * mcu[ch]
                    if v < 0.0 or v > 255.0:
                        n_over += 1
                    out[x, y, ch] = v
            else:
                for ch in range(3):
                    if not det[x, y, ch]:
                        continue
                    acc = 0.0
                    cnt = 0
                    for a in range(3):
                        if det[x, y, a]:
                            continue
                        ma = mcu[a]
                        if ma > 0.0:
                            acc += src[x, y, a] * (mcu[ch] / ma)
                            cnt += 1
                    if cnt > 0:
                        v = acc / cnt
                        if v < 0.0 or v > 255.0:
                            n_over += 1
                        out[x, y, ch] = v
    return out, n_skip, n_over


def phase2_variant(estimate, marked, radius=3, stat=STAT_DIRECTION_MEDIAN, scope=SCOPE_ALL):
    """One Phase 2 configuration. Returns (restored, skipped pixels, restored values outside [0, 255])."""
    estimate = np.ascontiguousarray(estimate)
    marked = np.ascontiguousarray(marked)
    h, w, _ = estimate.shape
    if radius not in (1, 2, 3, 4):
        raise ValueError("radius must be 1, 2, 3 or 4")
    if h <= radius or w <= radius:
        raise ValueError(f"radius {radius} needs an image of at least {radius + 1} x {radius + 1} pixels")
    return _variant_kernel(estimate, marked, h, w, radius, stat, scope)
