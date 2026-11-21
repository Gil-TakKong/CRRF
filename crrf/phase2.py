# -*- coding: utf-8 -*-
"""Phase 2: restoration of the marked components from the median color direction of the 7x7 window.

For a pixel p with k >= 1 marked components, with u(q) the Phase 1 vector at q:

    c(q)  = u(q) / ||u(q)||                 for the 49 window samples (center included, reflect_101)
    m     = per-component median of c(q)    (the 25th smallest of each component)
    c_hat = m / ||m||
    k < 3: x_j = mean over anchors a of u_a(p) * c_hat_j / c_hat_a
           anchors = unmarked channels of p with c_hat_a > 0; no anchor keeps the Phase 1 value
    k = 3: x_j = ||u(p)|| * c_hat_j

Safeguards: a sample with u(q) = 0 is excluded; a pixel with u(p) = 0 or m = 0 keeps
its Phase 1 values. They act only where Phase 1 left a component of value 0, which in the paper
happens only in the known-mask setting. Unmarked components never change. All reads use the Phase 1 output (one
synchronous pass) and the values are not clipped here.

Implementation: c(q) is computed once per pixel, and the medians of all windows of an image row are
computed together by the sorting networks of median_network.py. Windows with fewer than 49 valid
samples, which occur only where Phase 1 left zero vectors, use a per-pixel insertion sort. The
kernel in variants.py computes the same values window by window and the tests compare the two bit
for bit.
"""
import math

import numpy as np
from numba import njit

from .geometry import NORM_MIN, insertion_median, moore_rings, ref101
from .median_network import CQ, PR, SR, _lv1, _lv2, _lv3, _lv4, _lv5

M_DR, M_DC, _M_CUM = moore_rings(3)
N_NEIGHBORS = int(_M_CUM[-1])       # 48


@njit(cache=True)
def _finish(src, det, out, x, y, nc, I_t, mcu):
    """Normalize the median direction and restore the marked components of one pixel.
    Returns the increments of (skipped pixels, restored values outside [0, 255])."""
    n_skip = 0
    n_over = 0
    nrm = np.sqrt(mcu[0]**2 + mcu[1]**2 + mcu[2]**2)
    if nrm == 0.0:
        n_skip += 1
        return n_skip, n_over
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
    return n_skip, n_over


@njit(cache=True)
def _prep(src, h, w):
    """Norms, validity and color directions on padded columns j = 0 .. w+6 (j <-> image column
    ref101(j - 3)), split into even j (Ce) and odd j (Co). Also reports whether the input lies outside
    the domain of the networks (NaN, inf, negative value or -0.0, none of which Phase 1 produces)."""
    Wp = w + 7
    Q = (w + 1) // 2
    NL = ((Q + CQ - 1) // CQ) * CQ + SR          # every chunk reads SR lanes
    inten = np.empty((h, w), np.float64)
    ok = np.empty((h, w), np.bool_)
    Ce = np.zeros((3, h, NL), np.float64)
    Co = np.zeros((3, h, NL), np.float64)
    rowb = np.empty((3, Wp), np.float64)
    nbad = 0
    bad_input = False
    for x in range(h):
        for y in range(w):
            b = src[x, y, 0]
            g = src[x, y, 1]
            rr = src[x, y, 2]
            for v in (b, g, rr):
                if not (v >= 0.0 and v <= 1.7976931348623157e308):
                    bad_input = True
                elif v == 0.0 and math.copysign(1.0, v) < 0.0:
                    bad_input = True
            nI = np.sqrt(b * b + g * g + rr * rr)
            inten[x, y] = nI
            good = nI > NORM_MIN
            ok[x, y] = good
            if good:
                rowb[0, y + 3] = src[x, y, 0] / nI
                rowb[1, y + 3] = src[x, y, 1] / nI
                rowb[2, y + 3] = src[x, y, 2] / nI
            else:
                nbad += 1
                rowb[0, y + 3] = 0.0
                rowb[1, y + 3] = 0.0
                rowb[2, y + 3] = 0.0
        for c in range(3):
            for j in range(3):                   # j -> image column 3 - j
                rowb[c, j] = rowb[c, 6 - j]
            for j in range(w + 3, w + 6):        # j -> image column 2w - 2 - (j - 3)
                rowb[c, j] = rowb[c, 2 * w + 4 - j]
            rowb[c, w + 6] = rowb[c, w + 5]      # read only by the absent window y = w (odd w)
            for i in range((Wp + 1) // 2):
                Ce[c, x, i] = rowb[c, 2 * i]
            for i in range(Wp // 2):
                Co[c, x, i] = rowb[c, 2 * i + 1]
    return inten, ok, Ce, Co, nbad, bad_input


@njit(cache=True)
def _phase2_core(src, det, h, w, inten, ok, Ce, Co, nbad):
    out = src.copy()
    Q = (w + 1) // 2                              # window pairs (y = 2q, 2q + 1)
    NQ = ((Q + CQ - 1) // CQ) * CQ
    Se = np.empty(7 * SR, np.float64)             # sorted even columns of the chunk
    So = np.empty(7 * SR, np.float64)             # sorted odd columns of the chunk
    P = np.empty(14 * PR, np.float64)
    D = np.empty(28 * CQ, np.float64)
    K = np.empty(36 * CQ, np.float64)
    Me = np.empty((3, NQ), np.float64)            # medians of the windows y = 2q
    Mo = np.empty((3, NQ), np.float64)            # medians of the windows y = 2q + 1
    xr = np.empty(7, np.int64)
    tgt = np.empty(w, np.int64)
    tnc = np.empty(w, np.int64)
    cbuf = np.empty((N_NEIGHBORS + 1, 3), np.float64)
    tmp = np.empty(N_NEIGHBORS + 1, np.float64)
    mcu = np.empty(3, np.float64)
    n_skip = 0
    n_over = 0

    for x in range(h):
        nt = 0
        for y in range(w):
            nc = 0
            for ch in range(3):
                if det[x, y, ch]:
                    nc += 1
            if nc == 0:
                continue
            if inten[x, y] <= NORM_MIN:
                n_skip += 1
                continue
            tgt[nt] = y
            tnc[nt] = nc
            nt += 1
        if nt == 0:
            continue
        for a in range(7):
            xr[a] = ref101(x + a - 3, h)

        # medians of all windows of row x (padded column j of window y: y .. y+6)
        for c in range(3):
            e0 = Ce[c, xr[0]]; e1 = Ce[c, xr[1]]; e2 = Ce[c, xr[2]]; e3 = Ce[c, xr[3]]
            e4 = Ce[c, xr[4]]; e5 = Ce[c, xr[5]]; e6 = Ce[c, xr[6]]
            o0 = Co[c, xr[0]]; o1 = Co[c, xr[1]]; o2 = Co[c, xr[2]]; o3 = Co[c, xr[3]]
            o4 = Co[c, xr[4]]; o5 = Co[c, xr[5]]; o6 = Co[c, xr[6]]
            for q0 in range(0, Q, CQ):
                _lv1(e0[q0:], e1[q0:], e2[q0:], e3[q0:], e4[q0:], e5[q0:], e6[q0:], Se)
                _lv1(o0[q0:], o1[q0:], o2[q0:], o3[q0:], o4[q0:], o5[q0:], o6[q0:], So)
                _lv2(So, Se, P)                   # P_m = (column 2m+1, column 2m+2)
                _lv3(P, D)
                _lv4(D, P, K)
                _lv5(K, Se, Me[c, q0:])           # window 2q   : core + column 2q
                _lv5(K, So[3:], Mo[c, q0:])       # window 2q+1 : core + column 2q+7

        for t in range(nt):
            y = tgt[t]
            nc = tnc[t]
            I_t = inten[x, y]
            full = True
            if nbad > 0:
                for a in range(7):
                    for bb in range(7):
                        if not ok[xr[a], ref101(y + bb - 3, w)]:
                            full = False
            if full:
                if y % 2 == 0:
                    mcu[0] = Me[0, y // 2]
                    mcu[1] = Me[1, y // 2]
                    mcu[2] = Me[2, y // 2]
                else:
                    mcu[0] = Mo[0, y // 2]
                    mcu[1] = Mo[1, y // 2]
                    mcu[2] = Mo[2, y // 2]
            else:
                # fewer than 49 valid samples: gather them one by one, center first
                cbuf[0, 0] = src[x, y, 0] / I_t
                cbuf[0, 1] = src[x, y, 1] / I_t
                cbuf[0, 2] = src[x, y, 2] / I_t
                nval = 1
                for k in range(N_NEIGHBORS):
                    nx = ref101(x + M_DR[k], h)
                    ny = ref101(y + M_DC[k], w)
                    nI = inten[nx, ny]
                    if nI <= NORM_MIN:
                        continue
                    cbuf[nval, 0] = src[nx, ny, 0] / nI
                    cbuf[nval, 1] = src[nx, ny, 1] / nI
                    cbuf[nval, 2] = src[nx, ny, 2] / nI
                    nval += 1
                for ci in range(3):
                    for i in range(nval):
                        tmp[i] = cbuf[i, ci]
                    mcu[ci] = insertion_median(tmp, nval)
            s, o = _finish(src, det, out, x, y, nc, I_t, mcu)
            n_skip += s
            n_over += o
    return out, n_skip, n_over


def phase2(estimate, marked):
    """Phase 2 on the Phase 1 output (float64, H x W x 3, H, W >= 4) and the identification mask.
    Returns (restored, skipped pixels, restored values outside [0, 255])."""
    estimate = np.ascontiguousarray(estimate)
    marked = np.ascontiguousarray(marked)
    if estimate.dtype != np.float64 or marked.dtype != np.bool_:
        raise TypeError("phase2 expects a float64 estimate and a boolean mask")
    if estimate.ndim != 3 or estimate.shape[2] != 3 or marked.shape != estimate.shape:
        raise ValueError("phase2 expects H x W x 3 arrays of the same shape")
    h, w, _ = estimate.shape
    if h < 4 or w < 4:
        raise ValueError("the 7x7 reflect_101 window needs an image of at least 4 x 4 pixels")
    inten, ok, Ce, Co, nbad, bad_input = _prep(estimate, h, w)
    if bad_input:
        raise ValueError("phase2 expects finite, non-negative values (Phase 1 of an 8-bit image gives these)")
    return _phase2_core(estimate, marked, h, w, inten, ok, Ce, Co, nbad)
