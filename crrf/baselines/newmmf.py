#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEWMMF: neighborhood expansion weighted mean-median filter (grayscale).

This file is the released implementation of NEWMMF (newmmf/newmmf.py of its repository) with the
same constants and the same arithmetic. Only the imports differ: reflect_index and is_clean are
defined here, and median_array comes from helpers.py of this package (the same function).
It is applied to each channel separately, like the other filters of GRAY_METHODS.

Public API:
  - run_newmmf(noisy, return_phases=False)

Parameters of the NEWMMF paper (module constants):
  - T1, T2, T3 = 2, 2, 3   radius-wise thresholds (counts of usable pixels)
  - K_MAX = 300            safeguard: upper limit on the Phase 1 iterations
"""

import numpy as np
from numba import njit

from .helpers import median_array


@njit(cache=True, inline='always')
def reflect_index(idx, size):
    """Half-sample symmetric reflection: -1 -> 0, -2 -> 1, ..., size -> size-1, size+1 -> size-2."""
    if idx < 0:
        return -idx - 1
    elif idx >= size:
        return 2 * size - 1 - idx
    return idx


@njit(cache=True, inline='always')
def is_clean(v):
    """True if v is neither salt nor pepper."""
    return v != 0.0 and v != 255.0


# ============================================================================
# NEWMMF — Neighborhood Expansion Weighted Mean-Median Filter
#
# Noise mask N_M:
#   every pixel whose value in the noisy image is 0 or 255.
#
# Phase 1 (iterative):
#   - Every pixel of N_M that still has the value 0 or 255 is evaluated with
#     the von Neumann neighborhoods of radius r = 1, 2, 3 (4, 12 and 24
#     pixels), tried in this order.
#   - A neighbor is usable if it lies outside N_M or has already been
#     restored. A restored value is a weighted mean of values in [1, 254] and
#     can never be 0 or 255, so "restored" is decided from the value itself
#     (is_clean).
#   - The pixel is restored at the first radius whose number of usable
#     neighbors is at least T_r (a count, not a proportion), with the weighted
#     mean of those neighbors; the weight of a neighbor is 1 / (its Manhattan
#     distance from the pixel).
#   - If no radius qualifies, the pixel is deferred to the next iteration.
#   - Synchronous update: every pixel reads the state left by the previous
#     iteration, and values written in the current iteration are used from
#     the next iteration on.
#   - Phase 1 ends when no pixel of N_M has the value 0 or 255. Two
#     safeguards stop it otherwise: an iteration without restoration, and
#     reaching K_MAX = 300 iterations.
#
# Phase 2 (one synchronous pass, applied to all of N_M):
#   - Each pixel of N_M is replaced by the median of its four von Neumann
#     neighbors (radius 1, center excluded); the median of four values is
#     the mean of the two middle values.
#
# Boundary handling: half-sample symmetric reflection (-1 -> 0,
# size -> size - 1). All values stay in float64; the caller converts the
# output to 8 bits (to_uint8).
# ============================================================================

T1, T2, T3 = 2, 2, 3
K_MAX = 300


def build_von(r):
    """Build von Neumann neighborhood offsets up to Manhattan distance r,
    with weight 1/distance."""
    offsets = []
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            if dx == 0 and dy == 0:
                continue
            md = abs(dx) + abs(dy)
            if md <= r:
                offsets.append((dx, dy, 1.0 / md))
    return (
        np.array([o[0] for o in offsets], np.int64),
        np.array([o[1] for o in offsets], np.int64),
        np.array([o[2] for o in offsets], np.float64),
    )


V1_DR, V1_DC, V1_WT = build_von(1)
V2_DR, V2_DC, V2_WT = build_von(2)
V3_DR, V3_DC, V3_WT = build_von(3)


@njit(cache=True)
def _phase1(ch_img, noise_mask, h, w):
    """
    NEWMMF Phase 1: von Neumann neighborhood expansion weighted mean,
    iterated with synchronous updates (see the header comment).
    """
    img = ch_img.copy()
    vals = np.empty(64, np.float64)
    wts = np.empty(64, np.float64)

    for it in range(K_MAX):
        # Check if any noise pixels remain
        has_n = False
        for i in range(h):
            for j in range(w):
                if noise_mask[i, j] and (img[i, j] == 0.0 or img[i, j] == 255.0):
                    has_n = True
                    break
            if has_n:
                break
        if not has_n:
            break

        nxt = img.copy()
        cnt_total = 0

        for x in range(h):
            for y in range(w):
                if not noise_mask[x, y]:
                    continue
                if img[x, y] != 0.0 and img[x, y] != 255.0:
                    continue

                found = False
                cc = 0

                # r=1 attempt
                for kk in range(V1_DR.shape[0]):
                    nx = reflect_index(x + V1_DR[kk], h)
                    ny = reflect_index(y + V1_DC[kk], w)
                    v = img[nx, ny]
                    if not noise_mask[nx, ny] or is_clean(v):
                        vals[cc] = v
                        wts[cc] = V1_WT[kk]
                        cc += 1
                if cc >= T1:
                    found = True

                # r=2 attempt
                if not found:
                    cc = 0
                    for kk in range(V2_DR.shape[0]):
                        nx = reflect_index(x + V2_DR[kk], h)
                        ny = reflect_index(y + V2_DC[kk], w)
                        v = img[nx, ny]
                        if not noise_mask[nx, ny] or is_clean(v):
                            vals[cc] = v
                            wts[cc] = V2_WT[kk]
                            cc += 1
                    if cc >= T2:
                        found = True

                # r=3 attempt
                if not found:
                    cc = 0
                    for kk in range(V3_DR.shape[0]):
                        nx = reflect_index(x + V3_DR[kk], h)
                        ny = reflect_index(y + V3_DC[kk], w)
                        v = img[nx, ny]
                        if not noise_mask[nx, ny] or is_clean(v):
                            vals[cc] = v
                            wts[cc] = V3_WT[kk]
                            cc += 1
                    if cc >= T3:
                        found = True

                if not found:
                    continue

                # Weighted mean
                sw = 0.0
                swv = 0.0
                for i2 in range(cc):
                    sw += wts[i2]
                    swv += wts[i2] * vals[i2]
                nxt[x, y] = swv / sw
                cnt_total += 1

        img = nxt
        if cnt_total == 0:
            break

    return img


@njit(cache=True)
def _phase2(img, noise_mask, h, w):
    """
    NEWMMF Phase 2: median of the four von Neumann (r = 1) neighbors,
    center excluded. 1 pass, applied to every pixel of the noise mask.

    Synchronous (Jacobi) update: every cell reads from the frozen previous
    state `src` and writes to `result`, so the pass is order-independent and
    consistent with the synchronous updating used in Phase 1.
    """
    src = img.copy()
    result = img.copy()
    vals = np.empty(4, np.float64)
    for x in range(h):
        for y in range(w):
            if not noise_mask[x, y]:
                continue
            cc = 0
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if abs(dr) + abs(dc) != 1:
                        continue
                    nx = reflect_index(x + dr, h)
                    ny = reflect_index(y + dc, w)
                    vals[cc] = src[nx, ny]
                    cc += 1
            result[x, y] = median_array(vals, cc)
    return result


def run_newmmf(noisy, return_phases=False):
    """
    NEWMMF on one 2-D channel.

    The noise mask N_M consists of the pixels whose value in the noisy image
    is 0 or 255.

    Args:
        noisy:         2D array, SPN-corrupted image.
        return_phases: if True, return (phase1_result, phase2_result).

    Returns:
        - return_phases=False (default): final Phase 2 image as float64.
        - return_phases=True:            (s1, s2) tuple, both float64.
        Caller may clip and cast to uint8 (to_uint8).
    """
    h, w = noisy.shape
    img_f = noisy.astype(np.float64)
    noise_mask = (img_f == 0.0) | (img_f == 255.0)
    s1 = _phase1(img_f, noise_mask, h, w)
    s2 = _phase2(s1, noise_mask, h, w)
    if return_phases:
        return s1, s2
    return s2
