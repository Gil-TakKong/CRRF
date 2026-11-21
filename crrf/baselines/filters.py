#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
filters.py - DBA, NAFSMF and DAMF, compared per channel.

Implementations in this file:
  - DBA    (Srinivasan & Ebenezer, 2007, IEEE SPL 14(3): 189-192)
  - NAFSMF (Toh & Mat Isa, 2010, IEEE SPL 17(3): 281-284)
  - DAMF   (Erkan et al., 2018, Computers & Electrical Engineering 70: 789-798)

Public API:
  - run_dba(noisy)
  - run_nafsmf(noisy, T1=10.0, T2=30.0, max_window=7)
  - run_damf(noisy, max_window=21)
"""

import numpy as np
from numba import njit
from .helpers import is_noisy, median_array


# ============================================================================
# DBA - Decision-Based Algorithm
# Reference:
#   K.S. Srinivasan, D. Ebenezer,
#   "A new fast and efficient decision-based algorithm for removal of
#   high-density impulse noises,"
#   IEEE Signal Processing Letters, vol. 14, no. 3, pp. 189-192, March 2007.
# ============================================================================
@njit(cache=True)
def _dba_kernel(noisy):
    """
    DBA core algorithm (canonical Srinivasan & Ebenezer, 2007).

    For each pixel in raster-scan order:
      1. If pixel is not 0/255 (not corrupted): keep unchanged.
      2. If pixel is 0 or 255 (corrupted):
         a. Form the full 3x3 window from the CURRENT (partially restored)
            state, INCLUDING the center and any corrupted (0/255) values.
         b. Compute the median of that window.
         c. If the median is NOT 0/255 (a noise-free value) -> replace the
            pixel with the window median.
         d. If the median is itself 0/255 (the window is dominated by
            corrupted pixels) -> replace with the previously processed pixel
            (left, then top). This left-propagation is the defining behavior
            of DBA and the source of its high-density streaking.

    Boundary handling:
      Within-image entries only (window positions outside the image are
      excluded; the median is taken over the remaining entries).

    Recursive: result[i, j] is updated in place; subsequent pixels see the
    new value (as specified in the original paper).
    """
    h, w = noisy.shape
    result = noisy.copy()
    win = np.empty(9, np.float64)

    for i in range(h):
        for j in range(w):
            cv = result[i, j]
            if not is_noisy(cv):
                continue

            # Full 3x3 window (in-image entries), corrupted 0/255 included
            cc = 0
            for di in range(-1, 2):
                for dj in range(-1, 2):
                    ni = i + di
                    nj = j + dj
                    if 0 <= ni < h and 0 <= nj < w:
                        win[cc] = result[ni, nj]
                        cc += 1

            med = median_array(win, cc)
            if not is_noisy(med):
                # Window median is noise-free -> use it
                result[i, j] = med
            else:
                # Window median is also 0/255 -> previously processed pixel
                if j > 0:
                    result[i, j] = result[i, j - 1]
                elif i > 0:
                    result[i, j] = result[i - 1, j]
                # else: top-left corner -> leave

    return result


def run_dba(noisy):
    """
    Public API for DBA.

    Args:
        noisy: 2D numpy array (uint8 or float), with 0/255 SPN corruption.

    Returns:
        Denoised image as float64. Caller may cast to uint8 if needed.
    """
    return _dba_kernel(noisy.astype(np.float64))


# ============================================================================
# NAFSMF - Noise Adaptive Fuzzy Switching Median Filter
# Reference:
#   K.K.V. Toh, N.A. Mat Isa,
#   "Noise adaptive fuzzy switching median filter for salt-and-pepper noise
#   reduction,"
#   IEEE Signal Processing Letters, vol. 17, no. 3, pp. 281-284, March 2010.
#
# Notes:
#   - Two-stage:
#       Stage 1: histogram-based detection (we use the SPN convention that
#                values 0 and 255 are noise candidates).
#       Stage 2: adaptive window expansion (3->5->7) + fuzzy reasoning.
#   - Fuzzy thresholds T1=10, T2=30 (paper-recommended).
#   - Recursive: updated values feed back into subsequent windows. Paper
#     justifies using the upper-left 4 noise-free pixels (already restored)
#     for a more accurate median (Section II.B, Fig. 3 discussion). We
#     implement the equivalent simplification: collect noise-free values from
#     the recursive `result`, take the median of up to the first 4 found.
# ============================================================================
@njit(cache=True)
def _nafsmf_kernel(noisy, T1, T2, max_window):
    """
    NAFSMF core algorithm.

    For each pixel in raster-scan order:
      1. If pixel value is not 0/255: keep (detected as noise-free).
      2. Else: adaptive filtering
         a. Start with 3x3 window
         b. Collect noise-free values from current (recursive) state
         c. If count < 4 and window < max: expand to 5, 7, ...
         d. If at least one noise-free value found:
              M = median of up to first 4 noise-free values
              D = max |v - X_center| over noise-free values
              Fuzzy: f = 0 if D<T1
                     f = (D-T1)/(T2-T1) if T1<=D<T2
                     f = 1 if D>=T2
              Output = (1 - f) * X_center + f * M
         e. Else (no noise-free in max window): leave as is
    """
    h, w = noisy.shape
    result = noisy.copy()
    buf_size = max_window * max_window
    nf_vals = np.empty(buf_size, np.float64)

    for i in range(h):
        for j in range(w):
            cv = result[i, j]
            if cv != 0.0 and cv != 255.0:
                continue  # noise-free, keep

            # --- Stage 2: adaptive window expansion ---
            window_size = 3
            cc = 0
            while True:
                r = window_size // 2
                cc = 0
                for di in range(-r, r + 1):
                    for dj in range(-r, r + 1):
                        if di == 0 and dj == 0:
                            continue
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            v = result[ni, nj]
                            if v != 0.0 and v != 255.0:
                                nf_vals[cc] = v
                                cc += 1
                if cc >= 4 or window_size >= max_window:
                    break
                window_size += 2

            if cc >= 1:
                # Median of up to first 4 noise-free values
                use_cc = cc if cc < 4 else 4
                M = median_array(nf_vals, use_cc)

                # Local information D: max |v - X_center|
                # (paper formula 5: max abs diff between center and neighbors)
                D = 0.0
                for k in range(cc):  # use all collected noise-free for D
                    diff = abs(nf_vals[k] - cv)
                    if diff > D:
                        D = diff

                # Fuzzy reasoning (paper formula 6)
                if D < T1:
                    f = 0.0
                elif D < T2:
                    f = (D - T1) / (T2 - T1)
                else:
                    f = 1.0

                # Linear combination (paper formula 7)
                result[i, j] = (1.0 - f) * cv + f * M
            # else: keep (no noise-free in max window - very rare)

    return result


def run_nafsmf(noisy, T1=10.0, T2=30.0, max_window=7):
    """
    Public API for NAFSMF.

    Args:
        noisy:      2D numpy array, 0/255 SPN corruption.
        T1, T2:     fuzzy thresholds (paper: T1=10, T2=30).
        max_window: maximum adaptive window size (odd, paper uses up to 7).

    Returns:
        Denoised image as float64.
    """
    return _nafsmf_kernel(
        noisy.astype(np.float64),
        float(T1),
        float(T2),
        int(max_window),
    )


# ============================================================================
# DAMF - Different Applied Median Filter
# Reference:
#   U. Erkan, L. Gokrem, S. Enginoglu,
#   "Different applied median filter in salt and pepper noise,"
#   Computers & Electrical Engineering, vol. 70, pp. 789-798, August 2018.
#   doi: 10.1016/j.compeleceng.2018.01.019
#
# Notes:
#   - Adaptive window expansion until at least one noise-free value is found.
#   - Replace noisy pixel with median of noise-free values in window.
#   - Non-noisy pixels are left unchanged.
#   - Implementation mirrors Erkan's official MATLAB Central code (in-place
#     recursive update: noise pixels processed earlier in the scan provide
#     valid noise-free values for later windows).
#   - Default max_window=21 is a conservative upper bound on the window size.
# ============================================================================
@njit(cache=True)
def _damf_kernel(noisy, max_window):
    """
    DAMF core algorithm.

    For each pixel in raster-scan order:
      1. If pixel is not 0/255: keep unchanged.
      2. Else: expand window 3, 5, 7, ..., max_window until at least one
         noise-free value is in the window; replace center with median of
         those noise-free values.
      3. If even max_window has no noise-free values: leave as is (extreme
         case, only occurs when entire image is saturated locally).
    """
    h, w = noisy.shape
    result = noisy.copy()
    buf_size = max_window * max_window
    nf_vals = np.empty(buf_size, np.float64)

    for i in range(h):
        for j in range(w):
            cv = result[i, j]
            if cv != 0.0 and cv != 255.0:
                continue

            window_size = 3
            cc = 0
            while True:
                r = window_size // 2
                cc = 0
                for di in range(-r, r + 1):
                    for dj in range(-r, r + 1):
                        ni = i + di
                        nj = j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            v = result[ni, nj]
                            if v != 0.0 and v != 255.0:
                                nf_vals[cc] = v
                                cc += 1
                if cc > 0 or window_size >= max_window:
                    break
                window_size += 2

            if cc > 0:
                result[i, j] = median_array(nf_vals, cc)
            # else: extremely rare; keep as is

    return result


def run_damf(noisy, max_window=21):
    """
    Public API for DAMF.

    Args:
        noisy:      2D numpy array, 0/255 SPN corruption.
        max_window: maximum adaptive window size (odd; default 21).

    Returns:
        Denoised image as float64.
    """
    return _damf_kernel(noisy.astype(np.float64), int(max_window))
