# -*- coding: utf-8 -*-
"""
FAPGF (Fast Averaging Peer Group Filter).

Reference:
  L. Malinski, B. Smolka, "Fast averaging peer group filter for the impulsive noise removal in color
  images," Journal of Real-Time Image Processing 11(3) (2016) 427-444. doi:10.1007/s11554-015-0500-z

Definition (section and equation numbers of that paper):
  Vectors    A pixel is one vector. Distances and means do not depend on the channel order.
  Distance   rho(a, b) = ||a - b||_2 / (255 * sqrt(3))  (Sect. 2). For 8-bit values, a and b are close
             when ||a - b||_2^2 < (d * 255 * sqrt(3))^2, which is 1950.75 for d = 0.1.
  Peer group m_k = number of the 8 non-center samples of the 3x3 window with rho(x_k, x_j) < d (Eq. 1).
             All m are computed on the noisy input before filtering (not recursive).
  Detection  m_1 >= 2 keeps the pixel; m_1 <= 1 replaces it with the WAF output (Sect. 2).
  WAF        mu_i = m_i ^ gamma, y_1 = sum_i mu_i x_i / sum_i mu_i (Eqs. 2, 3), over the non-center
             window samples with their own 3x3 peer group sizes. A sample with m_i = 0 has no weight.
  Enlarging  If all 8 samples of the 3x3 window have m_i = 0, the window grows to 5x5, 7x7, ... until
             it holds at least 2 non-center samples with m_i >= 2, and Eqs. 2-3 apply to all its
             non-center samples.
  Parameters d = 0.1, gamma = 0.8 (the setting of all comparisons of the paper; not tuned)
  Boundary   Not specified in the paper. border="crop" clips the window to the image and reproduces
             Table 3 of the paper best (see README.md), so it is used for the
             comparison. border="reflect101" uses reflect_101, repeating with period 2(n-1).
  Safeguard  (not in the paper) If the window reaches radius max(H, W) - 1 without meeting the stop
             rule, it stops there. A positive weight sum then gives that window's mean; a zero sum
             keeps the center pixel.
  Input is H x W x 3 uint8; the output is H x W x 3 float64 (the caller converts to 8 bits).

A replaced pixel changes in all three components, so clean components can change as well, while a
corrupted pixel with m >= 2 stays.
"""

import math

import numpy as np
from numba import njit

D_DEFAULT = 0.1         # normalized distance threshold (Sect. 4-5 of the paper)
GAMMA_DEFAULT = 0.8     # WAF exponent (Sect. 4-5 of the paper)
BORDERS = ("reflect101", "crop")


def _thr2(d):
    """Threshold on the squared 8-bit RGB distance: close <=> ||a - b||_2^2 < thr2."""
    t = float(d) * 255.0 * math.sqrt(3.0)
    return t * t


def _mu_table(gamma):
    """mu(m) = m ^ gamma for m = 0..8 (the 3x3 window has 8 non-center samples)."""
    return np.array([float(k) ** float(gamma) for k in range(9)], np.float64)


@njit(cache=True)
def _reflect101(i, n):
    """reflect_101 boundary (n >= 2), repeating with period 2(n-1) for windows larger than the image."""
    p = 2 * (n - 1)
    i = i % p
    if i < 0:
        i += p
    if i >= n:
        i = p - i
    return i


@njit(cache=True)
def _peer_sizes(x, thr2, crop):
    """Eq. (1): number of the 8 non-center 3x3 samples with squared distance < thr2."""
    h = x.shape[0]
    w = x.shape[1]
    m = np.zeros((h, w), np.int64)
    for i in range(h):
        for j in range(w):
            c = 0
            for di in range(-1, 2):
                ii = i + di
                if crop:
                    if ii < 0 or ii >= h:
                        continue
                else:
                    ii = _reflect101(ii, h)
                for dj in range(-1, 2):
                    if di == 0 and dj == 0:
                        continue
                    jj = j + dj
                    if crop:
                        if jj < 0 or jj >= w:
                            continue
                    else:
                        jj = _reflect101(jj, w)
                    d0 = x[i, j, 0] - x[ii, jj, 0]
                    d1 = x[i, j, 1] - x[ii, jj, 1]
                    d2 = x[i, j, 2] - x[ii, jj, 2]
                    if d0 * d0 + d1 * d1 + d2 * d2 < thr2:
                        c += 1
            m[i, j] = c
    return m


@njit(cache=True)
def _fapgf_kernel(x, m, mu, crop, rmax):
    """
    Replace the pixels with m <= 1 by the WAF output (Eqs. 2, 3 and window enlarging).
    radius: 0 = kept (m >= 2), r >= 1 = replaced from a (2r+1)x(2r+1) window,
            -1 = m <= 1 but the largest window has no weight, so the pixel stays (safeguard).
    """
    h = x.shape[0]
    w = x.shape[1]
    out = np.empty((h, w, 3), np.float64)
    radius = np.zeros((h, w), np.int64)
    for i in range(h):
        for j in range(w):
            if m[i, j] >= 2:
                out[i, j, 0] = x[i, j, 0]
                out[i, j, 1] = x[i, j, 1]
                out[i, j, 2] = x[i, j, 2]
                continue
            r = 1
            sw = 0.0
            a0 = 0.0
            a1 = 0.0
            a2 = 0.0
            while True:
                sw = 0.0
                a0 = 0.0
                a1 = 0.0
                a2 = 0.0
                ngood = 0
                for di in range(-r, r + 1):
                    ii = i + di
                    if crop:
                        if ii < 0 or ii >= h:
                            continue
                    else:
                        ii = _reflect101(ii, h)
                    for dj in range(-r, r + 1):
                        if di == 0 and dj == 0:
                            continue
                        jj = j + dj
                        if crop:
                            if jj < 0 or jj >= w:
                                continue
                        else:
                            jj = _reflect101(jj, w)
                        mi = m[ii, jj]
                        if mi >= 2:
                            ngood += 1
                        if mi > 0:
                            wgt = mu[mi]
                            sw += wgt
                            a0 += wgt * x[ii, jj, 0]
                            a1 += wgt * x[ii, jj, 1]
                            a2 += wgt * x[ii, jj, 2]
                if r == 1:
                    done = sw > 0.0
                else:
                    done = ngood >= 2
                if done or r >= rmax:
                    break
                r += 1
            if sw > 0.0:
                out[i, j, 0] = a0 / sw
                out[i, j, 1] = a1 / sw
                out[i, j, 2] = a2 / sw
                radius[i, j] = r
            else:
                out[i, j, 0] = x[i, j, 0]
                out[i, j, 1] = x[i, j, 1]
                out[i, j, 2] = x[i, j, 2]
                radius[i, j] = -1
    return out, radius


def _check(x, border):
    x = np.asarray(x)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"expected an H x W x 3 image: shape={x.shape}")
    if x.dtype != np.uint8:
        raise TypeError(f"expected a uint8 image: dtype={x.dtype}")
    if border not in BORDERS:
        raise ValueError(f"border must be one of {BORDERS}: {border!r}")
    if border == "reflect101" and (x.shape[0] < 2 or x.shape[1] < 2):
        raise ValueError("the reflect_101 boundary needs a height and a width of at least 2")
    if x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("empty image")
    return x


def _check_params(d, gamma):
    if not (0.0 < float(d) <= 1.0):
        raise ValueError(f"d must lie in (0, 1]: {d}")
    if not (float(gamma) > 0.0):
        raise ValueError(f"gamma must be positive: {gamma}")


def fapgf_detail(noisy_u8, d=D_DEFAULT, gamma=GAMMA_DEFAULT, border="reflect101"):
    """
    The computation of fapgf, returned with diagnostics as a dict:
      out       H x W x 3 float64  filter output
      m         H x W int64        peer group sizes (Eq. 1)
      replaced  H x W bool         pixels replaced by a WAF output (m <= 1 and a positive weight sum)
      radius    H x W int64        0 = kept, r >= 1 = replaced from a (2r+1)x(2r+1) window, -1 = safeguard
    """
    x8 = _check(noisy_u8, border)
    _check_params(d, gamma)
    x = np.ascontiguousarray(x8, dtype=np.float64)
    crop = border == "crop"
    h, w = x.shape[0], x.shape[1]
    rmax = max(h - 1, w - 1, 1)
    m = _peer_sizes(x, _thr2(d), crop)
    out, radius = _fapgf_kernel(x, m, _mu_table(gamma), crop, rmax)
    return dict(out=out, m=m, replaced=radius > 0, radius=radius)


def fapgf(noisy_u8, d=D_DEFAULT, gamma=GAMMA_DEFAULT, border="reflect101"):
    """FAPGF (d = 0.1, gamma = 0.8, 3x3). Input H x W x 3 uint8, output H x W x 3 float64."""
    return fapgf_detail(noisy_u8, d=d, gamma=gamma, border=border)["out"]
