# -*- coding: utf-8 -*-
"""
ARmF (Adaptive Riesz Mean Filter), compared per channel.

Reference:
  S. Enginoglu, U. Erkan, S. Memis, "Pixel similarity-based adaptive Riesz
  mean filter for salt-and-pepper noise removal," Multimedia Tools and
  Applications 78 (2019) 35401-35418. doi:10.1007/s11042-019-08110-1
Python port of the authors' MATLAB code ARmF.m (https://github.com/serdarenginoglu/ARmF).
The 8-bit conversion follows MATLAB uint8(): floor(A + 0.5) for non-negative values.

Algorithm:
  Five passes with p = 5, 4, 3, 2, 1. For each pixel of value 0 or 255 in the snapshot taken at the
  start of the pass, the smallest (2k+1)x(2k+1) window (k <= p) with at least one clean pixel gives a
  Riesz mean with weights (1/(1+|ds|+|dt|))^2. Reads use the snapshot, writes go to A.
  The boundary follows MATLAB padarray 'symmetric'.
"""
import numpy as np
from numba import njit


# ============================================================
# Symmetric reflection padding (matches MATLAB padarray 'symmetric')
# ============================================================
@njit(cache=True, inline='always')
def _ref(idx, size):
    if idx < 0:
        return -idx - 1
    elif idx >= size:
        return 2*size - 1 - idx
    return idx


@njit(cache=True, inline='always')
def _is_clean(v):
    return (v != 0.0) and (v != 255.0)


# ============================================================
# ARmF core (Numba JIT)
# ============================================================
@njit(cache=True)
def _armf_pass(A, snapshot, h, w, p):
    """
    Single ARmF pass with max window radius p.

    For each noisy pixel in snapshot:
      - Find smallest k in [1, p] such that (2k+1)x(2k+1) window
        contains at least one clean pixel
      - Compute Riesz mean over clean pixels with weight (1/(1+d))^2
        where d = |drow| + |dcol| (Manhattan distance from center)
      - Write result to A[x, y]

    Reads come from snapshot (frozen at pass start), writes go to A.
    Mirrors MATLAB semantics where pA is created once per p.
    """
    for x in range(h):
        for y in range(w):
            # Skip if pixel is not noisy
            if _is_clean(snapshot[x, y]):
                continue

            # Find smallest k where window has >= 1 clean pixel
            best_k = 0
            for k in range(1, p + 1):
                has_clean = False
                for ds in range(-k, k + 1):
                    for dt in range(-k, k + 1):
                        nx = _ref(x + ds, h)
                        ny = _ref(y + dt, w)
                        if _is_clean(snapshot[nx, ny]):
                            has_clean = True
                            break
                    if has_clean:
                        break
                if has_clean:
                    best_k = k
                    break

            if best_k == 0:
                # No clean pixel found in any window up to k=p
                continue

            # Compute Riesz mean over clean pixels in (2k+1)x(2k+1) window
            k = best_k
            Rz = 0.0
            TRz = 0.0
            for ds in range(-k, k + 1):
                for dt in range(-k, k + 1):
                    nx = _ref(x + ds, h)
                    ny = _ref(y + dt, w)
                    v = snapshot[nx, ny]
                    if _is_clean(v):
                        # ps weight: (1 / (1 + |ds| + |dt|))^2
                        # (Manhattan distance from center, despite paper's
                        #  misleading "pixel similarity" naming)
                        d = abs(ds) + abs(dt)
                        w_ps = 1.0 / (1.0 + d)
                        w_ps = w_ps * w_ps
                        Rz += v * w_ps
                        TRz += w_ps

            if TRz > 0.0:
                A[x, y] = Rz / TRz


def armf(noisy_u8):
    """ARmF on a 2-D uint8 image. Returns uint8, rounded like MATLAB uint8()."""
    A = noisy_u8.astype(np.float64).copy()
    h, w = A.shape
    # Outer loop: p = 5, 4, 3, 2, 1 (matches MATLAB for p=5:-1:1)
    for p in range(5, 0, -1):
        snapshot = A.copy()
        _armf_pass(A, snapshot, h, w, p)
    return np.clip(np.floor(A + 0.5), 0, 255).astype(np.uint8)
