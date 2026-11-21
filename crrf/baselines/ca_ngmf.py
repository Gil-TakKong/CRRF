#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ca_ngmf.py - CA-NGMF (compared method).

Reference:
  G.-T. Kong, U.-S. Choi, A cellular automata-based method for
  salt-and-pepper noise removal in images, Multimedia Tools and
  Applications 85(3) (2026) Article 247.

Rule (one synchronous pass):
  For each pixel of value 0 or 255, the four von Neumann neighbor groups
  (C = the pixel itself) are examined in this fixed order:
      G1: N, W, C, E      G2: W, N, C, S
      G3: W, C, E, S      G4: S, C, N, E
  The first group with at least two in-image values that are not in {0, 255}
  replaces the pixel with int(mean) of those values (truncation).
  Out-of-image neighbors are ignored; the center is itself 0 or 255 and is
  therefore never counted. If no group qualifies, the pixel is left unchanged.

Iteration and stop rules:
  Passes are repeated until no pixel of value 0 or 255 remains, until
  max_iterations passes (default 1000), or until the number of 0/255 pixels
  has been identical for the last 10 of more than 10 recorded passes.

Input and output are uint8.

Public API:
  - run_ca_ngmf(noisy, max_iterations=1000)
"""

import numpy as np
from numba import njit


# Group offsets in the order G1..G4; (0, 0) is the center "C".
#   Group 1: N, W, C, E    Group 2: W, N, C, S
#   Group 3: W, C, E, S    Group 4: S, C, N, E
_GROUP_DR = np.array([[-1, 0, 0, 0],
                      [0, -1, 0, 1],
                      [0, 0, 0, 1],
                      [1, 0, -1, 0]], np.int64)
_GROUP_DC = np.array([[0, -1, 0, 1],
                      [-1, 0, 0, 0],
                      [-1, 0, 1, 0],
                      [0, 0, 0, 1]], np.int64)


@njit(cache=True)
def _ca_ngmf_kernel(noisy, max_iterations):
    """
    CA-NGMF core algorithm.

    Each pass reads from the previous state `img` and writes to `nxt`.
    For each noisy pixel, the first group with at least two non-noise
    in-image values gives int(mean) (truncation). The center "C" is the
    noisy pixel itself and is therefore never counted.

    Stops when no 0/255 pixel remains, after max_iterations passes, or when
    the noise count has been identical for the last 10 of more than 10
    recorded passes.
    """
    h, w = noisy.shape
    img = noisy.copy()
    iteration_count = 0
    hist_len = 0
    last_count = -1
    same_run = 0

    while iteration_count < max_iterations:
        noise_count = 0
        for i in range(h):
            for j in range(w):
                if img[i, j] == 0 or img[i, j] == 255:
                    noise_count += 1
        if noise_count == 0:
            break

        # Safety: 10 consecutive identical noise_count -> break
        hist_len += 1
        if noise_count == last_count:
            same_run += 1
        else:
            same_run = 1
            last_count = noise_count
        if hist_len > 10 and same_run >= 10:
            break

        nxt = img.copy()
        for i in range(h):
            for j in range(w):
                if img[i, j] != 0 and img[i, j] != 255:
                    continue
                for g in range(4):
                    s = 0.0
                    cnt = 0
                    for k in range(4):
                        ni = i + _GROUP_DR[g, k]
                        nj = j + _GROUP_DC[g, k]
                        if 0 <= ni < h and 0 <= nj < w:
                            v = img[ni, nj]
                            if v != 0 and v != 255:
                                s += v
                                cnt += 1
                    if cnt >= 2:
                        nxt[i, j] = int(s / cnt)
                        break
        img = nxt
        iteration_count += 1

    return img


def run_ca_ngmf(noisy, max_iterations=1000):
    """
    Public API for CA-NGMF.

    Args:
        noisy:          2D numpy array, 0/255 SPN corruption. It is clipped to
                        [0, 255] and cast to uint8 (callers pass
                        to_uint8(noisy)).
        max_iterations: maximum number of passes (default 1000).

    Returns:
        Denoised image as uint8.

    Notes:
        - Safety: if noise_count is unchanged for 10 consecutive iterations,
          break (prevents infinite loop in pathological cases).
    """
    return _ca_ngmf_kernel(np.clip(noisy, 0, 255).astype(np.uint8),
                           int(max_iterations))
