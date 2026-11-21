# -*- coding: utf-8 -*-
"""The CRRF method: identification, Phase 1 and the public entry points.

Channels are indexed as given (B, G, R for images read by OpenCV). The method treats the three
channels symmetrically, so the order only matters for reproducing the addition order of the paper's
runs. A component is identified as noise when its value is 0 or 255. Uncorrupted components are
never changed.
"""
import numpy as np
from numba import njit

from .geometry import ref101
from .metrics import to_uint8
from .phase2 import phase2

MAX_ITER = 500          # safety cap of Phase 1; not reached on any image of the paper


def identify(noisy):
    """Boolean mask of components equal to 0 or 255 (any value <= 0 or >= 255 for float input)."""
    noisy = np.asarray(noisy)
    return np.ascontiguousarray((noisy <= 0) | (noisy >= 255))


@njit(cache=True)
def _phase1_channel(channel, marked, h, w):
    """Synchronous (Jacobi) restoration of one channel.

    In each iteration every marked component with at least one valid neighbor among its four von
    Neumann neighbors takes their mean and becomes valid for the next iteration. Reads use the state
    of the previous iteration. The loop stops at the first iteration that restores nothing, which is
    counted in the returned number of iterations.
    """
    img = channel.copy()
    valid = np.empty((h, w), np.bool_)
    for x in range(h):
        for y in range(w):
            valid[x, y] = not marked[x, y]

    n_iter = 0
    for it in range(MAX_ITER):
        nxt = img.copy()
        vnx = valid.copy()
        changed = 0
        for x in range(h):
            for y in range(w):
                if valid[x, y]:
                    continue
                s = 0.0
                c = 0
                # the addition order N, W, E, S is part of the bit-exact behavior
                for dx, dy in ((-1, 0), (0, -1), (0, 1), (1, 0)):
                    nx = ref101(x + dx, h)
                    ny = ref101(y + dy, w)
                    if valid[nx, ny]:
                        s += img[nx, ny]
                        c += 1
                if c >= 1:
                    nxt[x, y] = s / c
                    vnx[x, y] = True
                    changed += 1
        img = nxt
        valid = vnx
        n_iter = it + 1
        if changed == 0:
            break

    left = 0
    for x in range(h):
        for y in range(w):
            if not valid[x, y]:
                left += 1
    return img, n_iter, left


def phase1(noisy, marked):
    """Phase 1 on a float64 image. Returns (estimate, iterations, unrestored components)."""
    h, w, _ = noisy.shape
    out = np.empty_like(noisy)
    n_iter = 0
    left = 0
    for c in range(3):
        rest, it, nl = _phase1_channel(np.ascontiguousarray(noisy[:, :, c]),
                                       np.ascontiguousarray(marked[:, :, c]), h, w)
        out[:, :, c] = rest
        n_iter = max(n_iter, it)
        left += nl
    return np.ascontiguousarray(out), n_iter, left


def _as_float(noisy):
    noisy = np.asarray(noisy)
    if noisy.ndim != 3 or noisy.shape[2] != 3:
        raise ValueError("expected an H x W x 3 image")
    return np.ascontiguousarray(noisy.astype(np.float64))


def phase1_only(noisy):
    """Phase 1 alone on an 8-bit noisy image (configuration 'Phase 1' of the paper). Returns uint8."""
    noisy_f = _as_float(noisy)
    return to_uint8(phase1(noisy_f, identify(noisy_f))[0])


def crrf(noisy):
    """CRRF on an 8-bit H x W x 3 noisy image, H, W >= 4. Returns the restored uint8 image."""
    noisy_f = _as_float(noisy)
    marked = identify(noisy_f)
    estimate = phase1(noisy_f, marked)[0]
    return to_uint8(phase2(estimate, marked)[0])


def warmup():
    """Compile (or load from the numba cache) Phase 1 and Phase 2, so that timing starts clean."""
    rng = np.random.default_rng(0)
    img = rng.integers(20, 236, (12, 11, 3)).astype(np.float64)
    img[rng.random(img.shape) < 0.3] = 255.0
    crrf(img)
