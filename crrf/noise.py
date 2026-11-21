# -*- coding: utf-8 -*-
"""Channel-independent salt-and-pepper noise and the seed rule of the paper.

In each channel exactly round(d*H*W) positions are drawn without replacement, independently of the
other two channels, and each drawn position becomes 0 or 255 with probability 1/2. Channels are
processed in array order (B, G, R for images read by OpenCV).
"""
import numpy as np

DENSITIES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]      # iota is the index in this list
SEED_BASE = 42
SEED_R = 100000
SEED_I = 100


def draw_noise(h, w, density, rng):
    """Positions and values of one noise image. Returns (mask, values), both of shape (h, w, 3).

    mask[x, y, c] tells whether the noise process wrote this component, and values holds what it
    wrote (0.0 or 255.0) there. The written value may equal the clean value.
    """
    n = h * w
    k = int(round(density * n))
    mask = np.zeros((h, w, 3), np.bool_)
    values = np.zeros((h, w, 3), np.float64)
    for c in range(3):
        idx = rng.choice(n, size=k, replace=False)
        val = np.where(rng.random(k) < 0.5, 0.0, 255.0)
        mask[:, :, c].flat[idx] = True
        values[:, :, c].flat[idx] = val
    return mask, values


def add_noise(image, density, rng):
    """Noisy copy of a float64 or uint8 image, as float64."""
    clean = np.ascontiguousarray(np.asarray(image).astype(np.float64))
    h, w, _ = clean.shape
    mask, values = draw_noise(h, w, density, rng)
    return np.ascontiguousarray(np.where(mask, values, clean))


def seed_of(r, i, iota):
    """Seed of realization r, image index i (lexicographic file order), density index iota."""
    return SEED_BASE + SEED_R * int(r) + SEED_I * int(i) + int(iota)


def density_index(density):
    """Index iota of a density value in DENSITIES."""
    for iota, value in enumerate(DENSITIES):
        if abs(value - float(density)) < 1e-9:
            return iota
    raise ValueError(f"density {density} is not in {DENSITIES}")


def make_noisy(clean, r, i, iota):
    """Noisy image of the paper for one (realization, image index, density index). Returns (noisy, seed)."""
    seed = seed_of(r, i, iota)
    return add_noise(clean, DENSITIES[iota], np.random.default_rng(seed)), seed
