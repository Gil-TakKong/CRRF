# -*- coding: utf-8 -*-
"""CRRF: Color Ratio Refinement Filter for channel-independent salt-and-pepper noise.

    from crrf import crrf
    restored = crrf(noisy_uint8)

Public API
    crrf, phase1_only, identify      the method (see crrf/core.py)
    add_noise, draw_noise            the noise model (see crrf/noise.py)
    make_noisy, seed_of, DENSITIES   the noise images of the paper
    cpsnr, ssim, ncd, fsimc, to_uint8  the evaluation (see crrf/metrics.py)
"""
import os
import tempfile

# Numba's cache=True fails on long non-ASCII paths, so the default cache lives under the system temp
# folder. This is the only place that sets it and it must run before the first numba import.
os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache_crrf"))

from .core import crrf, identify, phase1, phase1_only, warmup             # noqa: E402
from .metrics import cpsnr, fsimc, ncd, ssim, to_uint8                    # noqa: E402
from .noise import DENSITIES, add_noise, draw_noise, make_noisy, seed_of  # noqa: E402
from .phase2 import phase2                                                # noqa: E402

__version__ = "1.0.0"
__all__ = ["crrf", "phase1_only", "identify", "phase1", "phase2", "warmup",
           "add_noise", "draw_noise", "make_noisy", "seed_of", "DENSITIES",
           "cpsnr", "ssim", "ncd", "fsimc", "to_uint8"]
