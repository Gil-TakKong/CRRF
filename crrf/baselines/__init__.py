# -*- coding: utf-8 -*-
"""The nine compared methods with the settings of the paper.

    GRAY_METHODS    name -> function(2-D uint8 noisy channel) -> 2-D array; applied to each channel
    VECTOR_METHODS  name -> function(H x W x 3 uint8 noisy image) -> H x W x 3 array
    BASELINES       all nine names in the order of the tables
    run_baseline    run_baseline(name, noisy_u8) -> H x W x 3 uint8

Every method receives the noisy image only. The output type differs between methods (float64 or
uint8), and run_baseline applies the common conversion of the paper to every output.
"""
from collections import OrderedDict

import numpy as np

from ..metrics import to_uint8
from .armf import armf
from .ca_ngmf import run_ca_ngmf
from .fapgf import fapgf, fapgf_detail
from .filters import run_damf, run_dba, run_nafsmf
from .gsr import run_gsr
from .newmmf import run_newmmf
from .vmf import vmf

__all__ = ["GRAY_METHODS", "VECTOR_METHODS", "BASELINES", "per_channel", "run_baseline",
           "vmf", "fapgf", "fapgf_detail"]


def _gray_u8(g):
    g = np.asarray(g)
    if g.ndim != 2:
        raise ValueError(f"expected a 2-D channel: shape={g.shape}")
    if g.dtype != np.uint8:
        raise TypeError(f"expected a uint8 channel: dtype={g.dtype}")
    return g


def _dba(g):
    return run_dba(_gray_u8(g).astype(np.float64))


def _nafsmf(g):
    return run_nafsmf(_gray_u8(g).astype(np.float64), T1=10.0, T2=30.0, max_window=7)


def _damf(g):
    return run_damf(_gray_u8(g).astype(np.float64), max_window=21)


def _ca_ngmf(g):
    return run_ca_ngmf(_gray_u8(g), max_iterations=1000)


def _gsr(g):
    return run_gsr(_gray_u8(g))


def _armf(g):
    return armf(_gray_u8(g))


def _newmmf(g):
    return run_newmmf(_gray_u8(g))


def _fapgf(x):
    # The paper states no boundary rule; crop reproduces its Table 3 best.
    return fapgf(x, d=0.1, gamma=0.8, border="crop")


GRAY_METHODS = OrderedDict([
    ("DBA",     _dba),
    ("NAFSMF",  _nafsmf),
    ("DAMF",    _damf),
    ("CA-NGMF", _ca_ngmf),
    ("GSR",     _gsr),
    ("ARmF",    _armf),
    ("NEWMMF",  _newmmf),
])

VECTOR_METHODS = OrderedDict([
    ("VMF",   vmf),
    ("FAPGF", _fapgf),
])

BASELINES = list(GRAY_METHODS) + list(VECTOR_METHODS)


def per_channel(fn, noisy_u8):
    """Apply a grayscale method to each channel independently and stack the results on axis 2."""
    x = np.asarray(noisy_u8)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError(f"expected an H x W x 3 image: shape={x.shape}")
    if x.dtype != np.uint8:
        raise TypeError(f"expected a uint8 image: dtype={x.dtype}")
    outs = []
    for c in range(3):
        ch = np.ascontiguousarray(x[:, :, c])
        if np.shares_memory(ch, x):                 # a contiguous slice still gets its own copy
            ch = ch.copy()
        outs.append(np.asarray(fn(ch)))
    return np.stack(outs, axis=2)


def run_baseline(name, noisy_u8):
    """One compared method on an H x W x 3 uint8 noisy image, with the 8-bit output of the paper."""
    if name in GRAY_METHODS:
        return to_uint8(per_channel(GRAY_METHODS[name], noisy_u8))
    if name in VECTOR_METHODS:
        return to_uint8(VECTOR_METHODS[name](noisy_u8))
    raise ValueError(f"unknown method {name!r}; available: {BASELINES}")
