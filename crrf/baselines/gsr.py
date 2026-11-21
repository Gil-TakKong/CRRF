# -*- coding: utf-8 -*-
"""
gsr.py
======

GSR (compared method) --- faithful Python port of the salt-and-pepper
(SPN) removal method of

    Q. Zhang, C. Huang, L. Yang, Z. Yang,
    "Salt and pepper noise removal method based on graph signal reconstruction",
    Digital Signal Processing 135 (2023) 103941.

The port follows the authors' released MATLAB code (GitHub branches
``functions`` / ``experiments``), *not* the equations printed in the paper,
wherever the two disagree.  Every such disagreement, and every deviation this
port makes from the MATLAB code, is listed in the two tables below.

Entry point
-----------
    run_gsr(noisy_uint8) -> restored_uint8

2-D grayscale ``uint8`` in, ``uint8`` out, same shape.  No clean image, no
noise density, no RNG.  All parameters are fixed constants taken verbatim from
``exp1.m`` .. ``exp4.m``::

    bh = bw = 8,  beta = 0.001,  sigma1 = 0.1,  sigma2 = 2500,
    fun(x) = 1/(x**2 + eps),  RAMF max radius = 21

MATLAB pipeline being reproduced (identical in exp1/exp2/exp3/exp4)::

    I        = imnoise(img,'salt & pepper',d);
    img_RAMF = RAMF(I,21);
    Mask     = (img_RAMF~=I) & (I==0 | I==255);   mask = ~Mask;
    emask    = expandimg(mask,4,4);  img = expandimg(img_RAMF,4,4);
    img_PSD  = denoise(img,emask,8,8,0.001,0.1,2500,@(x)1./(x.^2+eps));
    img_PSD  = uint8(smblock(img_PSD,mask,8,8));


Code-vs-paper discrepancies (reproduced as in the code)
-------------------------------------------------------
P1  Edge weights.  The paper's Eq. (3.18) is exp(-||.||_2^2 / sigma^2) and
    Section 4 sets *sigma1^2 = 0.1* and *sigma2^2 = 2500* (i.e. the printed
    constants are already the squared ones, so the paper's kernel is
    exp(-||.||^2 / 0.1)).  The released code instead divides the feature rows
    by 0.1 and 2500 and applies MATLAB ``pdist``, which is the *un-squared*
    Euclidean distance, so it computes exp(-||.|| / 0.1): it treats the paper's
    sigma^2 values as sigma and drops the square in the exponent.  The two
    kernels coincide only at distance 1.  The port follows the code.
P2  Weight function: paper w(x) = 1/x^2; code 1/(x^2 + eps).
P3  Linear system: paper Eq. (2.13) (Ik'Ik + beta*L) f = Ik' y_k; code solves
    (Ek/lbs + beta*KL/ws) f = (Ek.*s)/lbs, i.e. beta rescaled by lbs/ws with
    ws = 4032 -- the effective beta is 2.5e-7 .. 1.6e-5, not 0.001.
P4  "AMF maximum window size 21" means maximum *radius* 21, i.e. windows up to
    43x43.
P5  "RAMF" is implemented non-recursively (it reads only the padded input and
    writes to a separate array), despite the name and despite amf.m existing.
P6  The reconstructed signal is the residual of the block against the *mean of
    the 81 neighbourhood blocks*, not a DC-removed block.
P7  The 64 graph-node feature rows are built in row-major order over the 8x8
    grid while the signal / mask / covariance are column-major (MATLAB ``(:)``).
    Node p therefore carries the structural features of the transposed pixel
    location.  The coordinate columns are unaffected (W1 is symmetric under the
    swap), the four gradient columns are not, so W2 -- and hence L, U, psd --
    differ from the "index-consistent" variant.  This is an apparent bug in the
    released code, but it is what produced the published numbers, so it is
    reproduced verbatim.  *Measured cost of reproducing it*:
    against an index-consistent variant (feat reshaped (b,8,8,4) ->
    transpose(0,2,1,3)), PSNR moves by only -0.02 .. +0.06 dB (three test
    images at d = 0.5 and 0.9, 3 seeds), although 57 % of the
    output pixels change (Baboon256 d = 0.7: 37090/65536, max |diff| 40 grey
    levels).  It is a code-reading note, not a defect worth reporting about the
    method in a paper.
    P7 does *not* break transpose equivariance: transposing the image swaps the
    column-major signal order with the row-major feature order, which is a
    single consistent relabelling of the 64 nodes, and the Euclidean pdist over
    the four gradient columns is invariant to the induced kir1<->kir2,
    kir3<->kir4 column permutation.  Measured: PSNR(f(x)) == PSNR(f(x^T)^T) to
    three decimals at 512x512, 480x320 and 481x321, d = 0.5 and 0.9 (1-10
    pixels out of 1.5e5 differ, from floating-point tie-breaking only).
P8  The PSD is estimated from the AMF output, and the reconstruction replaces
    *all* 64 pixels of a block, trusted ones included.

Deviations of this port from the MATLAB code
--------------------------------------------
D1  Arbitrary image size.  The MATLAB code is square-only and multiple-of-8
    only: ``denoise.m`` swaps m/n in its loop bounds, ``resblocks.m`` swaps the
    row/column extents, ``denoise.m`` calls resblocks with sqrt(#blocks)*8, and
    ``smblock.m`` uses a width-derived vector to index rows and ``1:h`` as the
    row-pass query grid.  Here the row/column roles are kept separate
    throughout, and the image is symmetric-padded on the *bottom and right
    only* to a multiple of 8 before blocking, then cropped back.  Bottom/right
    padding keeps the 8x8 block grid anchored at the original origin, so for a
    square image whose side is a multiple of 8 the output is identical to
    MATLAB's; for other sizes the deviation is confined to the <=7 discarded
    rows/columns and to smblock's boundary rule on the last block row/column.
D2  MATLAB ``round`` (half away from zero) is reproduced explicitly; numpy's
    banker's rounding is not used.
D3  smblock hardening.  The original drops NaN positions from ``xx`` but not
    from the trusted-pixel index set, so one non-finite trusted sample turns an
    entire row (or column) into NaN.  Here non-finite positions are dropped
    from *both* sets.  Also ``~isfinite`` is tested rather than only ``isnan``,
    because an ill-conditioned solve can produce +-Inf (exp4.m runs with
    ``warning('off','all')``).
D4  Degenerate PSD.  If a 16x16 neighbourhood is exactly constant, cov(s1')=0,
    psd/sum(psd) = 0/0 and MATLAB emits an all-NaN block.  Here such a block is
    passed through unchanged (its AMF values), which is the correct answer for
    a constant block.  Set ``degenerate_psd='nan'`` for the literal MATLAB
    behaviour.  This never triggers on the authors' test images.
D5  ``inv(V)*b`` is replaced by ``np.linalg.solve(V,b)``.  V is SPD but badly
    conditioned: cond(V) measured over real blocks of a test image runs 7e9 .. 2e13
    (median 3e10 at d = 0.5, 2e12 at d = 0.9).  Literal MATLAB ``inv`` and the
    batched ``solve`` therefore differ by up to ~5e-3 grey levels in the float
    image (median ~5e-15); the rounded uint8 output has been bit-identical in
    every test, but do not read "identical" as a 1e-8 tolerance.
    A per-block ``lstsq`` retry runs when the batched solve raises, or when it
    returns a non-finite f for a block whose V and rhs were finite.  Blocks
    whose V or rhs is itself non-finite are *not* retried -- they join the
    pass-through set below and keep their AMF values.  (MATLAB would emit
    NaN/Inf there and leave it to smblock; the path is unreachable on real
    data, see D9.)
D6  RAMF is computed with a vectorised min/max/median-filter sweep for small
    radii and an exact per-pixel sweep for the remaining pixels.  The result is
    bit-identical to the MATLAB double loop (scipy's odd-window median equals
    MATLAB's ``W(ceil(K/2))``).
D7  ``reconstructer.m``'s unreachable ``isempty(labels)`` branch (which would
    build a row vector and corrupt the concatenation) is not ported; the
    reachable ``isempty(idx)`` early return is.
D8  The 81 neighbourhood blocks are stacked ii-major (row offset slowest),
    whereas ``denoise.m`` lines 25-28 loop ``for jj ... for ii ...`` and so
    stacks them jj-major.  Inert: ``bs`` is consumed only through
    ``mean(bs,2)``, ``cov(s1')`` and the centre column, all invariant to a
    permutation of the 81 columns, and the centre index is 4*9+4 = 40 in both
    orderings.  Flagged only because the rest of the port claims MATLAB ``(:)``
    ordering, so an auditor would otherwise read this as a mismatch.
D9  Two defensive branches that cannot fire on real data.  (a)
    ``reconstructer.m``'s ``max(w)`` ignores NaN; the port uses ``np.nanmax``
    to match, so ``degenerate_psd='nan'`` is literal MATLAB for a partially-NaN
    psd as well as for a fully-NaN one.  (b) ``_spline_line`` falls back to
    piecewise-linear ``np.interp`` when fewer than 4 knots survive, where
    MATLAB's ``interp1(...,'spline')`` would fit a line (2 knots) or a parabola
    (3).  The keep vector alone contributes >= 7 knots to every row and column,
    so this needs most of a line to be non-finite.

Validation status (all runs: our own noise protocol, 3 realisations per cell)
----------------------------------------------------------------------------
The authors published two disjoint result sets and the port agrees with one of
them but not the other.  State this, do not claim blanket agreement with "the
paper's results".

  psnrs.mat (exp3.m; barbara256/baboon256/couple256/pepper256, d=0.1..0.9)
  -- REPRODUCED.  20 cells, PSD PSNR deviation mean -0.011 dB,
  max 0.177 dB; PSD-over-AMF gain max 0.148 dB; noisy and AMF rows within
  0.20 dB.  Includes the low densities d = 0.1 and 0.3.

  snr1s.mat / snr2s.mat (exp2.m; walkbridge/pepper/mandril/lake/jetplane512,
  d=0.3..0.9) -- source of the paper's Table 1 -- NOT
  reproduced at low density.  The AMF row matches (snr2 mean -0.024 dB, max
  0.115 dB over all 20 cells), so the noise model, RAMF and the metric are
  correct, but the OURS row diverges with density: d=0.3 +1.01..+2.71 dB,
  d=0.5 +0.06..+1.34, d=0.7 -0.32..+0.10, d=0.9 -0.51..+0.02.

This is not a porting defect: the same code reproduces psnrs.mat at d = 0.1 and
0.3 to within 0.18 dB.  The published exp2 d=0.3 slice is internally odd -- its
gains over AMF (0.20 dB on pepper512, 0.48 on lake512) are far below the
1.2-8.0 dB the same authors report for every exp3 image at the same density,
and OURS there sits below TSM/PDA/PADMM while being best at d=0.7/0.9.  Likely
Table 1 predates the released PSD step (my_psd_estimate.m still carries
commented-out smooth(psd,61/51/45/39) lines, exp2.m a commented-out
exp(-200*x) weight function).

Consequence for a comparison table: in the 50-90 % band the port scores between
-0.51 and +1.34 dB relative to the authors' own Table 1 -- i.e. it may show the
method *stronger* than its source paper, never weaker.  Report our own protocol
runs, and state that the port reproduces the authors' psnrs.mat to within
0.2 dB at every density.

NAME TRAP: the paper's Table 1 calls mandril512.bmp "Baboon"; Baboon256.bmp is
a different image, used in exp3/exp4.  Do not conflate them.

Requires: numpy, scipy.  Tested with numpy 1.24.3 / scipy 1.10.0 / Python 3.11.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import maximum_filter, median_filter, minimum_filter
from scipy.spatial.distance import pdist, squareform

__all__ = [
    "ramf",
    "detect_mask",
    "expandimg",
    "denoise",
    "resblocks",
    "smblock",
    "run_gsr",
]

# ----------------------------------------------------------------------------
# Fixed parameters (exp1.m .. exp4.m)
# ----------------------------------------------------------------------------
BH = 8          # block height
BW = 8          # block width
BETA = 0.001
SIGMA1 = 0.1
SIGMA2 = 2500.0
RAMF_NMAX = 21  # maximum *radius* -> windows up to 43x43
EPS = float(np.finfo(np.float64).eps)   # 2.220446049250313e-16, MATLAB `eps`

# Sobel / Roberts-like 3x3 kernels of denoise.m.  Both `mn(:)` and `kir(:)` are
# column-major flattened before the dot product, so `sum(mn.*kir)` is a plain
# correlation -- no kernel flip.
_KIR1 = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
_KIR2 = _KIR1.T.copy()
_KIR3 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=np.float64)
_KIR4 = np.rot90(_KIR3).copy()          # [[0,-1,-2],[1,0,-1],[2,1,0]]
_KERNELS = np.stack([_KIR1, _KIR2, _KIR3, _KIR4], axis=0)   # (4,3,3)


# ----------------------------------------------------------------------------
# 1.  RAMF -- adaptive median filter (non-recursive, see P5)
# ----------------------------------------------------------------------------
def _ramf_pad(a: np.ndarray, nmax: int) -> np.ndarray:
    """Reproduce RAMF.m's (deliberately odd) border extension, literally.

    The MATLAB code indexes one past the array end, so ``imgn`` silently grows
    to ``(m+2*Nmax+1, n+2*Nmax+1)``; the extra row/column stays zero and is
    never read by the filter loop.  The top border is a *straight copy* of the
    first Nmax rows (not a mirror); the other three are shifted copies.  The
    four assignments must run in this order.
    """
    m, n = a.shape
    if m < nmax + 1 or n < nmax + 1:
        raise ValueError("RAMF needs an image of at least %dx%d" % (nmax + 1, nmax + 1))
    p = np.zeros((m + 2 * nmax + 1, n + 2 * nmax + 1), dtype=np.uint8)
    p[nmax:nmax + m, nmax:nmax + n] = a                                     # centre
    p[0:nmax, nmax:nmax + n] = a[0:nmax, :]                                 # top
    p[0:m + nmax, n + nmax:n + 2 * nmax + 1] = \
        p[0:m + nmax, n - 1:n + nmax].copy()                                # right
    p[m + nmax:m + 2 * nmax + 1, nmax:n + 2 * nmax + 1] = \
        p[m - 1:m + nmax, nmax:n + 2 * nmax + 1].copy()                     # bottom
    p[:, 0:nmax] = p[:, nmax:2 * nmax].copy()                               # left
    return p


def ramf(img: np.ndarray, max_window: int = RAMF_NMAX,
         vector_radius: int = 7, chunk: int = 20000) -> np.ndarray:
    """Adaptive median filter of RAMF.m.

    ``max_window`` is the maximum *radius* (21 in the paper -> 43x43 windows).

    For each pixel the radius grows from 1 until ``Imin < Imed < Imax``; the
    pixel keeps its own value if ``Imin < p < Imax``, otherwise it takes
    ``Imed``.  If no radius satisfies the condition the values of the largest
    radius are used.  All comparisons are strict and on uint8.

    ``vector_radius`` only controls the split between the vectorised sweep and
    the exact per-pixel sweep; the result does not depend on it.
    """
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    m, n = img.shape
    nmax = int(max_window)
    p = _ramf_pad(img, nmax)

    centre = p[nmax:nmax + m, nmax:nmax + n]
    imin = np.zeros((m, n), np.uint8)
    imax = np.zeros((m, n), np.uint8)
    imed = np.zeros((m, n), np.uint8)
    resolved = np.zeros((m, n), bool)

    # -- stage A: vectorised for small radii -------------------------------
    for r in range(1, min(vector_radius, nmax) + 1):
        todo = ~resolved
        if not todo.any():
            break
        k = 2 * r + 1
        sl = (slice(nmax, nmax + m), slice(nmax, nmax + n))
        mn = minimum_filter(p, size=k, mode="nearest")[sl]
        mx = maximum_filter(p, size=k, mode="nearest")[sl]
        md = median_filter(p, size=k, mode="nearest")[sl]
        imin[todo] = mn[todo]
        imax[todo] = mx[todo]
        imed[todo] = md[todo]
        resolved |= todo & (mn < md) & (md < mx)

    # -- stage B: exact per-pixel sweep for the rest ------------------------
    ys, xs = np.nonzero(~resolved)
    for r in range(min(vector_radius, nmax) + 1, nmax + 1):
        if ys.size == 0:
            break
        k = 2 * r + 1
        half = (k * k - 1) // 2          # MATLAB W(ceil(K^2/2)), K^2 odd
        keep = np.ones(ys.size, bool)
        off = np.arange(k)
        for s in range(0, ys.size, chunk):
            e = min(s + chunk, ys.size)
            rr = (ys[s:e] + nmax - r)[:, None] + off        # (c,k)
            cc = (xs[s:e] + nmax - r)[:, None] + off        # (c,k)
            win = p[rr[:, :, None], cc[:, None, :]].reshape(e - s, k * k)
            win.sort(axis=1)
            mn, md, mx = win[:, 0], win[:, half], win[:, -1]
            imin[ys[s:e], xs[s:e]] = mn
            imed[ys[s:e], xs[s:e]] = md
            imax[ys[s:e], xs[s:e]] = mx
            keep[s:e] = ~((mn < md) & (md < mx))
        ys, xs = ys[keep], xs[keep]

    return np.where((imin < centre) & (centre < imax), centre, imed).astype(np.uint8)


# ----------------------------------------------------------------------------
# 2.  Noise mask
# ----------------------------------------------------------------------------
def detect_mask(noisy: np.ndarray, ramf_img: np.ndarray) -> np.ndarray:
    """``mask = ~((img_RAMF ~= I) & (I==0 | I==255))``.

    Returns the *trusted* mask: ``True`` means the pixel is believed clean.
    """
    noisy = np.asarray(noisy)
    ramf_img = np.asarray(ramf_img)
    corrupted = (ramf_img != noisy) & ((noisy == 0) | (noisy == 255))
    return ~corrupted


# ----------------------------------------------------------------------------
# 3.  expandimg
# ----------------------------------------------------------------------------
def expandimg(x: np.ndarray, h: int, w: int) -> np.ndarray:
    """``expandimg.m``: mirror the border rows/columns (edge pixel duplicated).

    Identical to ``np.pad(x, ((h,h),(w,w)), mode='symmetric')``.
    """
    return np.pad(x, ((h, h), (w, w)), mode="symmetric")


# ----------------------------------------------------------------------------
# 4.  Constant part of the graph: W1
# ----------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _build_w1(sigma1: float = SIGMA1) -> np.ndarray:
    """W1 = squareform(exp(-pdist(fve(:,1:2)/sigma1))), the same for every block.

    ``fve`` row p (0-based) holds the coordinates ``(p//8 + 1, p%8 + 1)`` --
    row-major over the 8x8 grid (see P7).  W1 is invariant under the row/column
    swap, so this matches the index-consistent version elementwise.
    ``squareform`` gives an exactly zero diagonal (no self-loops).

    Cached per ``sigma1``.  The returned array is shared, so callers must not
    write to it.
    """
    idx = np.arange(BH * BW)
    pos = np.stack([idx // BW + 1, idx % BW + 1], axis=1).astype(np.float64)
    w1 = squareform(np.exp(-pdist(pos / sigma1)))
    w1.flags.writeable = False
    return w1


_W1 = _build_w1()      # the default-sigma1 matrix, for callers that want it


# ----------------------------------------------------------------------------
# 5.  denoise.m -- the block loop
# ----------------------------------------------------------------------------
def _fortran_flat(a: np.ndarray) -> np.ndarray:
    """Flatten the last two axes of ``a`` in MATLAB ``(:)`` (column-major) order."""
    return np.swapaxes(a, -1, -2).reshape(a.shape[:-2] + (a.shape[-1] * a.shape[-2],))


def _process_chunk(imgp: np.ndarray, maskp: np.ndarray,
                   rb0: np.ndarray, cb0: np.ndarray,
                   beta: float, sigma2: float,
                   degenerate_psd: str, w1: np.ndarray) -> np.ndarray:
    """Reconstruct a chunk of blocks.  Returns ``(B, 64)`` (column-major pixels).

    ``w1`` is the sigma1-dependent coordinate weight matrix, passed in rather
    than read from a module global so that a non-default sigma1 stays local to
    the call.
    """
    b = rb0.size
    a9 = np.arange(9)
    a8 = np.arange(8)
    a10 = np.arange(10)

    # -- bs: the 81 neighbourhood 8x8 sub-blocks of the 16x16 window --------
    rows = rb0[:, None, None] + a9[:, None] + a8[None, :]      # (B,9,8)
    cols = cb0[:, None, None] + a9[:, None] + a8[None, :]      # (B,9,8)
    sub = imgp[rows[:, :, None, :, None], cols[:, None, :, None, :]]   # (B,9,9,8,8)
    bs = _fortran_flat(sub.astype(np.float64)).reshape(b, 81, 64)
    del sub, rows, cols

    x = bs[:, 4 * 9 + 4, :]                    # centre sub-block (ii=jj=5 in MATLAB)

    # -- xmask: trusted flags of the centre block, column-major -------------
    r8 = rb0[:, None] + 4 + a8
    c8 = cb0[:, None] + 4 + a8
    xmask = _fortran_flat(maskp[r8[:, :, None], c8[:, None, :]]).astype(np.float64)

    # -- fve: 10x10 extended block -> coordinates + 4 gradient responses ----
    # NOTE (P7): row-major over the 8x8 grid, unlike x / xmask / bs.
    r10 = rb0[:, None] + 3 + a10
    c10 = cb0[:, None] + 3 + a10
    fbs = imgp[r10[:, :, None], c10[:, None, :]].astype(np.float64)      # (B,10,10)
    win3 = np.lib.stride_tricks.sliding_window_view(fbs, (3, 3), axis=(1, 2))
    grad = np.einsum("byxuv,kuv->byxk", win3, _KERNELS)                  # (B,8,8,4)
    feat = grad.reshape(b, 64, 4)
    del fbs, win3, grad

    # -- W2, W, L, eigendecomposition --------------------------------------
    d = feat[:, :, None, :] - feat[:, None, :, :]
    dist = np.sqrt(np.einsum("bpqk,bpqk->bpq", d, d)) / sigma2
    del d
    w_mat = w1[None, :, :] * np.exp(-dist)           # W1 has a zero diagonal
    del dist
    deg = w_mat.sum(axis=2)
    lap = np.zeros_like(w_mat)
    di = np.arange(64)
    lap[:, di, di] = deg
    lap -= w_mat
    evals, u = np.linalg.eigh(lap)                   # ascending, orthonormal
    lmax = evals[:, -1]
    ws = np.count_nonzero(w_mat, axis=(1, 2)).astype(np.float64)
    del w_mat, lap, evals

    # -- mean of the 81 blocks, covariance, PSD ----------------------------
    mbs = bs.mean(axis=1)                            # (B,64)
    xc = bs - mbs[:, None, :]
    cov = np.matmul(np.swapaxes(xc, 1, 2), xc) / (bs.shape[1] - 1.0)   # ddof = 1
    del bs, xc
    psd = np.einsum("bji,bji->bi", u, np.matmul(cov, u))
    del cov
    psum = psd.sum(axis=1)
    degenerate = ~np.isfinite(psum) | (psum <= 0.0)
    with np.errstate(invalid="ignore", divide="ignore"), \
            warnings.catch_warnings():
        # a fully degenerate block gives an all-NaN row; nanmax warns, MATLAB
        # does not, and the block is discarded downstream either way.
        warnings.simplefilter("ignore", RuntimeWarning)
        psd = psd / psum[:, None]
        wf = 1.0 / (psd ** 2 + EPS)
        # reconstructer.m's `max(w)` ignores NaN -> np.nanmax, not .max (D9a).
        wgt = lmax[:, None] * wf / np.nanmax(wf, axis=1, keepdims=True)
    del psd, wf

    # -- reconstructer.m ---------------------------------------------------
    kl = np.matmul(u, wgt[:, :, None] * np.swapaxes(u, 1, 2))
    del u, wgt
    lbs = xmask.sum(axis=1)
    empty = lbs == 0.0                               # no trusted pixel in the block
    safe_lbs = np.where(empty, 1.0, lbs)

    v = beta * kl / ws[:, None, None]
    v[:, di, di] += xmask / safe_lbs[:, None]
    rhs = (x - mbs) * xmask / safe_lbs[:, None]
    del kl

    bad = empty | degenerate | ~np.isfinite(v).all(axis=(1, 2)) | ~np.isfinite(rhs).all(axis=1)
    if degenerate_psd == "nan":
        bad = empty | ~np.isfinite(v).all(axis=(1, 2)) | ~np.isfinite(rhs).all(axis=1)

    f = np.zeros((b, 64), np.float64)
    good = ~bad
    if good.any():
        try:
            f[good] = np.linalg.solve(v[good], rhs[good][:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            for i in np.nonzero(good)[0]:
                f[i] = np.linalg.lstsq(v[i], rhs[i], rcond=None)[0]
        nf = ~np.isfinite(f).all(axis=1) & good
        for i in np.nonzero(nf)[0]:
            f[i] = np.linalg.lstsq(v[i], rhs[i], rcond=None)[0]
    if bad.any():
        # isempty(idx) early return of reconstructer.m: ff = s -> out = x.
        # Degenerate PSD (D4): same pass-through, or literal NaN if requested.
        f[bad] = x[bad] - mbs[bad]
        if degenerate_psd == "nan":
            nan_blocks = bad & ~empty
            if nan_blocks.any():
                f[nan_blocks] = np.nan

    return f + mbs


def denoise(img: np.ndarray, mask: np.ndarray,
            bh: int = BH, bw: int = BW, beta: float = BETA,
            sigma1: float = SIGMA1, sigma2: float = SIGMA2,
            chunk: int = 256, degenerate_psd: str = "passthrough") -> np.ndarray:
    """``denoise.m``.

    ``img`` / ``mask`` are the ``expandimg(.,4,4)``-padded AMF output and
    trusted mask, of shape ``(H+8, W+8)`` with ``H``, ``W`` multiples of 8.
    Returns the reassembled ``(H, W)`` float image (possibly non-finite).

    W1 depends only on ``sigma1`` and is cached per value by ``_build_w1``; the
    matrix is passed down explicitly, so a non-default ``sigma1`` cannot leak
    into later calls and the function is safe to call from several threads.
    """
    w1 = _build_w1(float(sigma1))
    if (bh, bw) != (BH, BW):
        raise NotImplementedError("the port is specialised to 8x8 blocks")

    imgp = np.asarray(img)
    maskp = np.asarray(mask).astype(bool)
    hp, wp = imgp.shape
    h, w = hp - 2 * (bh // 2), wp - 2 * (bw // 2)
    if h % bh or w % bw:
        raise ValueError("padded image must be a multiple of the block size")
    nrb, ncb = h // bh, w // bw

    # MATLAB accumulates in column-major block order (outer loop = columns).
    cb, rb = np.meshgrid(np.arange(ncb), np.arange(nrb), indexing="ij")
    rb0 = (rb.ravel() * bh).astype(np.intp)
    cb0 = (cb.ravel() * bw).astype(np.intp)

    out = np.empty((rb0.size, 64), np.float64)
    for s in range(0, rb0.size, chunk):
        e = min(s + chunk, rb0.size)
        out[s:e] = _process_chunk(imgp, maskp, rb0[s:e], cb0[s:e],
                                  beta, sigma2, degenerate_psd, w1)

    return resblocks(out.T, bh, bw, h, w)


# ----------------------------------------------------------------------------
# 6.  resblocks.m (size-general; identical output for square images)
# ----------------------------------------------------------------------------
def resblocks(yy: np.ndarray, bh: int, bw: int, h: int, w: int) -> np.ndarray:
    """Reassemble ``(64, nblocks)`` column-major blocks into an ``(h, w)`` image.

    ``resblocks.m`` swaps the row and column extents and is called with
    ``sqrt(#blocks)*8`` for both, so the original is square-only (D1).  The
    block ordering (column-major over the block grid) and the column-major
    ``reshape(.,bh,bw)`` are preserved exactly.
    """
    nrb, ncb = h // bh, w // bw
    s = np.empty((h, w), np.float64)
    blocks = yy.T.reshape(ncb, nrb, bw, bh)          # C-order (64,) -> (bw,bh)
    for cbi in range(ncb):
        for rbi in range(nrb):
            s[rbi * bh:(rbi + 1) * bh, cbi * bw:(cbi + 1) * bw] = blocks[cbi, rbi].T
    return s


# ----------------------------------------------------------------------------
# 7.  smblock.m -- deblocking by 1-D not-a-knot cubic splines
# ----------------------------------------------------------------------------
def _keep_vector(length: int, bsz: int) -> np.ndarray:
    """``xx`` of smblock.m: 0 at the first/last column of every interior block
    and at the inner edge of the first/last block; 1 elsewhere."""
    if length % bsz or length // bsz < 2:
        raise ValueError("smblock needs a multiple of the block size, >= 2 blocks")
    mid = np.tile(np.r_[0, np.ones(bsz - 2, int), 0], length // bsz - 2)
    return np.r_[np.ones(bsz - 1, int), 0, mid, 0, np.ones(bsz - 1, int)].astype(bool)


def _spline_line(vals: np.ndarray, keep: np.ndarray, trusted: np.ndarray) -> np.ndarray:
    """One smblock pass over a single row or column."""
    finite = np.isfinite(vals)
    # D3: drop non-finite samples from *both* index sets (the original only
    # filters `keep`, so a single non-finite trusted sample NaNs the whole line).
    sel = (keep | trusted) & finite
    xq = np.nonzero(sel)[0]
    if xq.size < 2:
        return np.where(finite, vals, 0.0)
    if xq.size < 4:                                  # not-a-knot needs >= 4 knots
        return np.interp(np.arange(vals.size), xq, vals[xq])
    cs = CubicSpline(xq, vals[xq], bc_type="not-a-knot", extrapolate=True)
    return cs(np.arange(vals.size))


def smblock(ii: np.ndarray, mask: np.ndarray, bh: int = BH, bw: int = BW) -> np.ndarray:
    """``smblock.m``: average of a horizontal and a vertical spline pass.

    A pixel keeps its reconstructed value in a pass iff it is a block interior
    or trusted; otherwise it is replaced by the not-a-knot cubic spline through
    the kept samples.  The two passes are averaged, so a pixel on only one of
    the two block boundaries gets a 50/50 blend.  Result is rounded
    half-away-from-zero (D2) and clipped to [0, 255].
    """
    ii = np.asarray(ii, dtype=np.float64)
    mask = np.asarray(mask).astype(bool)
    h, w = ii.shape
    keep_col = _keep_vector(w, bw)      # length w, used for the row pass
    keep_row = _keep_vector(h, bh)      # length h, used for the column pass (D1/A6)

    hs = np.empty((h, w), np.float64)
    for i in range(h):
        hs[i] = _spline_line(ii[i, :], keep_col, mask[i, :])
    vs = np.empty((h, w), np.float64)
    for j in range(w):
        vs[:, j] = _spline_line(ii[:, j], keep_row, mask[:, j])

    out = (hs + vs) / 2.0
    with np.errstate(invalid="ignore"):
        out = np.sign(out) * np.floor(np.abs(out) + 0.5)     # MATLAB round
        out = np.clip(out, 0.0, 255.0)                       # +-Inf -> 255 / 0
    return np.where(np.isfinite(out), out, 0.0)              # uint8(NaN) == 0


# ----------------------------------------------------------------------------
# 8.  Top level
# ----------------------------------------------------------------------------
def run_gsr(noisy_uint8: np.ndarray,
                  beta: float = BETA, sigma1: float = SIGMA1,
                  sigma2: float = SIGMA2, ramf_max: int = RAMF_NMAX,
                  chunk: int = 256, degenerate_psd: str = "passthrough",
                  return_ramf: bool = False):
    """Full pipeline of exp1.m/exp2.m for a single grayscale image.

    Works for arbitrary ``(H, W)`` with ``H, W >= 22``: RAMF and the mask are
    computed at the original size, then the AMF output and the mask are
    symmetric-padded on the bottom/right to a multiple of 8, the block pipeline
    runs on the padded image, and the result is cropped back (D1).
    """
    noisy = np.asarray(noisy_uint8)
    if noisy.ndim != 2:
        raise ValueError("run_gsr expects a 2-D grayscale image")
    if noisy.dtype != np.uint8:
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    h, w = noisy.shape

    img_ramf = ramf(noisy, ramf_max)
    mask = detect_mask(noisy, img_ramf)

    ph, pw = (-h) % BH, (-w) % BW
    if ph or pw:
        img_b = np.pad(img_ramf, ((0, ph), (0, pw)), mode="symmetric")
        mask_b = np.pad(mask, ((0, ph), (0, pw)), mode="symmetric")
    else:
        img_b, mask_b = img_ramf, mask

    emask = expandimg(mask_b, BH // 2, BW // 2)
    eimg = expandimg(img_b, BH // 2, BW // 2)
    psd_img = denoise(eimg, emask, BH, BW, beta, sigma1, sigma2,
                      chunk=chunk, degenerate_psd=degenerate_psd)
    psd_img = smblock(psd_img, mask_b, BH, BW)

    out = psd_img[:h, :w].astype(np.uint8)
    if return_ramf:
        return out, img_ramf
    return out
