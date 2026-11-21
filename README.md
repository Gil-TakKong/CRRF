# CRRF: Color Ratio Refinement Filter

Python implementation of the Color Ratio Refinement Filter (CRRF) for removing channel-independent salt-and-pepper noise (SPN) from color images, by Gil-Tak Kong and Un-Sook Choi. The repository also contains the implementations of the nine methods compared with CRRF, the images used in the experiments, and unit tests.

## Summary

Under channel-independent SPN, each channel of a color image is corrupted independently, so that many corrupted pixels keep uncorrupted components in their other channels. CRRF is a non-learning filter that restores only the corrupted components of a pixel from its clean components. It needs no training and no noise density.

- Identification: every component of value 0 or 255 is treated as corrupted; the other components are clean.
- Phase 1: each channel is restored separately. In each iteration, every corrupted component with at least one clean or previously restored von Neumann neighbor takes the mean of those neighbors, and all components are updated synchronously.
- Phase 2: the color direction of each pixel is estimated as the component-wise median of the normalized RGB vectors in a 7x7 window of the Phase 1 output. A pixel with one or two corrupted components restores them from this direction and its clean components. A pixel with three corrupted components keeps the magnitude of its Phase 1 vector and takes the median direction.
- Clean components never change.

## Repository structure

```
README.md                    this file
LICENSE                      MIT license (see License)
requirements.txt             pinned package versions used for the paper
.gitignore                   run-time files that are not part of the repository
demo.py                      noise, restoration and scores on one image, with command-line arguments
crrf/                        the library
  __init__.py                the public API; sets the default Numba cache folder
  core.py                    identification, Phase 1, crrf, phase1_only
  phase2.py                  Phase 2 with the sorting networks of median_network.py
  median_network.py          generator and generated kernels of the median of 49 values
  variants.py                window-by-window Phase 2 kernel with the settings of the analysis
                             (window sizes, channel-wise median and others)
  geometry.py                reflect_101 boundary, window order, insertion-sort median
  noise.py                   noise model and seed rule
  metrics.py                 8-bit conversion, CPSNR, FSIMc, NCD (and SSIM)
  fsim.py, fsim_NOTICE.txt   port of the authors' FeatureSIM.m, and its terms
  baselines/                 the nine compared methods
    __init__.py              registry with the settings of the paper: BASELINES, run_baseline
    filters.py               DBA, NAFSMF, DAMF
    helpers.py               Numba primitives of filters.py
    ca_ngmf.py               CA-NGMF
    gsr.py                   GSR, ported from the authors' released MATLAB code
    armf.py                  ARmF
    newmmf.py                NEWMMF, the code of its paper with local imports
    vmf.py                   VMF
    fapgf.py                 FAPGF
tests/                       unit tests; they need no image
  references.py              slow pure Python versions of VMF, FAPGF and the Phase 2 kernel
data/
  README.md                  sources of the images and the role of their file names
  manifest.csv               file name, size and SHA-256 hash of the 71 images
  images/                    the 68 CBSD68 images and Baboon, Barbara, Peppers
```

## Installation

Python 3.11 (tested with 3.11.14). Create an environment (venv or conda), then install the pinned versions used for the paper:

```
pip install -r requirements.txt
```

The Numba kernels compile on the first call and are cached. `crrf/__init__.py` sets the default cache folder to `numba_cache_crrf` under the system temp folder, because Numba's `cache=True` fails on long or non-ASCII paths. Set the environment variable `NUMBA_CACHE_DIR` before the first import to use another folder.

For the timing conditions of the paper, set `NUMBA_NUM_THREADS=1` and the BLAS thread variables to one thread before the first import. All Numba kernels of this repository are single-threaded in any case.

Reproducibility of GSR: the results of the paper were computed with the Anaconda builds of NumPy 1.24.3 and SciPy 1.10.0, which use Intel MKL. The wheels installed by `pip` use OpenBLAS. CRRF, Phase 1 and the other eight methods call no BLAS or LAPACK routine, so their outputs do not depend on this. GSR solves ill-conditioned eigenproblems and linear systems, so its outputs can move by one gray level at some pixels under another BLAS, and its scores can then differ in the last digits.

## Quick start

Run from the repository root:

```python
import cv2
from crrf import cpsnr, crrf, fsimc, make_noisy, ncd, phase1_only, to_uint8

clean = cv2.imread("data/images/cbsd68/101085.jpg")          # uint8, B, G, R
noisy, seed = make_noisy(clean, r=0, i=0, iota=8)            # run 0, first CBSD68 image, 90 %
noisy = to_uint8(noisy)
for name, out in (("Phase 1", phase1_only(noisy)), ("CRRF", crrf(noisy))):
    print(f"{name:8s} CPSNR {cpsnr(clean, out):.2f} dB, FSIMc {fsimc(clean, out):.4f}, NCD {ncd(clean, out):.4f}")
```

Output (the same values as this run in the experiments of the paper):

```
Phase 1  CPSNR 19.34 dB, FSIMc 0.7488, NCD 0.3886
CRRF     CPSNR 21.32 dB, FSIMc 0.8055, NCD 0.2240
```

For an image of your own, `crrf.add_noise(image, density, rng)` adds the noise with a generator of your choice. The compared methods are called through `crrf.baselines.run_baseline(name, noisy_uint8)`, with `name` in `crrf.baselines.BASELINES`.

The demo uses a synthetic image if no image is given:

```
python demo.py
python demo.py --image data/images/Barbara.jpg --density 0.7 --seed 1 --out restored.png
python demo.py --method DBA
```

```
image synthetic (160 x 128), density 0.5, seed 0
noisy     CPSNR   8.54 dB   FSIMc 0.1305   NCD 0.9693
CRRF      CPSNR  38.21 dB   FSIMc 0.9888   NCD 0.0050   (0.302 s, the first call includes compilation)
```

The unit tests need no image (81 tests, about 20 s once the kernels are compiled):

```
python -m unittest discover -s tests
```

## Noise model and seeds

In each channel, exactly `k = round(d H W)` positions, rounded half to even, are drawn without replacement, and each drawn component is set to 0 or 255 with probability 0.5. The three channels are drawn independently, in the array order B, G, R:

```python
idx = rng.choice(h * w, size=k, replace=False)
val = np.where(rng.random(k) < 0.5, 0.0, 255.0)
```

The generator is `numpy.random.default_rng(seed)` of NumPy 1.24.3, with

```
seed = 42 + 100000 r + 100 i + iota
```

where `r = 0..19` is the run, `i` is the 0-based position of the image in the lexicographic order of the file names of its dataset (for the standard images, Baboon 0, Barbara 1, Peppers 2), and `iota = 0..8` is the position of the density in the list 10 %, 20 %, ..., 90 %. The paper reports the densities 10, 30, 50, 70 and 90 % (`iota` = 0, 2, 4, 6, 8), and some figures use 80 % (`iota` = 7). All methods process the same noisy image.

Every output is converted by `np.clip(np.rint(x), 0, 255).astype(np.uint8)` before it is scored. NumPy rounds exact halves to the nearest even integer (2.5 -> 2, 3.5 -> 4); a reimplementation must do the same to reproduce the scores exactly.

Timing: every method runs once on every noisy image of run 0, on a single thread and after a warm-up call that compiles the Numba code. The timer is `time.perf_counter`. The time excludes noise generation, the conversion to 8 bits and scoring. The time of Phase 1 includes the identification, and the time of CRRF includes Phase 1.

## Evaluation metrics

- CPSNR = 10 log10(255^2 / MSE), with one MSE over all pixels and the three channels, in float64. Higher is better.
- FSIMc: `crrf/fsim.py`, a port of the FeatureSIM.m of Zhang et al. (2011), on the image in R, G, B order. Higher is better.
- NCD: the sum over the pixels of the Euclidean CIELAB distance between the reference and the output, divided by the sum over the pixels of the Euclidean norm of the reference, with CIELAB from `skimage.color.rgb2lab` (sRGB, D65). Lower is better.

The images are read by OpenCV in B, G, R order; `fsimc` and `ncd` reverse the channels before computing. A value for a set of runs is the mean over those runs, and a difference is taken run by run on the same noisy image.

## Methods and parameter settings

### CRRF

`K` is the number of corrupted components of pixel `p`, `u(p)` its Phase 1 vector, and `x(p)` its output.

| Item | Setting |
|---|---|
| Identification | a component is corrupted if its value is 0 or 255; the noise density is not needed |
| Phase 1 | per channel; the mean of the clean and previously restored components among the von Neumann neighbors N, W, E, S; at least one such neighbor |
| Update | synchronous: every read uses the state of the previous iteration |
| Termination of Phase 1 | the first iteration that restores nothing; upper limit of 500 iterations, not reached on any image of the paper |
| Boundary | reflect_101 (-1 -> 1, n -> n - 2) in both phases |
| Color direction | c(q) = u(q) / \|\|u(q)\|\|, the normalized RGB vector of the Phase 1 output |
| Window | 7x7, 49 pixels including the center |
| Median direction | m = component-wise median of c(q) (rank 25 of 49); c_hat = m / \|\|m\|\| |
| K = 1, 2 | x_j = mean over the clean channels a of p with c_hat_a > 0 of u_a(p) c_hat_j / c_hat_a; without such a channel the Phase 1 value stays |
| K = 3 | x_j = \|\|u(p)\|\| c_hat_j |
| Safeguards | a pixel with u(q) = 0 is left out of the window; a pixel with u(p) = 0 or m = 0 keeps its Phase 1 values; none of them acts once Phase 1 has restored every component of value 0 or 255 |
| Clean components | never change |
| Output | Phase 2 reads the Phase 1 output once and does not clip; the final conversion rounds and clips |

The median of 49 values is computed by the min/max sorting networks of `crrf/median_network.py`, with the work shared between neighboring windows. A min/max network only moves values, so the output is the same float64 value that sorting would leave at rank 25. The tests check each level by the 0-1 principle and compare the complete path with `numpy.median` and, bit for bit, with the window-by-window kernel of `crrf/variants.py`.

### Compared methods

All nine methods receive the same noisy image and nothing else. DBA, NAFSMF, DAMF, CA-NGMF, GSR, ARmF and NEWMMF are applied to each channel separately, and each channel receives its own 8-bit noisy channel. VMF and FAPGF take the three channels as one vector. All methods except GSR run as Numba kernels; GSR runs on NumPy and SciPy. CA-NGMF and NEWMMF are our earlier filters; `crrf/baselines/newmmf.py` is the code of its paper with the imports made local, and its output equals that of the original code. Every other method was implemented from the original paper and, where available, the authors' code. Parameters follow the papers or the released code. Only GSR, VMF and FAPGF can change components that are not 0 or 255.

| Method | Reference | Parameters and implementation details |
|---|---|---|
| DBA | K. S. Srinivasan, D. Ebenezer, IEEE Signal Processing Letters 14(3) (2007) 189-192 | only 0/255 components are processed; 3x3 window including corrupted values (in-image entries); the median if it is not 0/255, otherwise the already processed left neighbor (the upper neighbor in the first column); recursive; no tunable parameters |
| NAFSMF | K. K. V. Toh, N. A. Mat Isa, IEEE Signal Processing Letters 17(3) (2010) 281-284 | 0/255 identification in place of the histogram search; window 3, 5, 7 (max 7) until at least 4 noise-free values; M = median of the first (up to four) noise-free values found; D = max \|v - X\| over the noise-free values v, X = center value; T1 = 10, T2 = 30; output (1 - f) X + f M; recursive |
| DAMF | U. Erkan, L. Gokrem, S. Enginoglu, Computers & Electrical Engineering 70 (2018) 789-798 | window 3, 5, ..., 21 until at least 1 noise-free value; median of the noise-free values; recursive. The largest window of 21x21 was set by the authors of this repository |
| CA-NGMF | G.-T. Kong, U.-S. Choi, Multimedia Tools and Applications 85(3) (2026) Article 247 | max_iterations = 1000; uint8 input; groups G1 (N, W, C, E), G2 (W, N, C, S), G3 (W, C, E, S), G4 (S, C, N, E) in fixed order; the first group with at least 2 valid (not 0/255) values gives int(mean); out-of-image neighbors ignored; synchronous; stops when no 0/255 pixel remains, at 1000 passes, or when the 0/255 count is unchanged for 10 passes |
| GSR | Q. Zhang, C. Huang, L. Yang, Z. Yang, Digital Signal Processing 135 (2023) 103941 | RAMF with maximum radius 21; the corruption mask is the set of pixels whose value is 0 or 255 and which the RAMF output changes; the RAMF image and the mask are expanded by 4 pixels on each side; 8x8 blocks, beta = 0.001, sigma1 = 0.1, sigma2 = 2500, w(x) = 1 / (x^2 + eps); the reconstructed blocks are recombined and smoothed block by block with a cubic spline; uint8 in, uint8 out. The constants are those of the authors' exp1.m-exp4.m |
| ARmF | S. Enginoglu, U. Erkan, S. Memis, Multimedia Tools and Applications 78 (2019) 35401-35418 | five passes with p = 5, 4, 3, 2, 1; for each 0/255 component in the snapshot taken at the start of the pass, the smallest (2k+1)x(2k+1) window (k <= p) with at least one clean pixel gives a Riesz mean with weights (1 / (1 + \|ds\| + \|dt\|))^2; symmetric boundary as in MATLAB padarray; the 8-bit conversion follows MATLAB uint8(). Python port of the authors' ARmF.m |
| NEWMMF | G.-T. Kong, U.-S. Choi, Computers and Electrical Engineering 141 (2027) 111644 | noise mask = components of value 0 or 255; Phase 1: von Neumann neighborhoods of radius 1, 2, 3 tried in this order, the first radius with at least T1 = 2, T2 = 2, T3 = 3 clean or restored neighbors gives their mean weighted by 1 / Manhattan distance; synchronous; at most 300 iterations, and it stops early when no 0/255 component remains or nothing is restored; Phase 2: one synchronous pass replacing every component of the mask by the median of its four von Neumann neighbors; half-sample symmetric boundary |
| VMF | J. Astola, P. Haavisto, Y. Neuvo, Proceedings of the IEEE 78(4) (1990) 678-689 | 3x3 window of 9 pixels; L2 distance; every pixel is replaced by the pixel vector of the window with the smallest sum of distances (no identification step); ties go to the first pixel in raster order; reflect_101 boundary |
| FAPGF | L. Malinski, B. Smolka, Journal of Real-Time Image Processing 11(3) (2016) 427-444 | d = 0.1, gamma = 0.8; a pixel with fewer than two of its eight neighbors within the normalized RGB distance d is replaced by the average of its window weighted by m^gamma, where m is the number of close neighbors of each pixel; the window grows to 5x5, 7x7, ... if no pixel of the 3x3 window has a close neighbor. The window is cut at the image border (`border="crop"`), which the paper leaves open. On the image PAR of its Table 3 (10 runs per density), this rule gave 37.08, 34.50 and 32.36 dB at 10, 20 and 30 % against the published 37.13, 34.75 and 32.40 dB, closer than the reflect_101 boundary |

The tests compare the Numba kernels of VMF and FAPGF with pure Python versions bit for bit, and check FAPGF on the hand-worked cases of `tests/test_baselines.py`.

#### GSR: port of the authors' MATLAB code

`crrf/baselines/gsr.py` is a line-by-line port of the MATLAB code released by the authors of the GSR paper (GitHub branches `functions` and `experiments`), not an implementation of the equations printed in the paper. Where the two disagree the port follows the code, because the code is what produced the published numbers. The module docstring lists every disagreement (labels P1 to P8) and every deviation of the port from the code (labels D1 to D9). Those that change the output are:

- The paper's edge weight is `exp(-||.||_2^2 / sigma^2)` with `sigma1^2 = 0.1` and `sigma2^2 = 2500`; the code divides the feature rows by 0.1 and 2500 and then applies MATLAB `pdist`, which is the un-squared Euclidean distance.
- The paper's weight function is `1 / x^2`; the code uses `1 / (x^2 + eps)`.
- The paper solves `(Ik' Ik + beta L) f = Ik' y_k`; the code solves `(Ek / lbs + beta KL / ws) f = (Ek .* s) / lbs`, which rescales beta by `lbs / ws`, so the effective beta is about `2.5e-7` to `1.6e-5` rather than 0.001.
- The 64 graph-node feature rows are built in row-major order while the signal, the mask and the covariance are column-major (MATLAB `(:)`), so node p carries the structural features of the transposed pixel location.

Two deviations from the released code apply to the experiments of this paper:

- The released code is defined only for square images whose side is a multiple of 8, and CBSD68 images are 481x321 or 321x481. The port computes the mask at the original size, pads the filtered image and the mask symmetrically on the bottom and right to a multiple of 8, runs the block pipeline, and crops the result back.
- The released code produces `NaN` when a 16x16 window of the adaptive-median output is exactly constant, and its block-smoothing step then spreads those values along whole rows and columns. The port keeps such a block unchanged.

Both choices make the GSR baseline stronger than a literal execution of the released code would be. The numbers reported for GSR are the numbers of this port, obtained under the same protocol as every other method.

Origin and terms of the original code, from the authors' repository: "Copyright (C) 2022 Qian Zhang and Zhihua Yang. All rights reserved. Notice: This repository is free software for scientific research, you can redistribute it and/or modify it."

## Data

The 71 images of the paper are in `data/images/`: the 68 CBSD68 images (481x321 or 321x481) and the standard images Baboon, Barbara and Peppers (512x512). See [data/README.md](data/README.md) for their sources and for why the file names must stay unchanged.

## Citation

Citation details will be added after publication.

## License

The code is released under the MIT license (see LICENSE), with these exceptions:

- `crrf/baselines/gsr.py` is derived from the MATLAB code of Zhang and Yang (see the GSR note above), which its authors release for scientific research; it is redistributed here on those terms and is not covered by the MIT license of this repository.
- `crrf/baselines/armf.py` is a Python port of the authors' ARmF.m, as noted in its docstring.
- `crrf/fsim.py` is a port of the FeatureSIM.m of Zhang et al., which includes code of P. Kovesi; it is redistributed on the terms in `crrf/fsim_NOTICE.txt`, which allow educational and research use and restrict commercial use, and it is not covered by the MIT license of this repository.
- The images in `data/images/` are third-party material, included so that the experiments can be repeated. The terms of use of their original sources apply, and the MIT license of this repository does not cover them.
