# -*- coding: utf-8 -*-
"""FSIM and FSIMc of Zhang et al. (IEEE Trans. Image Process. 20(8), 2378-2386, 2011): a port of the authors'
FeatureSIM.m, with the functions phasecong2 and lowpassfilter of that file (P. Kovesi).

    https://web.comp.polyu.edu.hk/cslzhang/IQA/FSIM/Files/FeatureSIM.m
    18,817 bytes, SHA-256 f830fcbe1db957c37f31fc99f3cc48b98d606eed49f757d9a98890e240bd3d89

The port follows the file step by step. Four behaviours of MATLAB are kept on purpose, because a direct translation
to NumPy or SciPy changes them:
    conv2(a, k, 'same')  zeros outside the image, and the result starts at floor(size(k) / 2) of the full convolution
                         (scipy.signal.convolve2d with mode='same' starts at floor((size(k) - 1) / 2))
    round(x)             halves go away from zero (Python's round sends them to the even integer)
    real(x .^ lambda)    a negative x gives the real part of the principal power, not |x| ^ lambda
    max(a, b)            ignores NaN

The port follows the code where the code and the paper differ. phasecong2 uses the phase deviation and subtracts a noise
threshold, and Eq. (3) of the paper shows neither. On 84 pairs of method outputs, Eq. (3) changed FSIMc by up to 0.095.

This file is not covered by the MIT license of the repository. The terms of the original code are in fsim_NOTICE.txt.
"""
# %% Imports
import math
from functools import lru_cache

import numpy as np
from scipy.signal import convolve2d

# %% Constants of FeatureSIM.m
NSCALE = 4
NORIENT = 4
MIN_WAVELENGTH = 6
MULT = 2
SIGMA_ONF = 0.55
D_THETA_ON_SIGMA = 1.2
K = 2.0
EPSILON = 0.0001
LOWPASS_CUTOFF = 0.45
LOWPASS_ORDER = 15
NOISE_RESCALE = 1.7
T1, T2, T3, T4 = 0.85, 160.0, 200.0, 200.0
LAMBDA = 0.03
DX = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]], np.float64) / 16
DY = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], np.float64) / 16


# %% MATLAB semantics
def conv2_same(a, kernel):
    """conv2(a, kernel, 'same')."""
    full = convolve2d(a, kernel, mode="full", boundary="fill", fillvalue=0)
    r, c = kernel.shape[0] // 2, kernel.shape[1] // 2
    return full[r:r + a.shape[0], c:c + a.shape[1]]


def downsampling_factor(rows, cols):
    """F = max(1, round(min(rows, cols) / 256))."""
    return max(1, math.floor(min(rows, cols) / 256 + 0.5))


def centered_range(n):
    """xrange and yrange of phasecong2 and lowpassfilter, for odd and even sizes."""
    if n % 2:
        return (np.arange(n) - (n - 1) / 2) / (n - 1)
    return (np.arange(n) - n / 2) / n


def chroma_term(product):
    """real(product .^ lambda). A negative product has the principal power |product| ^ lambda * exp(i * pi * lambda)."""
    power = np.abs(product) ** LAMBDA
    return np.where(product < 0, power * math.cos(math.pi * LAMBDA), power)


# %% Phase congruency
@lru_cache(maxsize=8)
def filter_bank(rows, cols):
    """For each orientation: the filters of the scales, EM_n, sumEstSumAn2 and sumEstSumAiAj."""
    x, y = np.meshgrid(centered_range(cols), centered_range(rows))
    radius = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(-y, x)
    lp = np.fft.ifftshift(1.0 / (1.0 + (radius / LOWPASS_CUTOFF) ** (2 * LOWPASS_ORDER)))

    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)
    radius[0, 0] = 1
    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    log_gabor = []
    for s in range(NSCALE):
        fo = 1.0 / (MIN_WAVELENGTH * MULT ** s)
        g = np.exp(-(np.log(radius / fo) ** 2) / (2 * math.log(SIGMA_ONF) ** 2)) * lp
        g[0, 0] = 0
        log_gabor.append(g)

    theta_sigma = math.pi / NORIENT / D_THETA_ON_SIGMA
    bank = []
    for o in range(NORIENT):
        angl = o * math.pi / NORIENT
        ds = sintheta * math.cos(angl) - costheta * math.sin(angl)
        dc = costheta * math.cos(angl) + sintheta * math.sin(angl)
        dtheta = np.abs(np.arctan2(ds, dc))
        spread = np.exp(-(dtheta ** 2) / (2 * theta_sigma ** 2))

        filters = [g * spread for g in log_gabor]
        ifft_filters = [np.real(np.fft.ifft2(f)) * math.sqrt(rows * cols) for f in filters]
        em_n = float(np.sum(filters[0] ** 2))
        sum_an2 = float(sum(np.sum(f ** 2) for f in ifft_filters))
        sum_ai_aj = float(sum(np.sum(ifft_filters[si] * ifft_filters[sj])
                              for si in range(NSCALE - 1) for sj in range(si + 1, NSCALE)))
        bank.append((filters, em_n, sum_an2, sum_ai_aj))
    return bank


def phasecong2(im):
    """Phase congruency map of a grayscale image of type float64."""
    rows, cols = im.shape
    imagefft = np.fft.fft2(im)
    energy_all = np.zeros((rows, cols))
    an_all = np.zeros((rows, cols))
    for filters, em_n, sum_an2, sum_ai_aj in filter_bank(rows, cols):
        sum_e = np.zeros((rows, cols))
        sum_o = np.zeros((rows, cols))
        sum_an = np.zeros((rows, cols))
        eo = []
        for f in filters:
            response = np.fft.ifft2(imagefft * f)
            eo.append(response)
            sum_an += np.abs(response)
            sum_e += response.real
            sum_o += response.imag

        x_energy = np.sqrt(sum_e ** 2 + sum_o ** 2) + EPSILON
        mean_e = sum_e / x_energy
        mean_o = sum_o / x_energy
        energy = np.zeros((rows, cols))
        for response in eo:
            e, o = response.real, response.imag
            energy += e * mean_e + o * mean_o - np.abs(e * mean_o - o * mean_e)

        # Noise threshold, estimated from the responses at the smallest scale
        median_e2n = float(np.median(np.abs(eo[0]) ** 2))
        mean_e2n = -median_e2n / math.log(0.5)
        noise_power = mean_e2n / em_n
        est_noise_energy2 = 2 * noise_power * sum_an2 + 4 * noise_power * sum_ai_aj
        tau = math.sqrt(est_noise_energy2 / 2)
        est_noise_energy = tau * math.sqrt(math.pi / 2)
        est_noise_energy_sigma = math.sqrt((2 - math.pi / 2) * tau ** 2)
        threshold = (est_noise_energy + K * est_noise_energy_sigma) / NOISE_RESCALE

        energy_all += np.fmax(energy - threshold, 0)
        an_all += sum_an
    return energy_all / an_all


# %% The index
def features(rgb_u8):
    """The maps of one image with channels in the order R, G, B: downsampled I and Q, phase congruency, gradient."""
    rgb = rgb_u8.astype(np.float64)
    rows, cols = rgb.shape[:2]
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    i = 0.596 * r - 0.274 * g - 0.322 * b
    q = 0.211 * r - 0.523 * g + 0.312 * b

    f = downsampling_factor(rows, cols)
    average = np.ones((f, f)) / (f * f)
    i = conv2_same(i, average)[::f, ::f]
    q = conv2_same(q, average)[::f, ::f]
    y = conv2_same(y, average)[::f, ::f]

    gradient = np.sqrt(conv2_same(y, DX) ** 2 + conv2_same(y, DY) ** 2)
    return dict(i=i, q=q, pc=phasecong2(y), gradient=gradient)


def similarity(ref, dis):
    """(FSIM, FSIMc) from two results of features."""
    pc_sim = (2 * ref["pc"] * dis["pc"] + T1) / (ref["pc"] ** 2 + dis["pc"] ** 2 + T1)
    gradient_sim = ((2 * ref["gradient"] * dis["gradient"] + T2)
                    / (ref["gradient"] ** 2 + dis["gradient"] ** 2 + T2))
    pc_m = np.fmax(ref["pc"], dis["pc"])
    i_sim = (2 * ref["i"] * dis["i"] + T3) / (ref["i"] ** 2 + dis["i"] ** 2 + T3)
    q_sim = (2 * ref["q"] * dis["q"] + T4) / (ref["q"] ** 2 + dis["q"] ** 2 + T4)
    denominator = np.sum(pc_m)
    fsim_value = np.sum(gradient_sim * pc_sim * pc_m) / denominator
    fsimc_value = np.sum(gradient_sim * pc_sim * chroma_term(i_sim * q_sim) * pc_m) / denominator
    return float(fsim_value), float(fsimc_value)


def feature_sim(ref_rgb_u8, dis_rgb_u8):
    """[FSIM, FSIMc] = FeatureSIM(imageRef, imageDis) for 8-bit images with channels in the order R, G, B."""
    return similarity(features(ref_rgb_u8), features(dis_rgb_u8))
