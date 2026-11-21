# -*- coding: utf-8 -*-
"""Output conversion and the evaluation metrics of the paper.

The images of the experiments are in the channel order of OpenCV (B, G, R). CPSNR and SSIM do not depend on that
order, but NCD and FSIMc do, so ncd and fsimc reverse the channels before they compute anything.
"""
import numpy as np
from skimage.color import rgb2lab
from skimage.metrics import structural_similarity

from . import fsim as _fsim


def to_uint8(x):
    """Round to the nearest integer (ties to even), clip to [0, 255], convert to uint8."""
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def cpsnr(a_u8, b_u8):
    """Color PSNR: one mean squared error over all components of the two 8-bit images."""
    mse = np.mean((a_u8.astype(np.float64) - b_u8.astype(np.float64)) ** 2)
    if mse <= 0:
        return float("inf")
    return 10.0 * np.log10(255.0 ** 2 / mse)


def ssim(a_u8, b_u8):
    """Mean of the three per-channel SSIM values with data_range = 255 (scikit-image defaults)."""
    return float(np.mean([structural_similarity(a_u8[:, :, c], b_u8[:, :, c], data_range=255)
                          for c in range(3)]))


def lab_of(bgr_u8):
    """CIELAB values of an 8-bit B, G, R image, taken as sRGB under D65 (scikit-image rgb2lab)."""
    return rgb2lab(bgr_u8[:, :, ::-1])


def ncd(ref_bgr_u8, out_bgr_u8, ref_lab=None):
    """Normalized color difference: the sum over the pixels of the Euclidean CIELAB distance between the reference
    and the output, divided by the sum over the pixels of the Euclidean norm of the reference. Lower is better."""
    lab = lab_of(ref_bgr_u8) if ref_lab is None else ref_lab
    difference = np.sqrt(np.sum((lab - lab_of(out_bgr_u8)) ** 2, axis=2))
    return float(np.sum(difference) / np.sum(np.sqrt(np.sum(lab ** 2, axis=2))))


def fsim_features(bgr_u8):
    """The maps of crrf.fsim for an 8-bit B, G, R image. Pass the result to fsimc to reuse it for one reference."""
    return _fsim.features(bgr_u8[:, :, ::-1])


def fsimc(ref_bgr_u8, out_bgr_u8, ref_features=None):
    """FSIMc of crrf/fsim.py, a port of the authors' FeatureSIM.m. Higher is better."""
    ref = fsim_features(ref_bgr_u8) if ref_features is None else ref_features
    return _fsim.similarity(ref, fsim_features(out_bgr_u8))[1]
