# -*- coding: utf-8 -*-
"""The noise generator, the seed rule, the output conversion and the metrics."""
import hashlib
import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crrf                             # noqa: E402
from crrf.noise import density_index    # noqa: E402


class Noise(unittest.TestCase):
    def test_exact_count_per_channel(self):
        for h, w, d in [(32, 48, 0.1), (32, 48, 0.5), (32, 48, 0.9), (321, 481, 0.3), (512, 512, 0.7)]:
            mask, _ = crrf.draw_noise(h, w, d, np.random.default_rng(1))
            expected = int(round(d * h * w))
            self.assertEqual(mask.sum(axis=(0, 1)).tolist(), [expected] * 3)

    def test_round_half_to_even(self):
        for d, expected in [(0.25, 2), (0.75, 8)]:              # 2.5 -> 2 and 7.5 -> 8 for 10 pixels
            mask, _ = crrf.draw_noise(2, 5, d, np.random.default_rng(1))
            self.assertEqual(mask.sum(axis=(0, 1)).tolist(), [expected] * 3)

    def test_values_and_untouched_components(self):
        clean = np.full((40, 40, 3), 100, np.uint8)
        noisy = crrf.add_noise(clean, 0.5, np.random.default_rng(2))
        self.assertEqual(noisy.dtype, np.float64)
        self.assertEqual(set(np.unique(noisy).tolist()), {0.0, 100.0, 255.0})
        self.assertEqual((noisy != 100.0).sum(axis=(0, 1)).tolist(), [800] * 3)

    def test_both_values_are_about_equally_likely(self):
        noisy = crrf.add_noise(np.full((200, 200, 3), 100, np.uint8), 0.9, np.random.default_rng(3))
        share = (noisy == 0.0).sum() / (noisy != 100.0).sum()
        self.assertLess(abs(share - 0.5), 0.01)

    def test_channels_are_drawn_independently(self):
        mask, _ = crrf.draw_noise(100, 100, 0.5, np.random.default_rng(4))
        self.assertFalse(np.array_equal(mask[:, :, 0], mask[:, :, 1]))
        self.assertFalse(np.array_equal(mask[:, :, 1], mask[:, :, 2]))
        # the share of pixels with all three components chosen is close to 0.5 ** 3
        self.assertLess(abs(mask.all(axis=2).mean() - 0.125), 0.02)

    def test_determinism(self):
        clean = np.random.default_rng(0).integers(1, 255, (16, 21, 3)).astype(np.uint8)
        a = crrf.add_noise(clean, 0.3, np.random.default_rng(7))
        b = crrf.add_noise(clean, 0.3, np.random.default_rng(7))
        c = crrf.add_noise(clean, 0.3, np.random.default_rng(8))
        self.assertEqual(a.tobytes(), b.tobytes())
        self.assertNotEqual(a.tobytes(), c.tobytes())

    def test_seed_rule(self):
        self.assertEqual(crrf.seed_of(0, 0, 0), 42)
        self.assertEqual(crrf.seed_of(19, 67, 8), 42 + 100000 * 19 + 100 * 67 + 8)
        self.assertEqual(crrf.DENSITIES, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self.assertEqual(density_index(0.8), 7)
        with self.assertRaises(ValueError):
            density_index(0.85)

    def test_make_noisy_follows_the_seed_rule(self):
        clean = np.random.default_rng(0).integers(1, 255, (16, 21, 3)).astype(np.uint8)
        noisy, seed = crrf.make_noisy(clean, 3, 5, 4)
        self.assertEqual(seed, 42 + 300000 + 500 + 4)
        self.assertEqual(noisy.tobytes(), crrf.add_noise(clean, 0.5, np.random.default_rng(seed)).tobytes())

    @unittest.skipUnless(np.__version__ == "1.24.3", "the stream is pinned for the NumPy version of the paper")
    def test_stream_of_the_paper(self):
        # The hash holds under NumPy 1.24.3. A change means that the noisy images differ from the paper's.
        noisy, _ = crrf.make_noisy(np.full((16, 21, 3), 100, np.uint8), 0, 0, 4)
        self.assertEqual(hashlib.sha256(noisy.tobytes()).hexdigest(), STREAM_SHA256)


# Pinned value of Noise.test_stream_of_the_paper
STREAM_SHA256 = "d13dff8a9e1a55b6908f030cf6ca3054f233e8d9c1285953d7c22376428541bc"


class Metrics(unittest.TestCase):
    def test_to_uint8_rounds_ties_to_even_and_clips(self):
        x = np.array([2.5, 3.5, 0.5, 1.5, 254.5, -3.0, 300.0, 255.4])
        self.assertEqual(crrf.to_uint8(x).tolist(), [2, 4, 0, 2, 254, 0, 255, 255])
        self.assertEqual(crrf.to_uint8(x).dtype, np.uint8)

    def test_cpsnr_uses_one_mse_over_all_components(self):
        a = np.zeros((4, 4, 3), np.uint8)
        b = a.copy()
        b[:, :, 0] = 15                     # MSE = 15^2 / 3 = 75, over the three channels together
        self.assertAlmostEqual(crrf.cpsnr(a, b), 10.0 * math.log10(255.0 ** 2 / 75.0), places=12)
        self.assertEqual(crrf.cpsnr(a, a), float("inf"))

    def test_ssim_is_the_mean_over_the_channels(self):
        from skimage.metrics import structural_similarity
        rng = np.random.default_rng(1)
        a = rng.integers(0, 256, (32, 40, 3)).astype(np.uint8)
        b = rng.integers(0, 256, (32, 40, 3)).astype(np.uint8)
        each = [structural_similarity(a[:, :, c], b[:, :, c], data_range=255) for c in range(3)]
        self.assertAlmostEqual(crrf.ssim(a, b), sum(each) / 3, places=12)
        self.assertEqual(crrf.ssim(a, a), 1.0)


def explicit_lab(r, g, b):
    """IEC 61966-2-1 sRGB and the CIE 1976 formulas under D65, written out for comparison with scikit-image."""
    def linear(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4

    def f(t):
        return t ** (1.0 / 3.0) if t > 0.008856 else 7.787 * t + 16.0 / 116.0

    lr, lg, lb = linear(r), linear(g), linear(b)
    x = 0.412453 * lr + 0.357580 * lg + 0.180423 * lb
    y = 0.212671 * lr + 0.715160 * lg + 0.072169 * lb
    z = 0.019334 * lr + 0.119193 * lg + 0.950227 * lb
    fx, fy, fz = f(x / 0.95047), f(y), f(z / 1.08883)
    return 116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)


class NormalizedColorDifference(unittest.TestCase):
    def test_lab_of_primary_colors(self):
        # (R, G, B) -> published CIELAB coordinates under D65; the image is stored as B, G, R
        for rgb, expected in [((255, 0, 0), (53.24, 80.09, 67.20)), ((0, 255, 0), (87.73, -86.18, 83.18)),
                              ((0, 0, 255), (32.30, 79.19, -107.86)), ((255, 255, 255), (100.0, 0.0, 0.0))]:
            bgr = np.array([[rgb[::-1]]], np.uint8)
            np.testing.assert_allclose(crrf.metrics.lab_of(bgr)[0, 0], expected, atol=0.02)

    def test_lab_of_matches_the_written_formulas(self):
        rng = np.random.default_rng(5)
        for rgb in rng.integers(0, 256, (200, 3)).tolist():
            bgr = np.array([[rgb[::-1]]], np.uint8)
            np.testing.assert_allclose(crrf.metrics.lab_of(bgr)[0, 0], explicit_lab(*rgb), atol=1e-3)

    def test_white_against_black_is_one(self):
        white = np.full((3, 4, 3), 255, np.uint8)
        self.assertAlmostEqual(crrf.ncd(white, np.zeros((3, 4, 3), np.uint8)), 1.0, places=9)

    def test_ratio_of_sums_and_not_mean_of_ratios(self):
        ref = np.array([[[255, 255, 255], [0, 0, 255]]], np.uint8)      # white and red, in B, G, R
        out = np.array([[[255, 255, 255], [0, 0, 0]]], np.uint8)        # white and black
        red = math.sqrt(sum(v ** 2 for v in explicit_lab(255, 0, 0)))
        self.assertAlmostEqual(crrf.ncd(ref, out), red / (100.0 + red), places=3)

    def test_the_reference_gives_the_denominator(self):
        rng = np.random.default_rng(6)
        a = rng.integers(0, 256, (16, 16, 3)).astype(np.uint8)
        b = rng.integers(0, 256, (16, 16, 3)).astype(np.uint8)
        self.assertEqual(crrf.ncd(a, a), 0.0)
        self.assertNotEqual(crrf.ncd(a, b), crrf.ncd(b, a))
        self.assertEqual(crrf.ncd(a, b), crrf.ncd(a, b, ref_lab=crrf.metrics.lab_of(a)))

    def test_the_channel_order_matters(self):
        rng = np.random.default_rng(7)
        a = rng.integers(0, 256, (16, 16, 3)).astype(np.uint8)
        b = rng.integers(0, 256, (16, 16, 3)).astype(np.uint8)
        self.assertNotEqual(crrf.ncd(a, b), crrf.ncd(a[:, :, ::-1], b[:, :, ::-1]))


def smooth_image(rows, cols, seed=8):
    """A color image with smooth content and edges, in place of a photograph."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)
    img = np.stack([120 + 80 * np.sin(x / 9.0) * np.cos(y / 13.0),
                    110 + 70 * np.sin((x + y) / 17.0),
                    100 + 60 * np.cos(x / 23.0 - y / 7.0)], axis=2)
    img[rows // 3:rows // 2, cols // 4:cols // 2] += 60
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8)


class FeatureSimilarity(unittest.TestCase):
    def test_conv2_same_follows_matlab(self):
        # MATLAB: conv2([1 2 3], [1 1], 'same') = [3 5 3]; conv2([1 2 3], [1 2 1], 'same') = [4 8 8]
        from crrf.fsim import conv2_same
        a = np.array([[1.0, 2.0, 3.0]])
        np.testing.assert_array_equal(conv2_same(a, np.array([[1.0, 1.0]])), [[3.0, 5.0, 3.0]])
        np.testing.assert_array_equal(conv2_same(a, np.array([[1.0, 2.0, 1.0]])), [[4.0, 8.0, 8.0]])

    def test_conv2_same_matches_the_examples_of_the_matlab_documentation(self):
        # mathworks.com/help/matlab/ref/conv.html: conv(u, v, "same") with a kernel of even length
        from scipy.signal import convolve2d

        from crrf.fsim import conv2_same
        u = np.array([[-1.0, 2.0, 3.0, -2.0, 0.0, 1.0, 2.0]])
        v = np.array([[2.0, 4.0, -1.0, 1.0]])
        np.testing.assert_array_equal(conv2_same(u, v), [[15.0, 5.0, -9.0, 7.0, 6.0, 7.0, -1.0]])
        # mathworks.com/help/matlab/ref/conv2.html: A is 3-by-3, B is 4-by-4, Csame = Cfull(3:5, 3:5)
        a = np.arange(1.0, 10.0).reshape(3, 3)
        b = np.arange(1.0, 17.0).reshape(4, 4)
        full = convolve2d(a, b, mode="full")
        np.testing.assert_array_equal(conv2_same(a, b), full[2:5, 2:5])

    def test_downsampling_factor_rounds_halves_up(self):
        from crrf.fsim import downsampling_factor
        for rows, cols, f in [(100, 100, 1), (321, 481, 1), (481, 321, 1), (384, 512, 2), (512, 512, 2),
                              (640, 900, 3), (1024, 768, 3)]:
            self.assertEqual(downsampling_factor(rows, cols), f)

    def test_frequency_coordinates(self):
        from crrf.fsim import centered_range
        np.testing.assert_array_equal(centered_range(5), [-0.5, -0.25, 0.0, 0.25, 0.5])
        np.testing.assert_array_equal(centered_range(4), [-0.5, -0.25, 0.0, 0.25])

    def test_chroma_term_is_the_real_part_of_the_principal_power(self):
        from crrf.fsim import LAMBDA, chroma_term
        product = np.array([0.0, 0.25, 1.0, -0.5, -1e-3])
        expected = [(complex(v) ** LAMBDA).real for v in product]
        np.testing.assert_allclose(chroma_term(product), expected, rtol=1e-14)

    def test_identical_images_give_one(self):
        for rows, cols in [(321, 481), (512, 512)]:
            img = smooth_image(rows, cols)
            self.assertAlmostEqual(crrf.fsimc(img, img), 1.0, places=12)

    def test_downsampling_is_the_block_mean_on_an_even_size(self):
        from crrf.fsim import features
        img = smooth_image(512, 512)
        f = features(img)
        r, g, b = [img[:, :, c].astype(np.float64) for c in range(3)]
        i = 0.596 * r - 0.274 * g - 0.322 * b
        blocks = i.reshape(256, 2, 256, 2).mean(axis=(1, 3))
        np.testing.assert_allclose(f["i"], blocks, atol=1e-10)

    def test_symmetry_and_distortion(self):
        img = smooth_image(128, 160)
        noisy = np.clip(img.astype(np.float64) + np.random.default_rng(9).normal(0, 25, img.shape), 0, 255)
        noisy = noisy.astype(np.uint8)
        self.assertAlmostEqual(crrf.fsimc(img, noisy), crrf.fsimc(noisy, img), places=12)
        self.assertLess(crrf.fsimc(img, noisy), 1.0)

    def test_chroma_changes_only_fsimc(self):
        # Equal luminance and different chrominance: the phase and gradient terms stay close to 1.
        from crrf.fsim import feature_sim
        rgb = smooth_image(128, 160)
        other = rgb.copy()
        other[:, :, 0], other[:, :, 2] = rgb[:, :, 2], rgb[:, :, 0]
        fs, fc = feature_sim(rgb, other)
        self.assertLess(fc, fs)

    def test_the_channel_order_matters(self):
        a = smooth_image(128, 160, seed=10)
        b = smooth_image(128, 160, seed=11)[:, ::-1]
        self.assertNotEqual(crrf.fsimc(a, b), crrf.fsimc(a[:, :, ::-1], b[:, :, ::-1]))

    def test_reference_maps_can_be_reused(self):
        a = smooth_image(64, 80, seed=12)
        b = smooth_image(64, 80, seed=13)
        self.assertEqual(crrf.fsimc(a, b), crrf.fsimc(a, b, ref_features=crrf.metrics.fsim_features(a)))


if __name__ == "__main__":
    unittest.main()
