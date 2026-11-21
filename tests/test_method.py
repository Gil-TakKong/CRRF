# -*- coding: utf-8 -*-
"""Identification, Phase 1 and Phase 2 on hand-worked cases, and the public API.

The images are small arrays written out here; no dataset is needed.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crrf                                                         # noqa: E402
from crrf.core import MAX_ITER                                      # noqa: E402
from crrf.geometry import insertion_median, moore_rings, ref101     # noqa: E402

M = 255.0                       # a value identified as noise in the hand-worked cases


def uniform(vector, size=9):
    """A size x size float64 image in which every pixel has the given vector."""
    img = np.empty((size, size, 3), np.float64)
    img[:] = vector
    return img


class Identification(unittest.TestCase):
    def test_marks_exactly_0_and_255(self):
        img = np.array([[[0, 1, 254], [255, 128, 0]]], np.uint8)
        expected = [[[True, False, False], [True, False, True]]]
        self.assertEqual(crrf.identify(img).tolist(), expected)
        self.assertEqual(crrf.identify(img.astype(np.float64)).tolist(), expected)


class Phase1(unittest.TestCase):
    def run_phase1(self, channel_0, channel_1=None, channel_2=None):
        h, w = channel_0.shape
        noisy = np.empty((h, w, 3), np.float64)
        noisy[:, :, 0] = channel_0
        noisy[:, :, 1] = 100.0 if channel_1 is None else channel_1
        noisy[:, :, 2] = 100.0 if channel_2 is None else channel_2
        return crrf.phase1(noisy, crrf.identify(noisy))

    def test_synchronous_update(self):
        # t = 1: (0,1) = (10+30)/2 = 20, (1,0) = (10+50)/2 = 30, (1,2) = (30+70)/2 = 50, (2,1) = (50+70)/2 = 60.
        #        The center has no valid neighbor in the state before this iteration, although its
        #        neighbors are restored in the same iteration. A sequential sweep would give 25 here.
        # t = 2: center = (20+30+50+60)/4 = 40.
        # t = 3: restores nothing and ends the loop.
        channel = np.array([[10, M, 30],
                            [M, M, M],
                            [50, M, 70]])
        est, n_iter, left = self.run_phase1(channel)
        self.assertEqual(est[:, :, 0].tolist(), [[10, 20, 30], [30, 40, 50], [50, 60, 70]])
        self.assertEqual(n_iter, 3)
        self.assertEqual(left, 0)

    def test_reflect_101_counts_a_mirrored_neighbor_twice(self):
        # Corner: N mirrors to S and W mirrors to E, so the mean is (10+40+40+10)/4.
        channel = np.array([[M, 40, 40],
                            [10, 40, 40],
                            [40, 40, 40]])
        est, n_iter, _ = self.run_phase1(channel)
        self.assertEqual(est[0, 0, 0], 25.0)
        self.assertEqual(n_iter, 2)

    def test_channels_are_independent(self):
        c0 = np.array([[M, 40, 40], [10, 40, 40], [40, 40, 40]])
        c1 = np.array([[20, 20, 20], [20, 0, 20], [20, 20, 20]])
        est, n_iter, left = self.run_phase1(c0, c1)
        self.assertEqual(est[0, 0].tolist(), [25.0, 20.0, 100.0])
        self.assertEqual(est[1, 1].tolist(), [40.0, 20.0, 100.0])
        self.assertEqual((n_iter, left), (2, 0))

    def test_without_noise_nothing_changes_and_one_iteration_runs(self):
        channel = np.arange(1.0, 26.0).reshape(5, 5)
        est, n_iter, left = self.run_phase1(channel)
        self.assertTrue(np.array_equal(est[:, :, 0], channel))
        self.assertEqual((n_iter, left), (1, 0))

    def test_a_fully_marked_channel_stays_unrestored(self):
        est, n_iter, left = self.run_phase1(np.full((3, 4), M))
        self.assertTrue(np.all(est[:, :, 0] == M))
        self.assertEqual((n_iter, left), (1, 12))

    def test_safety_cap(self):
        # One valid value in a 2 x 600 image spreads by one step per iteration, so the cap ends the loop.
        channel = np.full((2, 600), M)
        channel[0, 0] = 50.0
        est, n_iter, left = self.run_phase1(channel)
        self.assertEqual(n_iter, MAX_ITER)
        self.assertEqual(MAX_ITER, 500)
        self.assertGreater(left, 0)
        self.assertEqual(est[1, 0, 0], 50.0)


class Phase2Rules(unittest.TestCase):
    """The window direction comes from the uniform vector, c_hat = (1/3, 2/3, 2/3) for (10, 20, 20).

    The center sample alone cannot move the median of 49 samples.
    """

    def run_center(self, base, center, marked_channels):
        est = uniform(base)
        est[4, 4] = center
        marked = np.zeros(est.shape, np.bool_)
        marked[4, 4, list(marked_channels)] = True
        out, n_skip, n_over = crrf.phase2(est, marked)
        others = np.ones((9, 9), np.bool_)
        others[4, 4] = False
        self.assertTrue(np.array_equal(out[others], est[others]), "a pixel without a marked component changed")
        return out[4, 4], n_skip, n_over

    def test_k1_mean_over_two_anchors(self):
        # x_1 = (10 * (2/3)/(1/3) + 40 * (2/3)/(2/3)) / 2 = (20 + 40) / 2
        v, n_skip, n_over = self.run_center((10, 20, 20), (10, 77, 40), [1])
        self.assertAlmostEqual(v[1], 30.0, places=10)
        self.assertEqual((v[0], v[2]), (10.0, 40.0))
        self.assertEqual((n_skip, n_over), (0, 0))

    def test_k2_single_anchor(self):
        # x_0 = 40 * (1/3)/(2/3) = 20, x_1 = 40 * (2/3)/(2/3) = 40
        v, _, _ = self.run_center((10, 20, 20), (77, 88, 40), [0, 1])
        self.assertAlmostEqual(v[0], 20.0, places=10)
        self.assertAlmostEqual(v[1], 40.0, places=10)
        self.assertEqual(v[2], 40.0)

    def test_three_corrupted_keep_the_phase1_magnitude(self):
        # ||u(p)|| = ||(20, 20, 10)|| = 30, x = 30 * c_hat
        v, _, _ = self.run_center((10, 20, 20), (20, 20, 10), [0, 1, 2])
        np.testing.assert_allclose(v, (10.0, 20.0, 20.0), atol=1e-10)

    def test_an_anchor_with_a_zero_direction_is_excluded(self):
        # c_hat = (0, 0.6, 0.8). Channel 0 cannot serve as an anchor, so only channel 2 does: 40 * 0.6/0.8.
        v, _, _ = self.run_center((0, 3, 4), (7, 99, 40), [1])
        self.assertAlmostEqual(v[1], 30.0, places=10)

    def test_no_anchor_keeps_phase1(self):
        v, n_skip, _ = self.run_center((0, 3, 4), (7, 99, 40), [1, 2])
        self.assertEqual(v.tolist(), [7.0, 99.0, 40.0])
        self.assertEqual(n_skip, 0)

    def test_values_are_not_clipped_and_outliers_are_counted(self):
        # x_1 = (200 * 2 + 200 * 1) / 2 = 300, which only the final conversion clips.
        v, _, n_over = self.run_center((10, 20, 20), (200, 77, 200), [1])
        self.assertAlmostEqual(v[1], 300.0, places=9)
        self.assertEqual(n_over, 1)
        self.assertEqual(crrf.to_uint8(v).tolist(), [200, 255, 200])

    def test_zero_samples_are_excluded(self):
        # Thirty zero vectors have no color direction and are left out; the rest give (1/3, 2/3, 2/3).
        est = uniform((10, 20, 20))
        coords = [(x, y) for x in range(1, 8) for y in range(1, 8) if (x, y) != (4, 4)]
        for x, y in coords[:30]:
            est[x, y] = (0.0, 0.0, 0.0)
        est[4, 4] = (10, 77, 40)
        marked = np.zeros(est.shape, np.bool_)
        marked[4, 4, 1] = True
        out, n_skip, _ = crrf.phase2(est, marked)
        self.assertAlmostEqual(out[4, 4, 1], 30.0, places=10)
        self.assertEqual(n_skip, 0)

    def test_zero_pixel_keeps_phase1(self):
        v, n_skip, _ = self.run_center((10, 20, 20), (0.0, 0.0, 0.0), [0, 1, 2])
        self.assertEqual(v.tolist(), [0.0, 0.0, 0.0])
        self.assertEqual(n_skip, 1)

    def test_dark_nonzero_pixel_is_restored(self):
        # Only the zero vector is skipped: a norm below 1 still has a color direction, and rule (5) applies.
        v, n_skip, _ = self.run_center((10, 20, 20), (0.5, 0.0, 0.5), [0, 1, 2])
        r = np.sqrt(0.5)
        for got, want in zip(v.tolist(), (r / 3, 2 * r / 3, 2 * r / 3)):
            self.assertAlmostEqual(got, want, places=12)
        self.assertEqual(n_skip, 0)

    def test_zero_median_direction_keeps_phase1(self):
        # Samples along the three axes in thirds: rank 25 of each component is 0, so ||m|| = 0.
        est = np.zeros((9, 9, 3), np.float64)
        for x in range(9):
            for y in range(9):
                est[x, y, (x + y) % 3] = 100.0
        marked = np.zeros(est.shape, np.bool_)
        marked[4, 4] = True
        out, n_skip, _ = crrf.phase2(est, marked)
        self.assertTrue(np.array_equal(out, est))
        self.assertEqual(n_skip, 1)

    def test_invalid_input_is_refused(self):
        est, marked = uniform((10, 20, 20)), np.zeros((9, 9, 3), np.bool_)
        with self.assertRaises(TypeError):
            crrf.phase2(est.astype(np.float32), marked)
        with self.assertRaises(TypeError):
            crrf.phase2(est, marked.astype(np.uint8))
        with self.assertRaises(ValueError):
            crrf.phase2(est[:3, :3], marked[:3, :3])
        for bad in (-1.0, float("nan"), float("inf"), -0.0):
            wrong = est.copy()
            wrong[0, 0, 0] = bad
            with self.assertRaises(ValueError):
                crrf.phase2(wrong, marked)


class PublicApi(unittest.TestCase):
    def test_one_pixel_with_each_k_is_restored_exactly(self):
        # Phase 1 returns the neighbors' vector, and Phase 2 reproduces it on a uniform image.
        clean = uniform((10, 20, 20)).astype(np.uint8)
        for center in ((255, 20, 20), (255, 0, 20), (0, 255, 0)):
            noisy = clean.copy()
            noisy[4, 4] = center
            self.assertTrue(np.array_equal(crrf.crrf(noisy), clean), center)
            self.assertTrue(np.array_equal(crrf.phase1_only(noisy), clean), center)

    def test_output_type_input_and_untouched_components(self):
        clean = np.random.default_rng(1).integers(1, 255, (30, 41, 3)).astype(np.uint8)
        noisy = crrf.to_uint8(crrf.add_noise(clean, 0.5, np.random.default_rng(2)))
        keep = noisy.copy()
        marked = crrf.identify(noisy)
        for fn in (crrf.crrf, crrf.phase1_only):
            out = fn(noisy)
            self.assertEqual(out.dtype, np.uint8)
            self.assertEqual(out.shape, noisy.shape)
            self.assertTrue(np.array_equal(noisy, keep), "the input changed")
            self.assertTrue(np.array_equal(out[~marked], noisy[~marked]), "an unmarked component changed")
            self.assertTrue(np.array_equal(out, fn(noisy)), "not deterministic")

    def test_other_shapes_are_refused(self):
        with self.assertRaises(ValueError):
            crrf.crrf(np.zeros((9, 9), np.uint8))
        with self.assertRaises(ValueError):
            crrf.crrf(np.zeros((3, 9, 3), np.uint8))


class Geometry(unittest.TestCase):
    def test_reflect_101(self):
        self.assertEqual([ref101(i, 5) for i in (-3, -1, 0, 4, 5, 7)], [3, 1, 0, 4, 3, 1])

    def test_moore_rings(self):
        dr, dc, cum = moore_rings(4)
        self.assertEqual(cum.tolist(), [8, 24, 48, 80])
        self.assertEqual(len(set(zip(dr.tolist(), dc.tolist()))), 80)
        self.assertNotIn((0, 0), set(zip(dr.tolist(), dc.tolist())))
        radii = np.maximum(np.abs(dr), np.abs(dc))
        self.assertTrue(np.all(np.diff(radii) >= 0))

    def test_insertion_median(self):
        rng = np.random.default_rng(3)
        for n in (1, 2, 9, 48, 49):
            v = rng.random(n)
            self.assertEqual(insertion_median(v.copy(), n), np.median(v))


if __name__ == "__main__":
    unittest.main()
