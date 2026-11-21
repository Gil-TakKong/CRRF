# -*- coding: utf-8 -*-
"""The sorting-network path of Phase 2: each level, the composed median, and the window-by-window kernel."""
import itertools
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import crrf                                                         # noqa: E402
from crrf import median_network as N                                # noqa: E402
from crrf import variants as V                                      # noqa: E402
from references import reference_variant                            # noqa: E402


def apply_network(kept, wires):
    """Run a pruned network on a list of 0-1 values (op 0 = both outputs, 1 = min only, 2 = max only)."""
    v = list(wires)
    for i, j, op in kept:
        low, high = min(v[i], v[j]), max(v[i], v[j])
        if op in (0, 1):
            v[i] = low
        if op in (0, 2):
            v[j] = high
    return v


def sorted_inputs(n):
    """All sorted 0-1 sequences of length n."""
    return [[0] * (n - k) + [1] * k for k in range(n + 1)]


class ZeroOnePrinciple(unittest.TestCase):
    """A min/max network that is right on every 0-1 input is right on every input."""

    def test_l1_sorts_seven_values(self):
        for bits in itertools.product((0, 1), repeat=7):
            self.assertEqual(apply_network(N.LEVELS[0], bits), sorted(bits))

    def check_merge(self, kept, m, n, order, ranks):
        for a in sorted_inputs(m):
            for b in sorted_inputs(n):
                out = apply_network(kept, a + b)
                truth = sorted(a + b)
                for rank in ranks:
                    self.assertEqual(out[order[rank]], truth[rank], (a, b, rank))

    def test_l2_merges_two_columns(self):
        self.check_merge(N.LEVELS[1], 7, 7, N.P_ORD, range(14))

    def test_l3_keeps_ranks_3_to_24_of_28(self):
        self.check_merge(N.LEVELS[2], 14, 14, N.D_ORD, range(3, 25))

    def test_l4_keeps_ranks_14_to_21_of_36(self):
        self.check_merge(N.LEVELS[3], 22, 14, N.K_ORD, range(14, 22))

    def test_l5_finds_rank_7_of_15(self):
        self.check_merge(N.LEVELS[4], 8, 7, N.F_ORD, [7])

    def test_generated_code_equals_the_generator(self):
        self.assertTrue(N.check_generated())


class ComposedMedian(unittest.TestCase):
    """The five kernels on one chunk of a 7-row band, as Phase 2 calls them."""

    def medians_of_band(self, band):
        """band: 7 rows by 2 * SR padded columns. Returns the medians of windows 2q and 2q + 1, q < CQ."""
        even = np.ascontiguousarray(band[:, 0::2])
        odd = np.ascontiguousarray(band[:, 1::2])
        Se, So = np.empty(7 * N.SR), np.empty(7 * N.SR)
        P, D, K = np.empty(14 * N.PR), np.empty(28 * N.CQ), np.empty(36 * N.CQ)
        Me, Mo = np.empty(N.CQ), np.empty(N.CQ)
        N._lv1(even[0], even[1], even[2], even[3], even[4], even[5], even[6], Se)
        N._lv1(odd[0], odd[1], odd[2], odd[3], odd[4], odd[5], odd[6], So)
        N._lv2(So, Se, P)
        N._lv3(P, D)
        N._lv4(D, P, K)
        N._lv5(K, Se, Me)
        N._lv5(K, So[3:], Mo)
        return Me, Mo

    def check(self, band):
        Me, Mo = self.medians_of_band(band)
        for q in range(N.CQ):
            self.assertEqual(Me[q], np.median(band[:, 2 * q:2 * q + 7]), ("even window", q))
            self.assertEqual(Mo[q], np.median(band[:, 2 * q + 1:2 * q + 8]), ("odd window", q))

    def test_random_values(self):
        rng = np.random.default_rng(1)
        for _ in range(20):
            self.check(rng.random((7, 2 * N.SR)))

    def test_many_ties(self):
        rng = np.random.default_rng(2)
        for levels in (2, 3, 7):
            for _ in range(10):
                self.check(rng.integers(0, levels, (7, 2 * N.SR)).astype(np.float64))

    def test_color_directions(self):
        rng = np.random.default_rng(3)
        for _ in range(10):
            u = rng.integers(1, 255, (7, 2 * N.SR, 3)).astype(np.float64)
            self.check(u[:, :, 0] / np.linalg.norm(u, axis=2))


class FastAgainstWindowByWindow(unittest.TestCase):
    """crrf.phase2 and the window-by-window kernel of crrf.variants return the same bits."""

    def check(self, est, marked):
        fast = crrf.phase2(est, marked)
        slow = V.phase2_variant(est, marked)
        self.assertEqual(fast[0].tobytes(), slow[0].tobytes())
        self.assertEqual(fast[1:], slow[1:])
        return fast

    def test_noisy_images_of_many_sizes(self):
        # Widths around the chunk size of the networks, and small sizes where the window wraps.
        rng = np.random.default_rng(4)
        sizes = [(4, 4), (4, 5), (5, 4), (7, 7), (8, 13), (13, 8), (9, 63), (9, 64), (9, 65), (9, 127), (9, 128),
                 (9, 129), (20, 140), (140, 20), (33, 200)]
        for h, w in sizes:
            clean = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
            for d in (0.1, 0.5, 0.9):
                noisy = crrf.add_noise(clean, d, rng)
                marked = crrf.identify(noisy)
                self.check(crrf.phase1(noisy, marked)[0], marked)

    def test_zero_vectors_use_the_slow_path(self):
        rng = np.random.default_rng(5)
        for h, w in [(12, 15), (30, 70), (9, 130)]:
            est = rng.uniform(0.0, 1.5, (h, w, 3))
            est[rng.random((h, w)) < 0.3] = 0.0
            marked = rng.random((h, w, 3)) < 0.4
            self.assertGreater(self.check(est, marked)[1], 0)
            est = rng.uniform(0.0, 255.0, (h, w, 3))
            est[h // 3:2 * h // 3, w // 3:2 * w // 3] = 0.0
            self.check(est, marked)

    def test_single_saturated_channels(self):
        # The anchor rule divides by a small c_hat_a, and restored values leave [0, 255].
        rng = np.random.default_rng(6)
        est = rng.uniform(0.0, 3.0, (40, 45, 3))
        est[:, :, 2] = rng.uniform(200.0, 255.0, (40, 45))
        marked = rng.random((40, 45, 3)) < 0.5
        self.assertGreater(self.check(est, marked)[2], 0)


class VariantsAgainstReference(unittest.TestCase):
    """The configurations of the analysis against the pure Python version."""

    def images(self):
        rng = np.random.default_rng(7)
        est = rng.uniform(0.0, 255.0, (11, 12, 3))
        marked = rng.random(est.shape) < 0.5
        yield est, marked
        dark = rng.uniform(0.0, 2.0, (9, 11, 3))
        yield dark, rng.random(dark.shape) < 0.5
        low = rng.uniform(0.0, 3.0, (10, 10, 3))
        low[:, :, 1] = rng.uniform(100.0, 255.0, (10, 10))
        yield low, rng.random(low.shape) < 0.6

    def test_all_statistics_scopes_and_windows(self):
        for est, marked in self.images():
            for stat in (V.STAT_DIRECTION_MEDIAN, V.STAT_RGB_MEDIAN, V.STAT_DIRECTION_MEAN, V.STAT_CHANNEL_MEDIAN):
                for scope in (V.SCOPE_ALL, V.SCOPE_ANCHOR, V.SCOPE_MAGNITUDE):
                    for radius in (1, 2, 3, 4):
                        a = V.phase2_variant(est, marked, radius, stat, scope)
                        b = reference_variant(est, marked, radius, stat, scope)
                        self.assertEqual(a[0].tobytes(), b[0].tobytes(), (stat, scope, radius))
                        self.assertEqual(a[1:], b[1:], (stat, scope, radius))

    def test_scopes_split_the_pixels_by_k(self):
        est, marked = next(self.images())
        k = marked.sum(axis=2)
        full = V.phase2_variant(est, marked)[0]
        anchor = V.phase2_variant(est, marked, scope=V.SCOPE_ANCHOR)[0]
        magnitude = V.phase2_variant(est, marked, scope=V.SCOPE_MAGNITUDE)[0]
        self.assertTrue(np.array_equal(anchor[k == 3], est[k == 3]))
        self.assertTrue(np.array_equal(anchor[k < 3], full[k < 3]))
        self.assertTrue(np.array_equal(magnitude[k < 3], est[k < 3]))
        self.assertTrue(np.array_equal(magnitude[k == 3], full[k == 3]))


if __name__ == "__main__":
    unittest.main()
