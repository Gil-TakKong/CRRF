# -*- coding: utf-8 -*-
"""The nine compared methods on a small synthetic image; VMF and FAPGF on hand-worked cases and against
their pure Python versions."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import crrf                                                 # noqa: E402
from crrf.baselines import BASELINES, run_baseline          # noqa: E402
from crrf.baselines import fapgf as F                       # noqa: E402
from crrf.baselines.vmf import vmf                          # noqa: E402
from references import fapgf_reference, vmf_reference       # noqa: E402

fapgf_module = sys.modules["crrf.baselines.fapgf"]
BORDERS = fapgf_module.BORDERS


def synthetic(h=44, w=52):
    """Smooth gradients in the three channels, with no component at 0 or 255."""
    yy, xx = np.mgrid[0:h, 0:w]
    return np.clip(np.stack([2 * yy + 30, 2 * xx + 20, yy + xx + 40], axis=2), 1, 254).astype(np.uint8)


class AllNine(unittest.TestCase):
    def test_the_list_is_complete(self):
        self.assertEqual(BASELINES, ["DBA", "NAFSMF", "DAMF", "CA-NGMF", "GSR", "ARmF", "NEWMMF", "VMF", "FAPGF"])

    def test_run_on_a_synthetic_image(self):
        clean = synthetic()
        for d in (0.3, 0.9):
            noisy = crrf.to_uint8(crrf.add_noise(clean, d, np.random.default_rng(1)))
            keep = noisy.copy()
            for name in BASELINES:
                with self.subTest(method=name, density=d):
                    out = run_baseline(name, noisy)
                    self.assertEqual(out.dtype, np.uint8)
                    self.assertEqual(out.shape, noisy.shape)
                    self.assertTrue(np.array_equal(noisy, keep), "the input changed")
                    self.assertFalse(np.array_equal(out, noisy), "nothing was restored")
                    self.assertTrue(np.array_equal(out, run_baseline(name, noisy)), "not deterministic")
                    if d == 0.3:
                        self.assertGreater(crrf.cpsnr(clean, out), crrf.cpsnr(clean, noisy))

    def test_only_gsr_vmf_and_fapgf_change_other_components(self):
        clean = synthetic()
        noisy = crrf.to_uint8(crrf.add_noise(clean, 0.5, np.random.default_rng(2)))
        other = ~crrf.identify(noisy)
        for name in BASELINES:
            if name not in ("GSR", "VMF", "FAPGF"):
                with self.subTest(method=name):
                    self.assertTrue(np.array_equal(run_baseline(name, noisy)[other], noisy[other]))

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            run_baseline("nothing", synthetic())


class Vmf(unittest.TestCase):
    def test_an_impulse_is_replaced_by_the_window_vector(self):
        img = np.empty((5, 5, 3), np.uint8)
        img[:] = (10, 20, 30)
        img[2, 2] = (255, 0, 255)
        expected = np.empty((5, 5, 3), np.uint8)
        expected[:] = (10, 20, 30)
        self.assertTrue(np.array_equal(vmf(img), expected))

    def test_the_output_is_always_a_window_sample(self):
        img = np.random.default_rng(1).integers(0, 256, (12, 13, 3)).astype(np.uint8)
        out = vmf(img)
        for i in range(12):
            for j in range(13):
                window = img[max(i - 1, 0):i + 2, max(j - 1, 0):j + 2].reshape(-1, 3)
                self.assertTrue((window == out[i, j]).all(axis=1).any())

    def test_against_the_pure_python_version(self):
        rng = np.random.default_rng(2)
        for h, w in [(2, 2), (2, 7), (7, 2), (3, 3), (16, 16), (21, 27)]:
            img = rng.integers(0, 256, (h, w, 3)).astype(np.uint8)
            self.assertTrue(np.array_equal(vmf(img), vmf_reference(img)), (h, w))
        img = rng.choice(np.array([0, 255], np.uint8), size=(15, 17, 3))
        self.assertTrue(np.array_equal(vmf(img), vmf_reference(img)), "ties")


class Fapgf(unittest.TestCase):
    def same(self, img, border="reflect101", d=0.1, gamma=0.8):
        """The Numba code and the pure Python version agree bit for bit. Returns the diagnostics."""
        a = fapgf_module.fapgf_detail(img, d=d, gamma=gamma, border=border)
        b = fapgf_reference(img, d=d, gamma=gamma, border=border)
        self.assertEqual(a["out"].dtype, np.float64)
        self.assertTrue(np.array_equal(a["m"], b["m"]))
        self.assertTrue(np.array_equal(a["radius"], b["radius"]))
        self.assertEqual(a["out"].tobytes(), b["out"].tobytes())
        # m >= 2 keeps the pixel, and changes occur only in replaced pixels
        keep = a["m"] >= 2
        self.assertTrue(np.array_equal(a["out"][keep], img[keep].astype(np.float64)))
        self.assertTrue(np.all((a["radius"] == 0) == keep))
        changed = (a["out"] != img.astype(np.float64)).any(axis=2)
        self.assertFalse(np.any(changed & ~a["replaced"]))
        return a

    def test_hand_3x3_crop(self):
        A, B, C = (10, 10, 10), (200, 10, 10), (10, 200, 10)
        D, E, X = (10, 10, 200), (10, 10, 230), (255, 255, 255)
        img = np.array([[A, A, B], [A, X, C], [D, D, E]], np.uint8)
        det = self.same(img, border="crop")
        self.assertEqual(det["m"].tolist(), [[2, 2, 0], [2, 0, 0], [1, 2, 1]])
        mu2 = 2.0 ** 0.8
        A_, D_, E_ = (np.array(v, float) for v in (A, D, E))
        exp = img.astype(np.float64).copy()
        exp[1, 1] = (mu2 * (3 * A_ + D_) + D_ + E_) / (4 * mu2 + 2)     # the center X
        exp[0, 2] = A_                                                  # B: only A has weight among its neighbors
        exp[1, 2] = (mu2 * A_ + mu2 * D_ + E_) / (2 * mu2 + 1)          # C
        exp[2, 0] = (A_ + D_) / 2                                       # D (m = 1): A and D have m = 2
        exp[2, 2] = D_                                                  # E (m = 1): only D has m = 2
        np.testing.assert_allclose(det["out"], exp, atol=1e-9)
        self.assertEqual(det["radius"].tolist(), [[0, 0, 1], [0, 1, 1], [1, 0, 1]])

    def test_hand_5x5_enlarges_the_window(self):
        G = (128, 128, 128)
        inner = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 128, 255), (255, 255, 0),
                 (255, 0, 255), (0, 255, 255), (255, 255, 255)]
        img = np.empty((5, 5, 3), np.uint8)
        img[:] = G
        for k, c in enumerate(inner):
            img[1 + k // 3, 1 + k % 3] = c
        for border in BORDERS:
            det = self.same(img, border=border)
            self.assertTrue(np.all(det["m"][1:4, 1:4] == 0))
            self.assertEqual(det["radius"][2, 2], 2)                    # only the center needs 5 x 5
            np.testing.assert_allclose(det["out"], np.full(img.shape, 128.0), atol=1e-9)

    def test_threshold_boundary(self):
        # The squared threshold for d = 0.1 is 1950.75: (43, 10, 1) lies at 1950 and (44, 4, 0) at 1952.
        near = np.array([[[0, 0, 0], [43, 10, 1]]], np.uint8)
        far = np.array([[[0, 0, 0], [44, 4, 0]]], np.uint8)
        self.assertEqual(self.same(near, border="crop")["m"].tolist(), [[1, 1]])
        self.assertEqual(self.same(far, border="crop")["m"].tolist(), [[0, 0]])
        self.assertAlmostEqual(fapgf_module._thr2(0.1), 1950.75, places=9)

    def test_safeguard_without_weights(self):
        img = np.array([[[0, 0, 0], [255, 0, 0]], [[0, 255, 0], [0, 0, 255]]], np.uint8)
        for border in BORDERS:
            det = self.same(img, border=border)
            self.assertTrue(np.all(det["radius"] == -1))
            self.assertTrue(np.array_equal(det["out"], img.astype(np.float64)))

    def test_random_images(self):
        rng = np.random.default_rng(1)
        for border in BORDERS:
            for h, w in [(2, 2), (2, 7), (7, 2), (3, 3), (5, 9), (16, 16), (21, 27)]:
                self.same(rng.integers(0, 256, (h, w, 3)).astype(np.uint8), border=border)
            for lo, hi in [(100, 141), (0, 60), (90, 200)]:
                self.same(rng.integers(lo, hi, (20, 23, 3)).astype(np.uint8), border=border)
            self.same(rng.choice(np.array([0, 255], np.uint8), size=(22, 25, 3)), border=border)
        img = rng.integers(0, 256, (25, 25, 3)).astype(np.uint8)
        self.assertGreater((self.same(img)["radius"] >= 2).sum(), 0)    # the window really grew
        for d, g in [(0.05, 0.5), (0.2, 1.3)]:
            self.same(img, d=d, gamma=g)

    def test_channel_order_does_not_matter_and_input_stays(self):
        noisy = crrf.to_uint8(crrf.add_noise(synthetic(), 0.5, np.random.default_rng(5)))
        keep = noisy.copy()
        for border in BORDERS:
            a = F(noisy, border=border)
            self.assertTrue(np.array_equal(noisy, keep))
            b = F(np.ascontiguousarray(noisy[:, :, ::-1]), border=border)
            self.assertTrue(np.array_equal(a[:, :, ::-1], b))

    def test_bad_inputs(self):
        for bad in (np.zeros((4, 4), np.uint8), np.zeros((4, 4, 3), np.float64), np.zeros((1, 4, 3), np.uint8)):
            with self.assertRaises((ValueError, TypeError)):
                F(bad)


if __name__ == "__main__":
    unittest.main()
