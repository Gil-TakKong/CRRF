# -*- coding: utf-8 -*-
"""Add channel-independent salt-and-pepper noise to an image, restore it and print the scores.

    python demo.py                                  built-in synthetic image, density 50 %
    python demo.py --image photo.png --density 0.7 --out restored.png
    python demo.py --method DBA                     any of the nine compared methods, or Phase1
"""
# %% Imports
import argparse
import time

import cv2
import numpy as np

import crrf
from crrf.baselines import BASELINES, run_baseline


def synthetic_image(h=128, w=160):
    """Smooth color gradients with a few flat color patches and edges (8-bit, BGR)."""
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack([40 + 150 * xx / (w - 1), 60 + 130 * yy / (h - 1), 200 - 120 * (xx + yy) / (h + w - 2)], axis=2)
    img[h // 4:h // 2, w // 4:w // 2] = (35, 120, 210)
    img[h // 2:3 * h // 4, w // 2:3 * w // 4] = (180, 90, 40)
    return np.clip(np.rint(img), 1, 254).astype(np.uint8)


# %% Main
def main(argv=None):
    ap = argparse.ArgumentParser(description="CRRF demo")
    ap.add_argument("--image", default=None, help="color image file (default: a synthetic image)")
    ap.add_argument("--density", type=float, default=0.5, help="noise density d (default: 0.5)")
    ap.add_argument("--seed", type=int, default=0, help="seed of numpy.random.default_rng (default: 0)")
    ap.add_argument("--method", default="CRRF", help="CRRF, Phase1 or a compared method: " + ", ".join(BASELINES))
    ap.add_argument("--out", default=None, help="file for the restored image (default: not saved)")
    ap.add_argument("--noisy-out", default=None, help="file for the noisy image (default: not saved)")
    a = ap.parse_args(argv)

    if a.image is None:
        clean = synthetic_image()
    else:
        clean = cv2.imdecode(np.fromfile(a.image, np.uint8), cv2.IMREAD_COLOR)
        if clean is None:
            raise SystemExit(f"cannot read image: {a.image}")
    noisy = crrf.to_uint8(crrf.add_noise(clean, a.density, np.random.default_rng(a.seed)))

    t0 = time.perf_counter()
    if a.method == "CRRF":
        restored = crrf.crrf(noisy)
    elif a.method == "Phase1":
        restored = crrf.phase1_only(noisy)
    else:
        restored = run_baseline(a.method, noisy)
    sec = time.perf_counter() - t0

    h, w = clean.shape[:2]
    print(f"image {a.image or 'synthetic'} ({w} x {h}), density {a.density}, seed {a.seed}")
    print(f"noisy     CPSNR {crrf.cpsnr(clean, noisy):6.2f} dB   FSIMc {crrf.fsimc(clean, noisy):.4f}"
          f"   NCD {crrf.ncd(clean, noisy):.4f}")
    print(f"{a.method:<9} CPSNR {crrf.cpsnr(clean, restored):6.2f} dB   FSIMc {crrf.fsimc(clean, restored):.4f}"
          f"   NCD {crrf.ncd(clean, restored):.4f}"
          f"   ({sec:.3f} s, the first call includes compilation)")
    for path, img in ((a.out, restored), (a.noisy_out, noisy)):
        if path:
            ok, buf = cv2.imencode(path[path.rfind("."):], img)
            if not ok:
                raise SystemExit(f"cannot encode image: {path}")
            buf.tofile(path)
            print("saved", path)


# %% Spyder: change ARGS and run this cell
ARGS = None        # for example ["--density", "0.9", "--method", "Phase1"]
if __name__ == "__main__":
    main(ARGS)
