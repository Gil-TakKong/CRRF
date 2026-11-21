# -*- coding: utf-8 -*-
"""Slow pure Python versions of VMF, FAPGF and the Phase 2 kernel, written from their definitions.

The tests compare them bit for bit with the Numba code on small images, so they add in the same order.
"""
import math

import numpy as np

from crrf.geometry import NORM_MIN
from crrf.variants import G_CUM, G_DC, G_DR


def vmf_reference(noisy_u8):
    """VMF: the sample of the 3x3 window with the smallest sum of L2 distances (first in raster order on ties)."""
    x = np.asarray(noisy_u8)
    h, w, _ = x.shape

    def ref101(i, n):
        return -i if i < 0 else (2 * n - 2 - i if i >= n else i)

    offs = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)]
    out = np.empty_like(x)
    for i in range(h):
        for j in range(w):
            win = [tuple(float(v) for v in x[ref101(i + a, h), ref101(j + b, w)]) for a, b in offs]
            sums = []
            for p in win:
                s = 0.0
                for q in win:
                    s += math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 + (p[2] - q[2]) ** 2)
                sums.append(s)
            best = min(range(9), key=lambda k: (sums[k], k))
            out[i, j] = x[ref101(i + offs[best][0], h), ref101(j + offs[best][1], w)]
    return out


def fapgf_reference(noisy_u8, d=0.1, gamma=0.8, border="reflect101"):
    """FAPGF with the keys of fapgf_detail: out, m, replaced, radius."""
    x8 = np.asarray(noisy_u8)
    h, w, _ = x8.shape
    crop = border == "crop"
    t = float(d) * 255.0 * math.sqrt(3.0)
    thr2 = t * t
    g = float(gamma)
    rmax = max(h - 1, w - 1, 1)
    px = [[tuple(float(v) for v in x8[i, j]) for j in range(w)] for i in range(h)]

    def refl(i, n):
        p = 2 * (n - 1)
        i = i % p
        return p - i if i >= n else i

    def window(i, j, r):
        """Non-center window coordinates in raster order."""
        out = []
        for di in range(-r, r + 1):
            ii = i + di
            if crop:
                if not 0 <= ii < h:
                    continue
            else:
                ii = refl(ii, h)
            for dj in range(-r, r + 1):
                if di == 0 and dj == 0:
                    continue
                jj = j + dj
                if crop:
                    if not 0 <= jj < w:
                        continue
                else:
                    jj = refl(jj, w)
                out.append((ii, jj))
        return out

    m = [[0] * w for _ in range(h)]
    for i in range(h):
        for j in range(w):
            a = px[i][j]
            cnt = 0
            for ii, jj in window(i, j, 1):
                b = px[ii][jj]
                if (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2 < thr2:
                    cnt += 1
            m[i][j] = cnt

    out = np.empty((h, w, 3), np.float64)
    radius = np.zeros((h, w), np.int64)
    for i in range(h):
        for j in range(w):
            if m[i][j] >= 2:
                out[i, j] = px[i][j]
                continue
            r = 1
            while True:
                sw, acc, ngood = 0.0, [0.0, 0.0, 0.0], 0
                for ii, jj in window(i, j, r):
                    mi = m[ii][jj]
                    if mi >= 2:
                        ngood += 1
                    if mi > 0:
                        wgt = float(mi) ** g
                        sw += wgt
                        b = px[ii][jj]
                        acc = [acc[0] + wgt * b[0], acc[1] + wgt * b[1], acc[2] + wgt * b[2]]
                done = (sw > 0.0) if r == 1 else (ngood >= 2)
                if done or r >= rmax:
                    break
                r += 1
            if sw > 0.0:
                out[i, j] = (acc[0] / sw, acc[1] / sw, acc[2] / sw)
                radius[i, j] = r
            else:
                out[i, j] = px[i][j]
                radius[i, j] = -1
    return dict(out=out, m=np.array(m, np.int64), replaced=radius > 0, radius=radius)


def reference_variant(src, det, radius, stat, scope):
    """The Phase 2 configurations of crrf.variants, with sorted() for the median."""
    h, w, _ = src.shape
    out = src.copy()
    offs = [(int(a), int(b)) for a, b in zip(G_DR[:G_CUM[radius - 1]], G_DC[:G_CUM[radius - 1]])]

    def nrm3(v):
        return float(np.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))

    def med(vals):
        s = sorted(vals)
        n = len(s)
        return s[n // 2] if n % 2 == 1 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    def rf(i, n):
        return -i if i < 0 else (2 * n - 2 - i if i >= n else i)

    n_skip = n_over = 0
    for x in range(h):
        for y in range(w):
            ks = [c for c in range(3) if det[x, y, c]]
            k = len(ks)
            if k == 0 or (scope == 1 and k == 3) or (scope == 2 and k != 3):
                continue
            u = [float(src[x, y, c]) for c in range(3)]
            norm_p = nrm3(u)
            if norm_p <= NORM_MIN:
                n_skip += 1
                continue
            pts = [(x, y)] + [(rf(x + a, h), rf(y + b, w)) for a, b in offs]
            samp = []
            for (px, py) in pts:
                v = [float(src[px, py, c]) for c in range(3)]
                norm_q = nrm3(v)
                if norm_q <= NORM_MIN:
                    continue
                samp.append([v[c] / norm_q for c in range(3)] if stat in (0, 2) else v)
            if stat == 3:
                for c in ks:
                    val = med([s[c] for s in samp])
                    n_over += int(val < 0.0 or val > 255.0)
                    out[x, y, c] = val
                continue
            if stat == 2:
                m = []
                for c in range(3):
                    acc = 0.0
                    for s in samp:
                        acc += s[c]
                    m.append(acc / len(samp))
            else:
                m = [med([s[c] for s in samp]) for c in range(3)]
            nm = float(np.sqrt(m[0] * m[0] + m[1] * m[1] + m[2] * m[2]))
            if nm == 0.0:
                n_skip += 1
                continue
            m = [v / nm for v in m]
            if k == 3:
                for c in range(3):
                    val = norm_p * m[c]
                    n_over += int(val < 0.0 or val > 255.0)
                    out[x, y, c] = val
            else:
                for c in ks:
                    acc, cnt = 0.0, 0
                    for a in range(3):
                        if a in ks:
                            continue
                        if m[a] > 0.0:
                            acc += u[a] * (m[c] / m[a])
                            cnt += 1
                    if cnt > 0:
                        val = acc / cnt
                        n_over += int(val < 0.0 or val > 255.0)
                        out[x, y, c] = val
    return out, n_skip, n_over
