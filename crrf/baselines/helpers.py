#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Numba primitives shared by the filters in filters.py.

  - is_noisy:       value test, True if the value is 0 or 255
  - median_array:   insertion-sort median of a[0:n] (for even n, the mean of
                    the two middle values)
"""

import numpy as np
from numba import njit


@njit(cache=True, inline='always')
def is_noisy(v):
    """True if v is salt (255) or pepper (0)."""
    return v == 0.0 or v == 255.0


@njit(cache=True)
def median_array(a, n):
    """
    Median of first n elements of array `a` via insertion sort on copy.

    For even n the median is the mean of the two middle values.

    Args:
        a: 1D float64 numpy array (length >= n)
        n: number of elements to consider (a[0..n-1])

    Returns:
        median value (float)
    """
    b = np.empty(n, np.float64)
    for i in range(n):
        b[i] = a[i]
    for i in range(1, n):
        k = b[i]
        j = i - 1
        while j >= 0 and b[j] > k:
            b[j + 1] = b[j]
            j -= 1
        b[j + 1] = k
    return b[n // 2] if n % 2 == 1 else 0.5 * (b[n // 2 - 1] + b[n // 2])
