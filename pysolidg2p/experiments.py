#!/usr/bin/env python

# For python 2-3 compatibility
from __future__ import division, print_function

import numpy

__all__ = ['e12_06_109']


def e12_06_109(x, q2):
    x_sel, stat_err, a1p = {}, {}, {}
    x_sel[1] = [0.1, 0.15, 0.2, 0.25, 0.3, 1.0]
    stat_err[1] = [0.0005, 0.0007, 0.0013, 0.0022, 0.0034, 0.0059]
    a1p[1] = [0.131, 0.201, 0.259, 0.313, 0.365, 0.418]
    x_sel[2] = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 1.0]
    stat_err[2] = [0.0009, 0.0007, 0.0009, 0.0010, 0.0012, 0.0017, 0.0024, 0.0033, 0.0052, 0.0101]
    a1p[2] = [0.212, 0.277, 0.335, 0.391, 0.445, 0.493, 0.540, 0.585, 0.630, 0.682]
    x_sel[5] = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 1.0]
    stat_err[5] = [0.0027, 0.0025, 0.0027, 0.0032, 0.0042, 0.0058, 0.0090, 0.0249]
    a1p[5] = [0.504, 0.552, 0.598, 0.643, 0.686, 0.728, 0.771, 0.883]
    x_sel[9] = [0.6, 0.65, 0.7, 0.75, 1.0]
    stat_err[9] = [0.009, 0.009, 0.010, 0.013, 0.038]
    a1p[9] = [0.689, 0.730, 0.771, 0.811, 0.849]

    error = numpy.zeros_like(x)

    for ix, xx in enumerate(x):
        if q2 < 2:
            region = 1
            factor = numpy.sqrt(5)
        elif q2 < 5:
            region = 2
            factor = numpy.sqrt(5)
        elif q2 < 9:
            region = 5
            factor = numpy.sqrt(5)
        else:
            region = 9
            factor = numpy.sqrt(5)

        for jx, _ in enumerate(x_sel[region]):
            if xx < x_sel[region][jx]:
                error[ix] = stat_err[region][jx] / a1p[region][jx] * factor
                break

    return error
