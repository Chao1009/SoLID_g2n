#!/usr/bin/env python3

# For python 2-3 compatibility
from __future__ import division, print_function

import numpy

__all__ = ['load']


def load(filename='yield.txt'):
    x, q2, yield_ = numpy.loadtxt(filename, unpack=True)

    return x, q2, yield_
