#!/usr/bin/env python

# For python 2-3 compatibility
from __future__ import division, print_function

import numpy
from scipy import constants

from .cross_section import dxslp_slac, dxstp_slac, xsp_slac, dxsln_slac, dxstn_slac, xsn_slac
from .structure_f import f1p_slac, g1p_slac, g2p_slac, f1n_slac, g1n_slac, f2n_slac, g2n_slac

__all__ = ['a1p', 'a2p', 'alp', 'atp', 'a1n', 'a2n', 'aln', 'atn']

_alpha = constants.alpha
_m_p = constants.value('proton mass energy equivalent in MeV') * 1e-3


def a1p_slac(x, q2):
    gamma2 = 4 * _m_p**2 * x**2 / q2
    return (g1p_slac(x, q2) - gamma2 * g2p_slac(x, q2)) / f1p_slac(x, q2)


def a2p_slac(x, q2):
    gamma = numpy.sqrt(4 * _m_p**2 * x**2 / q2)
    return gamma * (g1p_slac(x, q2) + g2p_slac(x, q2)) / f1p_slac(x, q2)


def alp_slac(e, x, q2):
    return dxslp_slac(e, x, q2) / (2 * xsp_slac(e, x, q2))


def atp_slac(e, x, q2):
    return dxstp_slac(e, x, q2) / (2 * xsp_slac(e, x, q2))


def a1p(x, q2, model='slac', **kwargs):
    a1p_func = {
        'slac': a1p_slac,
    }.get(model, None)

    return a1p_func(x, q2, **kwargs)


def a2p(x, q2, model='slac', **kwargs):
    a2p_func = {
        'slac': a2p_slac,
    }.get(model, None)

    return a2p_func(x, q2, **kwargs)


def alp(e, x, q2, model='slac', **kwargs):
    alp_func = {
        'slac': alp_slac,
    }.get(model, None)

    return alp_func(e, x, q2, **kwargs)


def atp(e, x, q2, model='slac', **kwargs):
    atp_func = {
        'slac': atp_slac,
    }.get(model, None)

    return atp_func(e, x, q2, **kwargs)


def a1n_slac(x, q2):
    gamma2 = 4 * _m_p**2 * x**2 / q2
    return (g1n_slac(x, q2) - gamma2 * g2n_slac(x, q2)) / f1n_slac(x, q2)


def a2n_slac(x, q2):
    gamma = numpy.sqrt(4 * _m_p**2 * x**2 / q2)
    return gamma * (g1n_slac(x, q2) + g2n_slac(x, q2)) / f1n_slac(x, q2)


def aln_slac(e, x, q2):
    return dxsln_slac(e, x, q2) / (2 * xsn_slac(e, x, q2))


def atn_slac(e, x, q2):
    return dxstn_slac(e, x, q2) / (2 * xsn_slac(e, x, q2))


def a1n(x, q2, model='slac', **kwargs):
    a1n_func = {
        'slac': a1n_slac,
    }.get(model, None)

    return a1n_func(x, q2, **kwargs)


def a2n(x, q2, model='slac', **kwargs):
    a2n_func = {
        'slac': a2n_slac,
    }.get(model, None)

    return a2n_func(x, q2, **kwargs)


def aln(e, x, q2, model='slac', **kwargs):
    aln_func = {
        'slac': aln_slac,
    }.get(model, None)

    return aln_func(e, x, q2, **kwargs)


def atn(e, x, q2, model='slac', **kwargs):
    atn_func = {
        'slac': atn_slac,
    }.get(model, None)

    return atn_func(e, x, q2, **kwargs)
