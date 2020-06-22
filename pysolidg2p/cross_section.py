#!/usr/bin/env python

# For python 2-3 compatibility
from __future__ import division, print_function

import numpy
from scipy import constants

from .structure_f import r, f1p_lhapdf, f2p_lhapdf, f1p_slac, f2p_slac, g1p_slac, g2p_slac, f1n_slac, f2n_slac, g1n_slac, g2n_slac

__all__ = ['dxslp', 'dxstp', 'xsp', 'dxsln', 'dxstn', 'xsn']

_alpha = constants.alpha
_m_p = constants.value('proton mass energy equivalent in MeV') * 1e-3
_inv_fm_to_gev = constants.hbar * constants.c / constants.e * 1e6
_inv_gev_to_fm = _inv_fm_to_gev
_inv_gev_to_mkb = _inv_gev_to_fm**2 * 1e4


def _mott(e, theta):
    # mott xs
    cos_theta_2 = numpy.cos(theta / 2)
    sin2_theta_2 = 1 - cos_theta_2**2
    return (_alpha * cos_theta_2 / (2 * e * sin2_theta_2))**2 * _inv_gev_to_mkb


def xsp_lhapdf(e, x, q2):
    # total xs
    nu = q2 / (2 * _m_p * x)
    ep = e - nu
    theta = 2 * numpy.arcsin(numpy.sqrt(q2 / (4 * e * ep)))
    gamma2 = 4 * _m_p**2 * x**2 / q2

    rr, err = r(x, q2)
    f1, _ = f1p_lhapdf(x, q2)
    f2, ef2 = f2p_lhapdf(x, q2)

    A = 2 / _m_p * numpy.tan(theta / 2)**2
    B = 1 / nu
    C = (1 + gamma2) / (2 * x * (1 + rr))
    result = _mott(e, theta) * (A * f1 + B * f2)
    # f1 and f2 are correlated
    error = _mott(e, theta) * numpy.sqrt(((A * C + B) * ef2)**2 + (A * f2 * C / (1 + rr) * err)**2)

    return result, error


def xsp_slac(e, x, q2):
    # total xs
    nu = q2 / (2 * _m_p * x)
    ep = e - nu
    theta = 2 * numpy.arcsin(numpy.sqrt(q2 / (4 * e * ep)))

    return _mott(e, theta) * (2 / _m_p * f1p_slac(x, q2) * numpy.tan(theta / 2)**2 + 1 / nu * f2p_slac(x, q2))


def xsn_slac(e, x, q2):
    # total xs
    nu = q2 / (2 * _m_p * x)
    ep = e - nu
    theta = 2 * numpy.arcsin(numpy.sqrt(q2 / (4 * e * ep)))

    return _mott(e, theta) * (2 / _m_p * f1n_slac(x, q2) * numpy.tan(theta / 2)**2 + 1 / nu * f2n_slac(x, q2))


def dxslp_slac(e, x, q2):
    # longitudinal xs difference
    nu = q2 / (2 * _m_p * x)
    ep = e - nu
    theta = 2 * numpy.arcsin(numpy.sqrt(q2 / (4 * e * ep)))
    sigma0 = 4 * _alpha**2 * ep / (nu * _m_p * q2 * e) * _inv_gev_to_mkb

    return sigma0 * ((e + ep * numpy.cos(theta)) * g1p_slac(x, q2) - 2 * _m_p * x * g2p_slac(x, q2))


def dxsln_slac(e, x, q2):
    # longitudinal xs difference
    nu = q2 / (2 * _m_p * x)
    ep = e - nu
    theta = 2 * numpy.arcsin(numpy.sqrt(q2 / (4 * e * ep)))
    sigma0 = 4 * _alpha**2 * ep / (nu * _m_p * q2 * e) * _inv_gev_to_mkb

    return sigma0 * ((e + ep * numpy.cos(theta)) * g1n_slac(x, q2) - 2 * _m_p * x * g2n_slac(x, q2))


def dxstp_slac(e, x, q2):
    # transverse xs difference
    nu = q2 / (2 * _m_p * x)
    ep = e - nu
    theta = 2 * numpy.arcsin(numpy.sqrt(q2 / (4 * e * ep)))
    sigma0 = 4 * _alpha**2 * ep / (nu * _m_p * q2 * e) * _inv_gev_to_mkb

    return sigma0 * ep * numpy.sin(theta) * (g1p_slac(x, q2) + 2 * e * g2p_slac(x, q2) / nu)


def dxstn_slac(e, x, q2):
    # transverse xs difference
    nu = q2 / (2 * _m_p * x)
    ep = e - nu
    theta = 2 * numpy.arcsin(numpy.sqrt(q2 / (4 * e * ep)))
    sigma0 = 4 * _alpha**2 * ep / (nu * _m_p * q2 * e) * _inv_gev_to_mkb

    return sigma0 * ep * numpy.sin(theta) * (g1n_slac(x, q2) + 2 * e * g2n_slac(x, q2) / nu)


def xsp(e, x, q2, model='slac', **kwargs):
    xsp_func = {
        'lhapdf': xsp_lhapdf,
        'slac': xsp_slac,
    }.get(model, None)

    return xsp_func(e, x, q2, **kwargs)


def dxslp(e, x, q2, model='slac', **kwargs):
    dxslp_func = {
        'slac': dxslp_slac,
    }.get(model, None)

    return dxslp_func(e, x, q2, **kwargs)


def dxstp(e, x, q2, model='slac', **kwargs):
    dxstp_func = {
        'slac': dxstp_slac,
    }.get(model, None)

    return dxstp_func(e, x, q2, **kwargs)


def xsn(e, x, q2, model='slac', **kwargs):
    xsn_func = {
        'slac': xsn_slac,
    }.get(model, None)

    return xsn_func(e, x, q2, **kwargs)


def dxsln(e, x, q2, model='slac', **kwargs):
    dxsln_func = {
        'slac': dxsln_slac,
    }.get(model, None)

    return dxsln_func(e, x, q2, **kwargs)


def dxstn(e, x, q2, model='slac', **kwargs):
    dxstn_func = {
        'slac': dxstn_slac,
    }.get(model, None)

    return dxstn_func(e, x, q2, **kwargs)
