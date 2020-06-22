#!/usr/bin/env python

# For python 2-3 compatibility
from __future__ import division, print_function

import numpy as np
from scipy import constants

from .structure_f import r

__all__ = ['dxs_to_g1g2', 'g1g2_to_d2']

_alpha = constants.alpha
_m_p = constants.value('proton mass energy equivalent in MeV') * 1e-3
_inv_fm_to_gev = constants.hbar * constants.c / constants.e * 1e6
_inv_gev_to_fm = _inv_fm_to_gev
_inv_gev_to_mkb = _inv_gev_to_fm**2 * 1e4


def g1g2_to_dxs(e, x, q2, g1, g2):
    nu = q2 / (2 * _m_p * x)
    ep = e - nu
    theta = 2 * np.arcsin(np.sqrt(q2 / (4 * e * ep)))
    sigma0 = 4 * _alpha**2 * ep / (nu * _m_p * q2 * e) * _inv_gev_to_mkb

    A1 = (e + ep * np.cos(theta))
    B1 = -2 * _m_p * x
    A2 = ep * np.sin(theta)
    B2 = 2 * e * ep * np.sin(theta) / nu

    dxsL = (A1*g1 + B1*g2) * sigma0
    dxsT = (A2*g1 + B2*g2) * sigma0

    return dxsL, dxsT


def dxs_to_g1g2(e, x, q2, dxsl, dxst, edxsl, edxst):
    nu = q2 / (2 * _m_p * x)
    ep = e - nu
    theta = 2 * np.arcsin(np.sqrt(q2 / (4 * e * ep)))
    sigma0 = 4 * _alpha**2 * ep / (nu * _m_p * q2 * e) * _inv_gev_to_mkb

    A1 = (e + ep * np.cos(theta))
    B1 = -2 * _m_p * x
    C1 = dxsl / sigma0
    A2 = ep * np.sin(theta)
    B2 = 2 * e * ep * np.sin(theta) / nu
    C2 = dxst / sigma0
    D = A1 * B2 - A2 * B1

    g1 = (C1 * B2 - C2 * B1) / D
    g2 = (-C1 * A2 + C2 * A1) / D

    eC1 = edxsl / sigma0
    eC2 = edxst / sigma0
    eg1 = np.sqrt((eC1 * B2)**2 + (eC2 * B1)**2)/D
    eg2 = np.sqrt((eC1 * A2)**2 + (eC2 * A1)**2)/D

    return g1, g2, eg1, eg2


def atf1a1_to_g2(e, x, q2, at, f1, a1, eat, ef1, ea1):
    nu = q2 / (2 * _m_p * x)
    ep = e - nu
    theta = 2 * np.arcsin(np.sqrt(q2 / (4 * e * ep)))

    gamma2 = 4 * _m_p**2 * x**2 / q2
    epsilon = 1 / (1 + 2 * (1 + 1 / gamma2) * np.tan(theta / 2)**2)

    rr, err = r(x, q2)

    C = nu * f1 / (2 * e)
    A = C * (at * nu * ((1 + epsilon * rr) / (1 - epsilon)) / (ep * np.sin(theta)))
    B = C * a1

    g2 = (A - B) / (1 + gamma2 * nu / (2 * e))
    err_to_eg2 = C * (at * nu * (epsilon * err / (1 - epsilon)) / (ep * np.sin(theta))) / (1 + gamma2 * nu / (2 * e))
    eg2 = np.sqrt((g2 * ef1 / f1)**2 + (A * eat / at)**2 + (B * ea1 / a1)**2 + (err_to_eg2)**2)

    return g2, eg2


def g1g2_to_d2(x, g1, g2, eg1=None, eg2=None, sg1=None, sg2=None):
    # trapzoidal integration
    def trapz(xv, yv):
        return np.sum((yv[1:] + yv[:-1])*np.diff(xv)/2.)

    integrand = x**2 * (2 * g1 + 3 * g2)
    integral = trapz(x, integrand)

    error, syst = 0., 0.
    if eg1 is not None and eg2 is not None:
        eintegrand = x**2 * np.sqrt(4 * eg1**2 + 9 * eg2**2)
        error = trapz(x, eintegrand)

    if sg1 is not None and sg2 is not None:
        sintegrand = x**2 * (np.abs(2*sg1) + np.abs(3*sg2))
        syst = trapz(x, sintegrand)

    return integral, error, syst
