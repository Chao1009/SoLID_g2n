#!/usr/bin/env python

# For python 2-3 compatibility
from __future__ import division, print_function

import numpy
from scipy import constants, integrate

import lhapdf

__all__ = ['f1p', 'f2p', 'g1p', 'g2p', 'f1n', 'f2n', 'g1n', 'g2n', 'r']

_m_p = constants.value('proton mass energy equivalent in MeV') * 1e-3


def _parton_lhapdf(pid, x, q2, pdf_set, pdfs):
    vals = numpy.zeros(len(pdfs))
    for i, pdf in enumerate(pdfs):
        vals[i] = pdf.xfxQ2(pid, x, q2)

    errors = pdf_set.uncertainty(vals)

    return errors.central, errors.errsymm


def r(x, q2):
    # SLAC E143
    # Phys. Lett. B452(1999)194
    a = [0, 0.0485, 0.5470, 2.0621, -0.3804, 0.5090, -0.0285]
    b = [0, 0.0481, 0.6114, -0.3509, -0.4611, 0.7172, -0.0317]
    c = [0, 0.0577, 0.4644, 1.8288, 12.3708, -43.1043, 41.7415]

    theta = 1 + 12 * q2 / (q2 + 1) * 0.125**2 / (0.125**2 + x**2)

    ra = a[1] / numpy.log(q2 / 0.04) * theta + a[2] / (q2**4 + a[3]**4)**0.25 * (1 + a[4] * x + a[5] * x**2) * x**a[6]
    rb = b[1] / numpy.log(q2 / 0.04) * theta + (b[2] / q2 + b[3] / (q2**2 + 0.3**2)) * (1 + b[4] * x + b[5] * x**2) * x**b[6]
    q2thr = c[4] * x + c[5] * x**2 + c[6] * x**3
    rc = c[1] / numpy.log(q2 / 0.04) * theta + c[2] / ((q2 - q2thr)**2 + c[3]**2)**0.5

    result = (ra + rb + rc) / 3
    error = 0.0078 - 0.013 * x + (0.070 - 0.39 * x + 0.7 * x**2) / (1.7 + q2)

    return result, error


def f1p_lhapdf(x, q2):
    rr, err = r(x, q2)
    f2, ef2 = f2p_lhapdf(x, q2)

    gamma2 = 4 * _m_p**2 * x**2 / q2

    result = f2 * (1 + gamma2) / (2 * x * (1 + rr))
    error = numpy.sqrt((ef2 * (1 + gamma2) / (2 * x * (1 + rr)))**2 + (f2 * (1 + gamma2) / (2 * x * (1 + rr)**2) * err)**2)

    return result, error


def f2p_lhapdf(x, q2, select='CT14nnlo'):
    pdf_set = lhapdf.getPDFSet(select)
    pdfs = pdf_set.mkPDFs()

    d, ed = numpy.empty_like(x), numpy.empty_like(x)
    dbar, edbar = numpy.empty_like(x), numpy.empty_like(x)
    u, eu = numpy.empty_like(x), numpy.empty_like(x)
    ubar, eubar = numpy.empty_like(x), numpy.empty_like(x)
    s, es = numpy.empty_like(x), numpy.empty_like(x)

    for ix, xx in enumerate(x):
        d[ix], ed[ix] = _parton_lhapdf(1, xx, q2, pdf_set, pdfs)
        dbar[ix], edbar[ix] = _parton_lhapdf(-1, xx, q2, pdf_set, pdfs)
        u[ix], eu[ix] = _parton_lhapdf(2, xx, q2, pdf_set, pdfs)
        ubar[ix], eubar[ix] = _parton_lhapdf(-2, xx, q2, pdf_set, pdfs)
        s[ix], es[ix] = _parton_lhapdf(3, xx, q2, pdf_set, pdfs)

    result = ((2 / 3)**2 * (u + ubar) + (1 / 3)**2 * (d + dbar) + (1 / 3)**2 * 2 * s)
    error = numpy.sqrt((2 / 3)**4 * (eu**2 + eubar**2) + (1 / 3)**4 * (ed**2 + edbar**2) + (1 / 3)**4 * 4 * es**2)

    return result, error


def f1_rr(x, q2, f2_func):
    rr, _ = r(x, q2)
    return f2_func(x, q2) * (1 + 4 * _m_p**2 * x**2 / q2) / (2 * x * (1 + rr))


def f1p_slac(x, q2):
    return f1_rr(x, q2, f2p_slac)


def f1n_slac(x, q2):
    return f1_rr(x, q2, f2n_slac)


def f2p_slac(x, q2):
    # NMC
    # Phys. Lett. B364(1995)107
    a = [0, -0.02778, 2.926, 1.0362, -1.840, 8.123, -13.074, 6.215]
    b = [0, 0.285, -2.694, 0.0188, 0.0274]
    c = [0, -1.413, 9.366, -37.79, 47.10]

    ax = x**a[1] * (1 - x)**a[2] * (a[3] + a[4] * (1 - x) + a[5] * (1 - x)**2 + a[6] * (1 - x)**3 + a[7] * (1 - x)**4)
    bx = b[1] + b[2] * x + b[3] / (x + b[4])
    cx = c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4

    gamma2 = 0.25**2

    return ax * (numpy.log(q2 / gamma2) / numpy.log(20 / gamma2))**bx * (1 + cx / q2)


def f2d_slac(x, q2):
    # NMC
    # Phys. Lett. B364(1995)107
    a = [0, -0.04858, 2.863, 0.8367, -2.532, 9.145, -12.504, 5.473]
    b = [0, -0.008, -2.227, 0.0551, 0.0570]
    c = [0, -1.509, 8.553, -31.20, 39.98]

    ax = x**a[1] * (1 - x)**a[2] * (a[3] + a[4] * (1 - x) + a[5] * (1 - x)**2 + a[6] * (1 - x)**3 + a[7] * (1 - x)**4)
    bx = b[1] + b[2] * x + b[3] / (x + b[4])
    cx = c[1] * x + c[2] * x**2 + c[3] * x**3 + c[4] * x**4

    gamma2 = 0.25**2

    return ax * (numpy.log(q2 / gamma2) / numpy.log(20 / gamma2))**bx * (1 + cx / q2)


def f2n_slac(x, q2):
    # approximately
    return 2.*f2d_slac(x, q2) - f2p_slac(x, q2)


def g1p_slac(x, q2):
    # SLAC E155
    # Phys. Lett. B493(2000)19, Eq.(5)
    return x**0.700 * (0.817 + 1.014 * x - 1.489 * x**2) * (1 - 0.04 / q2) * f1p_slac(x, q2)


def g1n_slac(x, q2):
    # SLAC E155
    # Phys. Lett. B493(2000)19, Eq.(5)
    return x**-0.335 * (-0.013 - 0.330 * x + 0.761 * x**2) * (1 + 0.13 / q2) * f1n_slac(x, q2)


def g2ww_scalar(x, q2, g1_func):
    if x > 0:
        result = -g1_func(x, q2) + integrate.quad(lambda y: g1_func(y, q2) / y, x, 1)[0]
    else:
        result = numpy.inf
    return result


def g2ww(x, q2, g1_func):
    if numpy.isscalar(x) and numpy.isscalar(q2):
        return g2ww_scalar(x, q2, g1_func)
    elif numpy.isscalar(q2):
        return numpy.array([g2ww_scalar(xx, q2, g1_func) for xx in x])
    else:
        return numpy.array([g2ww_scalar(xx, qq2, g1_func) for xx, qq2 in zip(x, q2)])


def g2p_slac(x, q2):
    return g2ww(x, q2, g1p_slac)


def g2n_slac(x, q2):
    return g2ww(x, q2, g1n_slac)


def f1p(x, q2, model='slac', **kwargs):
    f1p_func = {
        'lhapdf': f1p_lhapdf,
        'slac': f1p_slac,
    }.get(model, None)

    return f1p_func(x, q2, **kwargs)


def f2p(x, q2, model='slac', **kwargs):
    f2p_func = {
        'lhapdf': f2p_lhapdf,
        'slac': f2p_slac,
    }.get(model, None)

    return f2p_func(x, q2, **kwargs)


def g1p(x, q2, model='slac', **kwargs):
    g1p_func = {
        'slac': g1p_slac,
    }.get(model, None)

    return g1p_func(x, q2, **kwargs)


def g2p(x, q2, model='slac', **kwargs):
    g2p_func = {
        'slac': g2p_slac,
    }.get(model, None)

    return g2p_func(x, q2, **kwargs)


def f1n(x, q2, model='slac', **kwargs):
    f1n_func = {
        'slac': f1n_slac,
    }.get(model, None)

    return f1n_func(x, q2, **kwargs)


def f2n(x, q2, model='slac', **kwargs):
    f2n_func = {
        'slac': f2n_slac,
    }.get(model, None)

    return f2n_func(x, q2, **kwargs)


def g1n(x, q2, model='slac', **kwargs):
    g1n_func = {
        'slac': g1n_slac,
    }.get(model, None)

    return g1n_func(x, q2, **kwargs)


def g2n(x, q2, model='slac', **kwargs):
    g2n_func = {
        'slac': g2n_slac,
    }.get(model, None)

    return g2n_func(x, q2, **kwargs)

