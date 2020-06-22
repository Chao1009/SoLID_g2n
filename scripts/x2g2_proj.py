#!/usr/bin/env python

# For python 2-3 compatibility
from __future__ import division, print_function

import sys
import pickle

sys.path.append('/home/cpeng/Projects')

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
import argparse
from scipy import constants

from pysolidg2p import asymmetry, cross_section, experiments, sim_reader, structure_f, tools

_m_p = constants.value('proton mass energy equivalent in MeV') * 1e-3


def auto_split(n):
    test = int(np.sqrt(n))
    if np.square(test) >= n:
        return test, test
    elif (test + 1)*test >= n:
        return test + 1, test
    else:
        return test + 1, test + 1

#
# input parameters
#
parser = argparse.ArgumentParser('x2g2 projections')
parser.add_argument('yield_file', help='a text file of yield')
parser.add_argument('-o', '--output-file', default='results_{energy}.csv',
                    dest='output', help='path to save the pickle file')
parser.add_argument('-p', dest='bpass', default=5, type=int, help='beam pass')
args = parser.parse_args()

yield_limit = 100
beam_pol = 0.85
target_pol = 0.55
dilution_factor = 0.15
prescale = 10.0
xs_syst = 0.1
# because polarization plane and scattering plane is not in coincidence
# full 2pi coverage can only be used with cos_phi weighting
phi_wt = 1./np.pi
luminosity = 1e6  # /ub/s
syst_asym = 0.001
syst_xs = 0.10

x_knots = np.linspace(0.05, 0.95, 10)
x_points = np.arange(0.1, 1.0, step=0.1)

if args.bpass == 5:
    e = 11
    q2_knots = np.linspace(0.9, 10.9, 51)
    q2_points = np.arange(2.0, 11.0, step=1.0)
    blum = luminosity*1152*3600
else:
    e = 8.8
    q2_knots = np.linspace(0.8, 8.8, 41)
    q2_points = np.arange(1.5, 7.0, step=1.0)
    blum = luminosity*1152*3600

#
# main program
#

sim = pd.read_csv(args.yield_file, header=None, sep='\s+')
sim.columns = ['x', 'Q2', 'yields']
result = pd.DataFrame(columns=[
    'x', 'xbw', 'Q2', 'Q2bw',
    'yields', 'asymL', 'asymT', 'stat_pol',
    'dxsL', 'dxsT', 'edxsL', 'edxsT',
    'g1', 'g2','eg1', 'eg2', 'sg1', 'sg2'
])

# rebin simulation results with the given binning as [a, b)
q2_bins = np.vstack([q2_knots[:-1], q2_knots[1:]]).T
x_bins = np.vstack([x_knots[:-1], x_knots[1:]]).T

for qbin in q2_bins:
    for xbin in x_bins:
        # group yields
        x, q2 = np.average(xbin), np.average(qbin)
        mask = (sim['x'] < xbin[1]) & (sim['x'] >= xbin[0])
        mask &= (sim['Q2'] < qbin[1]) & (sim['Q2'] >= qbin[0])
        yields = sim.loc[mask, 'yields'].sum()

        ep_min = q2 / (4 * e)
        x_min = q2 / (2 * _m_p * (e - ep_min))
        select = (x > x_min) & (yields > yield_limit)
        if not select:
            continue

        # corrected yields
        stat_unpl = np.sqrt(prescale) / np.sqrt(yields)
        stat_pol = np.sqrt(prescale) / np.sqrt(yields * beam_pol * target_pol * (1. - dilution_factor) * phi_wt)

        g1, g2 = structure_f.g1n(x, q2), structure_f.g2n(x, q2)
        # cross section from generator
        # xs0 = yields/blum
        xs0 = cross_section.xsp(e, x, q2)
        # directly propagate errors to g1, g2
        dxsL, dxsT = tools.g1g2_to_dxs(e, x, q2, g1, g2)
        asymL, asymT = dxsL/2./xs0, dxsT/2./xs0
        edxsL = 2. * stat_pol * np.sqrt(1152/504) * xs0
        edxsT = 2. * stat_pol * xs0
        _, _, eg1, eg2 = tools.dxs_to_g1g2(e, x, q2, dxsL, dxsT, edxsL, edxsT)

        sg1, sg2, ssg1, ssg2 = tools.dxs_to_g1g2(e, x, q2, syst_asym*2*xs0, syst_asym*2*xs0, dxsL*syst_xs, dxsT*syst_xs)

        result.loc[len(result)] = (
            x, (xbin[1] - xbin[0])/2., q2, (qbin[1] - qbin[0])/2.,
            yields/prescale, asymL, asymT, stat_pol,
            dxsL, dxsT, edxsL, edxsT,
            g1, g2, eg1, eg2, np.sqrt(sg1*sg1 + ssg1*ssg1), np.sqrt(sg2*sg2 + ssg2*ssg2)
        )

result.to_csv(args.output.format(energy=e), index=False)
print(result)


def getm(d):
    # return d['eg2']/np.max(np.abs(d['g2'])) < 0.5
    return [True]*len(d)


def getdata(d, q2, xp=None):
    q2c, q2bw = d[['Q2', 'Q2bw']].values.T
    qmask = (q2c - q2bw <= q2) & (q2c + q2bw > q2)

    if xp is None:
        res = d[qmask].sort_values('x')
        return res['x'].values, res

    xc, xbw = d[qmask][['x', 'xbw']].values.T
    xmask = np.full((len(xc), ), False, dtype=bool)
    for x in xp:
        xmask |= (xc - xbw <= x) & (xc + xbw > x)
    res = d[qmask][xmask].sort_values('x')
    return res['x'].values, res


# plots
bprops = dict(boxstyle='round', facecolor='white', alpha=0.5)
gridspec = {'left': 0.12, 'bottom': 0.10, 'right': 0.95, 'top': 0.90, 'hspace': 0., 'wspace': 0.}

nrows, ncols = auto_split(len(q2_points))
fs = (6*ncols, 4*nrows)
with PdfPages('x2g2_proj_{}GeV.pdf'.format(e)) as pdf:
    fig, axs = plt.subplots(nrows, ncols, figsize=fs, sharex='all', sharey='row', gridspec_kw=gridspec)
    axs[0][0].xaxis.set_major_locator(ticker.IndexLocator(base=0.2, offset=0.))
    for ax in axs.flat:
        # axis setup
        ax.set_xlim(0, 1)
        ax.tick_params(axis='both', direction='in', labelsize=20)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))

    lines, labels = [], []
    for ax, q2 in zip(axs.flat, q2_points):
        x, res = getdata(result, q2, x_points)
        qc, qbw = res[['Q2', 'Q2bw']].mean().values
        # model
        ep_min = q2 / (4 * e)
        x_min = q2 / (2 * _m_p * (e - ep_min))
        x_model = np.linspace(0.1, 0.9, 201)
        x_model = x_model[x_model > x_min]
        xm, xv = x_model, x_model**2 * structure_f.g2n(x_model, q2)
        ax.plot(xm, xv, 'k--', linewidth=0.75)
        # syst
        # all x points
        x0, res0 = getdata(result, q2)
        g2, sg2 = res0[['g2', 'sg2']].values.T
        ax.fill_between(x0, 0, x0**2*np.abs(sg2), color='black', alpha=0.2, hatch=r'////', label='Syst. Estimate')
        # data
        g2, eg2 = res[['g2', 'eg2']].values.T
        ax.errorbar(x, x**2*g2, x**2*eg2, fmt='k.', capsize=3, label='Prescale = {:d}'.format(int(prescale)))
        ax.errorbar(x + 0.02, x**2*g2, x**2*eg2/np.sqrt(prescale/2), fmt='r.',
                    label='Prescale = 2', capsize=3, fillstyle='none')
        ax.axhline(y=0, linestyle=':', linewidth=0.75, color='k')
        ax.text(0.10, 0.15, r'${:.1f} < Q^2 < {:.1f}$'.format(qc - qbw, qc + qbw),
                transform=ax.transAxes, fontsize=18, verticalalignment='top', bbox=bprops)
        lines, labels = ax.get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', ncol=5, fontsize=22)
    fig.text(0.5, 0.05, r'$x_{Bjorken}$', ha='center', fontsize=20)
    fig.text(0.05, 0.5, r'$x^2g_2$', va='center', rotation=90, fontsize=20)
    pdf.savefig(fig, bbox_inches='tight')


with PdfPages('x2g2_proj_{}GeV_details.pdf'.format(e)) as pdf:
    pdf.savefig(fig, bbox_inches='tight')
    # yields
    fig, axs = plt.subplots(nrows, ncols, figsize=fs, sharex='all', sharey='row', gridspec_kw=gridspec)
    axs[0][0].xaxis.set_major_locator(ticker.IndexLocator(base=0.2, offset=0.))
    for ax in axs.flat:
        # axis setup
        ax.set_yscale('log')
        # ax.set_ylim(1e2, 3e11)
        ax.set_xlim(0., 1.)
        ax.tick_params(axis='both', direction='in', labelsize=20)
        ax.tick_params(axis='y', which='minor', bottom=False)

    lines, labels = [], []
    ymin, ymax = result['yields'].max(), result['yields'].min()
    for i, (ax, q2) in enumerate(zip(axs.flat, q2_points)):
        # locate q2 bin
        x, res = getdata(result, q2, x_points)
        qc, qbw = res[['Q2', 'Q2bw']].mean().values

        y, err = res[['yields', 'edxsL']].values.T
        ax.errorbar(x, y, np.sqrt(y), fmt='k.', capsize=3, label='total yields')
        ax.axhline(y=0, linestyle=':', linewidth=0.75, color='k')
        ax.text(0.10, 0.15, r'${:.1f} < Q^2 < {:.1f}$'.format(qc - qbw, qc + qbw),
                transform=ax.transAxes, fontsize=18, verticalalignment='top', bbox=bprops)
        ymin = min(np.min(y), ymin)
        ymax = max(np.max(y), ymax)
        ax.set_ylim(np.exp(int(np.log10(ymin))*np.log(10)), 3.*np.exp((int(np.log10(ymax)) + 2)*np.log(10)))
        if (i % ncols) == 0:
            ymin, ymax = result['yields'].max(), result['yields'].min()

        lines, labels = ax.get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', ncol=5, fontsize=22)
    fig.text(0.5, 0.05, r'$x_{Bjorken}$', ha='center', fontsize=20)
    fig.text(0.05, 0.5, r'Counts', va='center', rotation=90, fontsize=20)
    pdf.savefig(fig, bbox_inches='tight')

    # asymmetries
    fig, axs = plt.subplots(nrows, ncols, figsize=fs, sharex='all', sharey='all', gridspec_kw=gridspec)
    axs[0][0].xaxis.set_major_locator(ticker.IndexLocator(base=0.2, offset=0.))
    for ax in axs.flat:
        # axis setup
        ax.set_xlim(0., 1.)
        ax.set_ylim(-0.05, 0.05)
        ax.tick_params(axis='both', direction='in', labelsize=20)

    lines, labels = [], []
    for ax, q2 in zip(axs.flat, q2_points):
        x, res = getdata(result, q2, x_points)
        qc, qbw = res[['Q2', 'Q2bw']].mean().values
        y, err = res[['asymL', 'stat_pol']].values.T
        ax.errorbar(x, np.zeros(shape=x.shape) + 0.02, err, capsize=3, fmt='r.', label='Longitudinal')
        # ax.errorbar(x, y, err, fmt='r.', label='Longitudinal')
        y, err = res[['asymT', 'stat_pol']].values.T
        ax.errorbar(x, np.zeros(shape=x.shape) - 0.02, err, capsize=3, fmt='b.', label='Transverse')
        # ax.errorbar(x, y, err, fmt='b.', label='Transverse')

        ax.axhline(y=0, linestyle=':', linewidth=0.75, color='k')
        ax.text(0.10, 0.92, r'${:.1f} < Q^2 < {:.1f}$'.format(qc - qbw, qc + qbw),
                transform=ax.transAxes, fontsize=18, verticalalignment='top', bbox=bprops)
        lines, labels = ax.get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', ncol=5, fontsize=22, title='Asymmetry Stat.', title_fontsize=22)
    fig.text(0.5, 0.05, r'$x_{Bjorken}$', ha='center', fontsize=20)
    fig.text(0.05, 0.5, r'$\delta Asym$', va='center', rotation=90, fontsize=20)
    pdf.savefig(fig, bbox_inches='tight')

    # delta_xs
    fig, axs = plt.subplots(nrows, ncols, figsize=fs, sharex='all', sharey='row', gridspec_kw=gridspec)
    axs[0][0].xaxis.set_major_locator(ticker.IndexLocator(base=0.2, offset=0.))
    for ax in axs.flat:
        # axis setup
        ax.set_xlim(0., 1.)
        ax.tick_params(axis='both', direction='in', labelsize=20)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 1))

    lines, labels = [], []
    for ax, q2 in zip(axs.flat, q2_points):
        x, res = getdata(result, q2, x_points)
        qc, qbw = res[['Q2', 'Q2bw']].mean().values

        y, err = res[['dxsL', 'edxsL']].values.T
        ax.errorbar(x, y, np.abs(err), fmt='.', label='Longitudinal', capsize=3)
        y, err = res[['dxsT', 'edxsT']].values.T
        ax.errorbar(x, y, np.abs(err), fmt='.', label='Transverse', capsize=3)

        ax.axhline(y=0, linestyle=':', linewidth=0.75, color='k')

        ax.text(0.10, 0.92, r'${:.1f} < Q^2 < {:.1f}$'.format(qc - qbw, qc + qbw),
                transform=ax.transAxes, fontsize=18, verticalalignment='top', bbox=bprops)
        lines, labels = ax.get_legend_handles_labels()
    fig.legend(lines, labels, loc='upper center', ncol=5, fontsize=22, title='$\Delta \sigma$', title_fontsize=22)
    fig.text(0.5, 0.05, r'$x_{Bjorken}$', ha='center', fontsize=20)
    fig.text(0.05, 0.5, r'$\Delta\sigma (\mu b \cdot GeV^{-1} \cdot sr^{-1})$', va='center', rotation=90, fontsize=20)
    pdf.savefig(fig, bbox_inches='tight')


