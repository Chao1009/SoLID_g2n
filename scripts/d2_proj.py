#!/usr/bin/env python

# For python 2-3 compatibility
from __future__ import division, print_function

import pickle

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import constants
from collections import OrderedDict
from pysolidg2p import asymmetry, cross_section, experiments, sim_reader, structure_f, tools
import argparse


_m_p = constants.value('proton mass energy equivalent in MeV') * 1e-3
unmeasured_syst = 0.15
xmin_thres = 0.42
xmax_thres = 0.85
prescale = 10

start = -1
merge = 4


def get_d2(path):
    res = pd.DataFrame(columns=['d2', 'stat', 'syst', 'xmin', 'xmax', 'mratio'])
    df = pd.read_csv(path)
    q2s = df['Q2'].unique()
    for q2 in q2s:
        data = df[np.isclose(df['Q2'], q2, atol=1e-6)].sort_values('x', ascending=True)
        x, g1, g2, eg1, eg2, sg1, sg2 = data[['x', 'g1', 'g2', 'eg1', 'eg2', 'sg1', 'sg2']].values.T
        # data integral
        d2, ed2, sd2 = tools.g1g2_to_d2(x, g1, g2, eg1, eg2, sg1, sg2)

        # fill unmeasured
        xl = np.linspace(1e-3, x[0], 50)
        d2l, _, _ = tools.g1g2_to_d2(xl, structure_f.g1n(xl, q2), structure_f.g2n(xl, q2))

        xh = np.linspace(x[-1], 1, 50)
        d2h, _, _ = tools.g1g2_to_d2(xh, structure_f.g1n(xh, q2), structure_f.g2n(xh, q2))

        xm = np.linspace(x[0], x[-1], 50)
        d2m, _, _ = tools.g1g2_to_d2(xm, structure_f.g1n(xm, q2), structure_f.g2n(xm, q2))

        measured_ratio = x[-1]**3 - x[0]**3

        res.loc[q2] = d2 + d2l + d2h, ed2, sd2, x[0], x[-1], measured_ratio
    return res


projections = [
#    ('11 GeV Projections', get_d2('results_11.csv'), {'color': 'r'}),
#    ('8 GeV Projections', get_d2('results_8.8.csv'), {'color': 'b'}),
    ('11 GeV Projections', get_d2('yields_11.csv'), {'color': 'r'}, np.linspace(0.9, 10.9, 51), 2.5),
    ('8 GeV Projections', get_d2('yields_8.8.csv'), {'color': 'b'}, np.linspace(0.8, 8.8, 41), 1.8),
]

other_data = OrderedDict([
    ('SLAC E155x', ([5], [0.0079], [0.0048], {'fmt': 'ko'})),
    ('JLab RSS + pQCD', ([5 - 0.1], [0.0031], [0.0038], {'fmt': 'kx'})),
    ('E01-012', ([2.4], [0.00034], [0.00152], {'fmt': 'kd'})),
    ('E99-117 + E155x', ([5.0 - 0.18], [0.0062], [0.0028], {'fmt': 'k>'})),
    ('E06-014', ([3.21, 4.32], [-0.00421, -0.00035], [0.00169, 0.00159], {'fmt': 'k<'})),
    ('Lattice QCD', ([5 + 0.1], [-0.001], [0.003], {'fmt': 'ks'})),
    ('E12-06-121 Projections', ([4, 5, 6], [-6e-3, -6e-3, -6e-3], [7.07e-4, 7.07e-4, 7.07e-4], {'fmt': 'ko', 'fillstyle': 'full'}))
])

common_kw = {'elinewidth': 1, 'capsize': 2, 'fillstyle': 'none'}


def sum_quad(x):
    return np.sqrt(np.average(np.square(x)))


with PdfPages('d2_proj.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.axhline(y=0, linestyle='--', linewidth=0.75, color='k')
    for i, (label, data, kw, q2_knots, q2_min) in enumerate(projections):
        vdata = data.copy()
        vdata.loc[:, 'bin'] = np.digitize(vdata.index.values, q2_knots)
        group = vdata.groupby('bin')
        res = group[['d2', 'syst', 'xmin', 'xmax', 'mratio']].mean()
        res.loc[:, 'stat'] = group['stat'].apply(sum_quad)
        res.loc[:, 'Q2'] = ((q2_knots[1:] + q2_knots[:-1])/2.)[res.index.astype(int).values - 1]
        res = res.iloc[1:]
        rmask = (res['xmax'] >= xmax_thres) & (res['xmin'] <= xmin_thres)
        print(res)
        q2, stat, syst, mratio = res.loc[rmask, ['Q2', 'stat', 'syst', 'mratio']].iloc[start::-merge].values.T
        # stat = np.average(stat)
        # np.pi is cos_phi weighting in dxsT
        stat = np.average(np.sqrt(stat**2))*np.sqrt(1./merge)
        syst = np.average(syst)
        print(label, stat, syst)
        ax.errorbar(q2, np.zeros(shape=(len(q2), )) - 0.003*i, np.sqrt(np.square(stat) + np.square(syst)),
            fmt='.', elinewidth=1, capsize=1.5, label=label, **kw)

    for label, vals in other_data.items():
        kw = dict()
        kw.update(common_kw)
        kw.update(vals[3])
        ax.errorbar(vals[0], vals[1], vals[2], label=label, **kw)
    ax.set_xlabel(r'$Q^2$ (GeV $^2$)')
    ax.set_ylabel(r'$d_2$')
    ax.set_xlim(1, 10)
    ax.set_ylim(-0.010, 0.015)
    ax.legend()
    pdf.savefig(bbox_inches='tight')


with PdfPages('d2_proj_p2.pdf') as pdf:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.axhline(y=0, linestyle='--', linewidth=0.75, color='k')
    for i, (label, data, kw, q2_knots, q2_min) in enumerate(projections):
        vdata = data.copy()
        vdata.loc[:, 'bin'] = np.digitize(vdata.index.values, q2_knots)
        group = vdata.groupby('bin')
        res = group[['d2', 'syst', 'xmin', 'xmax', 'mratio']].mean()
        res.loc[:, 'stat'] = group['stat'].apply(sum_quad)/np.sqrt(prescale/2.0)
        res.loc[:, 'Q2'] = ((q2_knots[1:] + q2_knots[:-1])/2.)[res.index.astype(int).values - 1]
        res = res.iloc[1:]
        rmask = (res['xmax'] >= xmax_thres) & (res['xmin'] <= xmin_thres)
        q2, stat, syst, mratio = res.loc[rmask, ['Q2', 'stat', 'syst', 'mratio']].iloc[start::-merge].values.T
        stat = np.average(np.sqrt(stat**2))*np.sqrt(1./merge)
        syst = np.average(syst)
        ax.errorbar(q2, np.zeros(shape=(len(q2), )) - 0.003*i, np.sqrt(np.square(stat/mratio) + np.square(syst)),
            fmt='.', elinewidth=1, capsize=1.5, label=label, **kw)

    for label, vals in other_data.items():
        kw = dict()
        kw.update(common_kw)
        kw.update(vals[3])
        ax.errorbar(vals[0], vals[1], vals[2], label=label, **kw)
    ax.set_xlabel(r'$Q^2$ (GeV $^2$)')
    ax.set_ylabel(r'$d_2$')
    ax.set_xlim(1, 10)
    ax.set_ylim(-0.010, 0.015)
    ax.legend()
    pdf.savefig(bbox_inches='tight')

