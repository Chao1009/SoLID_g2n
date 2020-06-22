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

yield_limit = 1000
beam_pol = 0.85
target_pol = 0.55
dilution_factor = 0.15
xs_syst = 0.1
prescale = 1.0
x_range = (0.05, 0.95)
yield_limit = 10
# because polarization plane and scattering plane is not in coincidence
# full 2pi coverage can only be used with cos_phi weighting
phi_wt = 1/np.pi

if args.bpass == 5:
    e = 11
else:
    e = 8.8

# main program

sim = pd.read_csv(args.yield_file, header=None, sep='\s+')
sim.columns = ['x', 'Q2', 'yields']
syst_asym = 0.001
syst_xs = 0.10

x, q2, yields = sim[['x', 'Q2', 'yields']].values.T
ep_min = q2 / (4 * e)
x_min = q2 / (2 * _m_p * (e - ep_min))
mask = (sim['x'] > x_min) & (sim['yields'] > yield_limit) & (sim['x'] >= x_range[0]) & (sim['x'] <= x_range[1])
sim = sim[mask]

# corrected yields
x, q2, yields = sim[['x', 'Q2', 'yields']].values.T

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

sim.loc[:, 'g1'] = g1
sim.loc[:, 'g2'] = g2
sim.loc[:, 'eg1'] = eg1
sim.loc[:, 'eg2'] = eg2
sim.loc[:, 'sg1'] = np.sqrt(sg1*sg1 + ssg1*ssg1)
sim.loc[:, 'sg2'] = np.sqrt(sg2*sg2 + ssg2*ssg2)

sim.to_csv('yields_{}.csv'.format(e), index=False)

