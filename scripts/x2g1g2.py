#!/usr/bin/env python

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy

from pysolidg2p.structure_f import g1p, g2p

x = numpy.linspace(0, 1, 101)

with PdfPages('x2g1g2.pdf') as pdf:
    plt.figure()
    ax = plt.gca()
    ax.axhline(y=0, linestyle='--', linewidth=1, color='k')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$x^2g_2$')
    plt.xlim(0, 1)
    plt.ylim(-0.03, 0.01)
    plt.plot(x, x**2 * g2p(x, 3), 'k-')  # q2 = 3
    plt.plot(x, x**2 * g2p(x, 4), 'r-')  # q2 = 4
    plt.plot(x, x**2 * g2p(x, 5), 'g-')  # q2 = 5
    plt.plot(x, x**2 * g2p(x, 6), 'b-')  # q2 = 6
    pdf.savefig(bbox_inches='tight')
    plt.close()

    fig = plt.figure()
    ax = plt.gca()
    ax.axhline(y=0, linestyle='--', linewidth=1, color='k')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$x^2g_1$')
    plt.xlim(0, 1)
    plt.ylim(-0.005, 0.035)
    plt.plot(x, x**2 * g1p(x, 3), 'k-')  # q2 = 3
    plt.plot(x, x**2 * g1p(x, 4), 'r-')  # q2 = 4
    plt.plot(x, x**2 * g1p(x, 5), 'g-')  # q2 = 5
    plt.plot(x, x**2 * g1p(x, 6), 'b-')  # q2 = 6
    pdf.savefig(bbox_inches='tight')
    plt.close()
