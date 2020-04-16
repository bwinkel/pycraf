#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

# from functools import partial, lru_cache
from astropy import units as apu
import numpy as np
from .cyantenna import ras_pattern_cython
from .. import conversions as cnv
from .. import utils


__all__ = [
    'ras_pattern',
    ]


@utils.ranged_quantity_input(
    phi=(-180, 180, apu.deg),
    diameter=(0.1, 1000., apu.m),
    wavelength=(0.001, 2, apu.m),
    eta_a=(0, None, cnv.dimless),
    strip_input_units=True, output_unit=cnv.dBi
    )
def ras_pattern(
        phi, diameter, wavelength, eta_a=100. * apu.percent, do_bessel=False
        ):
    '''
    Antenna gain as a function of angular distance after `ITU-R Rec RA.1631
    <https://www.itu.int/rec/R-REC-RA.1631-0-200305-I/en>`_.

    Parameters
    ----------
    phi : `~astropy.units.Quantity`
        Angular distance from looking direction [deg]
    diameter : `~astropy.units.Quantity`
        Antenna diameter [m]
    wavelength : `~astropy.units.Quantity`
        Observing wavelength [m]
    eta_a : `~astropy.units.Quantity`
        Antenna efficiency (default: 100%)
    do_bessel : bool, optional
        If set to True, use Bessel function approximation for inner 1 deg
        of the pattern (see RA.1631 for details). (default: False)

    Returns
    -------
    gain : `~astropy.units.Quantity`
        Antenna gain [dBi]

    Notes
    -----
    - See `ITU-R Rec. RA.1631-0
      <https://www.itu.int/rec/R-REC-RA.1631-0-200305-I/en>`_ for further
      explanations and applicability of this model.
    '''

    phi = np.abs(phi)
    # eta_a = eta_a / 100.

    # the following are independent on phi, no need to compute after broadcast
    d_wlen = diameter / wavelength

    # Note, we use the version that accounts for antenna efficiency
    # see ITU radio astronomy handbook, page 50
    # gmax = 20 * np.log10(np.pi * d_wlen)
    gmax = 10 * np.log10(eta_a * (np.pi * d_wlen) ** 2)

    g1 = -1. + 15. * np.log10(d_wlen)
    phi_m = 20. / d_wlen * np.sqrt(gmax - g1)
    phi_r = 15.85 * d_wlen ** -0.6

    # # note automatic broadcasting should be possible
    # # _tmp = np.broadcast(phi, _diam, _wlen)
    # (
    #     phi, d_wlen, gmax, g1, phi_m, phi_r,
    #     ) = np.broadcast_arrays(
    #     phi, d_wlen, gmax, g1, phi_m, phi_r,
    #     )
    # gain = np.zeros(phi.shape, np.float64)

    # # case 1:
    # mask = (0 <= phi) & (phi < phi_m)
    # gain[mask] = gmax[mask] - 2.5e-3 * (d_wlen[mask] * phi[mask]) ** 2

    # # case 2:
    # mask = (phi_m <= phi) & (phi < phi_r)
    # gain[mask] = g1[mask]

    # # case 3:
    # mask = (phi_r <= phi) & (phi < 10.)
    # gain[mask] = 29. - 25. * np.log10(phi[mask])

    # # case 4:
    # mask = (10. <= phi) & (phi < 34.1)
    # gain[mask] = 34. - 30. * np.log10(phi[mask])

    # # case 5:
    # mask = (34.1 <= phi) & (phi < 80.)
    # gain[mask] = -12.

    # # case 6:
    # mask = (80. <= phi) & (phi < 120.)
    # gain[mask] = -7.

    # # case 7:
    # mask = (120. <= phi) & (phi <= 180.)
    # gain[mask] = -12.

    gain = ras_pattern_cython(phi, d_wlen, gmax, g1, phi_m, phi_r)

    if do_bessel:

        # why not the spherical bessel function of first kind (spherical_jn)?
        from scipy.special import j1

        phi, d_wlen, gmax = np.broadcast_arrays(phi, d_wlen, gmax)

        phi_0 = 69.88 / d_wlen
        x_pi = np.radians(np.pi / 2. * d_wlen * phi)

        # case 1:
        mask = (1.e-32 < phi) & (phi < phi_0)
        tmp_x = x_pi[mask]
        gain[mask] = gmax[mask] + 20 * np.log10(j1(2 * tmp_x) / tmp_x)

        # case 2:
        mask = (phi_0 <= phi) & (phi < 1.)
        B_sqrt = 10 ** 1.6 * np.radians(np.pi * d_wlen[mask] / 2.)
        tmp_x = x_pi[mask]
        gain[mask] = 20 * np.log10(
            B_sqrt * np.cos(2 * tmp_x - 0.75 * np.pi + 0.0953) / tmp_x
            )

    return gain


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
