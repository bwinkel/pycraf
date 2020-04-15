#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

# from functools import partial, lru_cache
from astropy import units as apu
import numpy as np
from .cyantenna import fl_pattern_cython
from .. import conversions as cnv
from .. import utils


__all__ = [
    'fl_pattern',
    'fl_G_max_from_hpbw', 'fl_G_max_from_size', 'fl_hpbw_from_size',
    ]


def _fl_pattern_2_1(phi, diameter_over_wavelength, G_max):
    '''
    Antenna gain as a function of angular distance after `ITU-R Rec F.699
    <https://www.itu.int/rec/R-REC-F.699-7-200604-I/en>`_ 2.1.

    Valid for 1 to 70 GHz and cases where D / wavelength > 100
    '''

    phi = np.abs(phi)

    # the following are independent on phi, no need to compute after broadcast
    d_wlen = diameter_over_wavelength

    g1 = 2. + 15. * np.log10(d_wlen)  # gain of first side-lobe
    phi_m = 20. / d_wlen * np.sqrt(G_max - g1)
    phi_r = 15.85 * d_wlen ** -0.6

    # note automatic broadcasting should be possible
    (
        phi, d_wlen, G_max, g1, phi_m, phi_r,
        ) = np.broadcast_arrays(
        phi, d_wlen, G_max, g1, phi_m, phi_r,
        )
    gain = np.zeros(phi.shape, np.float64)

    # case 1:
    mask = (0 <= phi) & (phi < phi_m)
    gain[mask] = G_max[mask] - 2.5e-3 * (d_wlen[mask] * phi[mask]) ** 2

    # case 2:
    mask = (phi_m <= phi) & (phi < phi_r)
    gain[mask] = g1[mask]

    # case 3:
    mask = (phi_r <= phi) & (phi < 48.)
    gain[mask] = 32. - 25. * np.log10(phi[mask])

    # case 4:
    mask = (48. <= phi) & (phi <= 180.)
    gain[mask] = -10.

    return gain


def _fl_pattern_2_2(phi, diameter_over_wavelength, G_max):
    '''
    Antenna gain as a function of angular distance after `ITU-R Rec F.699
    <https://www.itu.int/rec/R-REC-F.699-7-200604-I/en>`_ 2.2.

    Valid for 1 to 70 GHz and cases where D / wavelength <= 100
    '''

    phi = np.abs(phi)

    # the following are independent on phi, no need to compute after broadcast
    d_wlen = diameter_over_wavelength

    g1 = 2. + 15. * np.log10(d_wlen)  # gain of first side-lobe
    phi_m = 20. / d_wlen * np.sqrt(G_max - g1)
    phi_r = 15.85 * d_wlen ** -0.6

    # note automatic broadcasting should be possible
    (
        phi, d_wlen, G_max, g1, phi_m, phi_r,
        ) = np.broadcast_arrays(
        phi, d_wlen, G_max, g1, phi_m, phi_r,
        )
    gain = np.zeros(phi.shape, np.float64)

    # case 1:
    mask = (0 <= phi) & (phi < phi_m)
    gain[mask] = G_max[mask] - 2.5e-3 * (d_wlen[mask] * phi[mask]) ** 2

    # case 2:
    mask = (phi_m <= phi) & (phi < phi_r)
    gain[mask] = g1[mask]

    # case 3:
    mask = (phi_r <= phi) & (phi < 48.)
    gain[mask] = 52 - 10 * np.log10(d_wlen[mask]) - 25 * np.log10(phi[mask])

    # case 4:
    mask = (48. <= phi) & (phi <= 180.)
    gain[mask] = -10. - 10 * np.log10(d_wlen[mask])

    return gain


def _fl_pattern_2_3(phi, diameter_over_wavelength, G_max):
    '''
    Antenna gain as a function of angular distance after `ITU-R Rec F.699
    <https://www.itu.int/rec/R-REC-F.699-7-200604-I/en>`_ 2.3.

    Valid for 0.1 to 1 GHz and cases where D / wavelength > 0.63
    '''

    phi = np.abs(phi)

    # the following are independent on phi, no need to compute after broadcast
    d_wlen = diameter_over_wavelength

    g1 = 2. + 15. * np.log10(d_wlen)  # gain of first side-lobe
    phi_m = 20. / d_wlen * np.sqrt(G_max - g1)
    phi_t = 100. / d_wlen
    phi_s = 144.5 * d_wlen ** -0.2

    # note automatic broadcasting should be possible
    (
        phi, d_wlen, G_max, g1, phi_m, phi_t, phi_s,
        ) = np.broadcast_arrays(
        phi, d_wlen, G_max, g1, phi_m, phi_t, phi_s,
        )
    gain = np.full(phi.shape, np.nan, np.float64)

    # case 1:
    mask = (0 <= phi) & (phi < phi_m)
    gain[mask] = G_max[mask] - 2.5e-3 * (d_wlen[mask] * phi[mask]) ** 2

    # case 2:
    mask = (phi_m <= phi) & (phi < phi_t)
    gain[mask] = g1[mask]

    # case 3:
    mask = (phi_t <= phi) & (phi < phi_s)
    gain[mask] = 52 - 10 * np.log10(d_wlen[mask]) - 25 * np.log10(phi[mask])

    # case 4:
    mask = (phi_s <= phi) & (phi <= 180.)
    gain[mask] = -2. - 5 * np.log10(d_wlen[mask])

    return gain


@utils.ranged_quantity_input(
    phi=(-180, 180, apu.deg),
    diameter=(0.1, 1000., apu.m),
    wavelength=(0.001, 2, apu.m),
    G_max=(None, None, cnv.dBi),
    strip_input_units=True, output_unit=cnv.dBi
    )
def fl_pattern(
        phi, diameter, wavelength, G_max
        ):
    '''
    Antenna gain as a function of angular distance after `ITU-R Rec F.699
    <https://www.itu.int/rec/R-REC-F.699-7-200604-I/en>`_.

    Parameters
    ----------
    phi : `~astropy.units.Quantity`
        Angular distance[deg]
    diameter : `~astropy.units.Quantity`
        Antenna diameter [m]
    wavelength : `~astropy.units.Quantity`
        Observing wavelength [m]
    G_max : `~astropy.units.Quantity`
        Antenna maximum gain [dBi]

    Returns
    -------
    gain : `~astropy.units.Quantity`
        Antenna gain [dBi]

    Notes
    -----
    - If `G_max` is unknown, there is a helper function
      `pycraf.antenna.G_max_from_HPBW` that can be used to estimate its
      value.
    - See `ITU-R Rec F.699
      <https://www.itu.int/rec/R-REC-F.699-7-200604-I/en>`_ for further
      explanation and applicability of this model.
    '''

    return fl_pattern_cython(phi, diameter, wavelength, G_max)


def _fl_hpbw_from_size(diameter, wavelength):

    return 70. * wavelength / diameter


@utils.ranged_quantity_input(
    diameter=(0.1, 1000., apu.m),
    wavelength=(0.001, 2, apu.m),
    strip_input_units=True, output_unit=apu.deg
    )
def fl_hpbw_from_size(diameter, wavelength):
    '''
    Estimate HPBW from antenna diameter after `ITU-R Rec F.699
    <https://www.itu.int/rec/R-REC-F.699-7-200604-I/en>`_.

    Parameters
    ----------
    diameter : `~astropy.units.Quantity`
        Antenna diameter [m]
    wavelength : `~astropy.units.Quantity`
        Observing wavelength [m]

    Returns
    -------
    hpbw : `~astropy.units.Quantity`
        Antenna HPBW (3-dB point) [deg]

    Notes
    -----
    See `ITU-R Rec F.699
    <https://www.itu.int/rec/R-REC-F.699-7-200604-I/en>`_ for further
    explanation and applicability of this model.
    '''

    return _fl_hpbw_from_size(diameter, wavelength)


def _fl_G_max_from_size(diameter, wavelength):

    return 20. * np.log10(diameter / wavelength) + 7.7


@utils.ranged_quantity_input(
    diameter=(0.1, 1000., apu.m),
    wavelength=(0.001, 2, apu.m),
    strip_input_units=True, output_unit=cnv.dBi
    )
def fl_G_max_from_size(diameter, wavelength):
    '''
    Estimate maximum gain from antenna diameter after `ITU-R Rec F.699
    <https://www.itu.int/rec/R-REC-F.699-7-200604-I/en>`_.

    Parameters
    ----------
    diameter : `~astropy.units.Quantity`
        Antenna diameter [m]
    wavelength : `~astropy.units.Quantity`
        Observing wavelength [m]

    Returns
    -------
    G_max : `~astropy.units.Quantity`
        Antenna maximum gain [dBi]

    Notes
    -----
    See `ITU-R Rec F.699
    <https://www.itu.int/rec/R-REC-F.699-7-200604-I/en>`_ for further
    explanation and applicability of this model.
    '''

    return _fl_G_max_from_size(diameter, wavelength)


def _fl_G_max_from_hpbw(hpbw):

    return 44.5 - 20 * np.log10(hpbw)


@utils.ranged_quantity_input(
    hpbw=(1.e-3, 90., apu.deg),
    strip_input_units=True, output_unit=cnv.dBi
    )
def fl_G_max_from_hpbw(hpbw):
    '''
    Estimate maximum gain from antenna hpbw after `ITU-R Rec F.699
    <https://www.itu.int/rec/R-REC-F.699-7-200604-I/en>`_.

    Parameters
    ----------
    hpbw : `~astropy.units.Quantity`
        Antenna HPBW (3-dB point) [deg]

    Returns
    -------
    G_max : `~astropy.units.Quantity`
        Antenna maximum gain [dBi]

    Notes
    -----
    See `ITU-R Rec F.699
    <https://www.itu.int/rec/R-REC-F.699-7-200604-I/en>`_ for further
    explanation and applicability of this model.
    '''

    return _fl_G_max_from_hpbw(hpbw)


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
