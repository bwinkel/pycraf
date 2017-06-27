#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

# from functools import partial, lru_cache
from astropy import units as apu
import numpy as np
from .. import conversions as cnv
from .. import helpers


__all__ = [
    'fl_pattern',
    'fl_G_max_from_hpbw', 'fl_G_max_from_size', 'fl_hpbw_from_size',
    ]


def _fl_pattern_2_1(phi, diameter_over_wavelength, G_max):
    '''
    Antenna gain as a function of angular distance after ITU-R F.699-7 2.1.

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
    Antenna gain as a function of angular distance after ITU-R F.699-7 2.2.

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
    Antenna gain as a function of angular distance after ITU-R F.699-7 2.3.

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


def _fl_pattern(phi, diameter, wavelength, G_max):

    d_wlen = diameter / wavelength

    phi, d_wlen, G_max = np.broadcast_arrays(phi, d_wlen, G_max)
    gain = np.full(phi.shape, np.nan, np.float64)

    tmp_mask = (0.00428 < wavelength) & (wavelength < 0.29979)  # 1...70 GHz
    mask_2_1 = (d_wlen > 100) & tmp_mask
    mask_2_2 = (d_wlen <= 100) & tmp_mask

    tmp_mask = (0.29979 <= wavelength) & (wavelength < 2.99792)  # 0.1...1 GHz
    mask_2_3 = (d_wlen > 0.63) & tmp_mask

    for mask, func in [
            (mask_2_1, _fl_pattern_2_1),
            (mask_2_2, _fl_pattern_2_2),
            (mask_2_3, _fl_pattern_2_3),
            ]:

        gain[mask] = func(phi[mask], d_wlen[mask], G_max[mask])

    return gain


@helpers.ranged_quantity_input(
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
    Antenna gain as a function of angular distance after ITU-R F.699-7.

    Parameters
    ----------
    phi - angular distance in degrees
    diameter - antenna diameter
    wavelength - observing wavelength
    G_max - antenna maximum gain
        Note: if this is unknown, there is a helper function G_max_from_HPBW
        that can be used to estimate a value.

    Returns
    -------
    antenna gain in dBi

    Notes
    -----
    See ITU-R F.699-7 for explanation and applicability of this model.


    Example
    -------

    from pycraf.antenna import *
    from pycraf.conversions import *
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as apu

    phi = np.linspace(-20, 20, 1000) * apu.deg
    diam = np.array([1, 2, 10])[:, np.newaxis] * apu.m
    wavlen = 0.03 * apu.m  # about 10 GHz
    G_max = fl_G_max_from_size(diam, wavlen)
    gain = fl_pattern(phi, diam, wavlen, G_max)
    plt.plot(phi, gain.T, '-')
    plt.show()

    '''

    return _fl_pattern(phi, diameter, wavelength, G_max)


def _fl_hpbw_from_size(diameter, wavelength):

    return 70. * wavelength / diameter


@helpers.ranged_quantity_input(
    diameter=(0.1, 1000., apu.m),
    wavelength=(0.001, 2, apu.m),
    strip_input_units=True, output_unit=apu.deg
    )
def fl_hpbw_from_size(diameter, wavelength):
    '''
    Estimate HPBW from antenna diameter after ITU-R F.699-7.

    Parameters
    ----------
    diameter - antenna diameter
    wavelength - observing wavelength

    Returns
    -------
    hpbw - antenna HPBW (3 dB point) in deg

    Notes
    -----
    See ITU-R F.699-7 for explanation and applicability of this model.
    '''

    return _fl_hpbw_from_size(diameter, wavelength)


def _fl_G_max_from_size(diameter, wavelength):

    return 20. * np.log10(diameter / wavelength) + 7.7


@helpers.ranged_quantity_input(
    diameter=(0.1, 1000., apu.m),
    wavelength=(0.001, 2, apu.m),
    strip_input_units=True, output_unit=cnv.dBi
    )
def fl_G_max_from_size(diameter, wavelength):
    '''
    Estimate G_max from antenna diameter after ITU-R F.699-7.

    Parameters
    ----------
    diameter - antenna diameter
    wavelength - observing wavelength

    Returns
    -------
    G_max - antenna maximum gain

    Notes
    -----
    See ITU-R F.699-7 for explanation and applicability of this model.
    '''

    return _fl_G_max_from_size(diameter, wavelength)


def _fl_G_max_from_hpbw(hpbw):

    return 44.5 - 20 * np.log10(hpbw)


@helpers.ranged_quantity_input(
    hpbw=(1.e-3, 90., apu.deg),
    strip_input_units=True, output_unit=cnv.dBi
    )
def fl_G_max_from_hpbw(hpbw):
    '''
    Estimate G_max from antenna hpbw after ITU-R F.699-7.

    Parameters
    ----------
    hpbw - antenna HPBW (3 dB point) in deg

    Returns
    -------
    G_max - antenna maximum gain

    Notes
    -----
    See ITU-R F.699-7 for explanation and applicability of this model.
    '''

    return _fl_G_max_from_hpbw(hpbw)


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
