#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

from astropy import units as apu
import numpy as np
from .. import conversions as cnv
from .. import helpers


__all__ = [
    'single_element_pattern', 'composite_pattern',
    ]


def _A_EH(phi, A_m, phi_3db):
    '''
    Antenna element's horizontal pattern according to IMT.MODEL document.

    Parameters
    ----------
    phi - azimuth [deg]
    A_m - front-to-back ratio (horizontal)
    phi_3db - horizontal 3dB bandwidth of single element [deg]

    Returns
    -------
    A_EH - Antenna element's horizontal radiation pattern [dB]
    '''

    return -np.minimum(12 * (phi / phi_3db) ** 2, A_m)


def _A_EV(theta, SLA_nu, theta_3db):
    '''
    Antenna element's vertical pattern according to IMT.MODEL document.

    Parameters
    ----------
    theta - elevation [deg]
    SLA_nu - front-to-back ratio (vertical)
    theta_3db - vertical 3dB bandwidth of single element [deg]

    Returns
    -------
    A_EV - Antenna element's vertical radiation pattern [dB]
    '''

    return -np.minimum(12 * ((theta - 90.) / theta_3db) ** 2, SLA_nu)


def _single_element_pattern(
        azim, elev,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db
        ):

    phi = azim
    theta = 90. - elev
    return G_Emax - np.minimum(
        -_A_EH(phi, A_m, phi_3db) - _A_EV(theta, SLA_nu, theta_3db),
        A_m
        )


@helpers.ranged_quantity_input(
    azim=(-180, 180, apu.deg),
    elev=(-90, 90, apu.deg),
    G_Emax=(None, None, cnv.dB),
    A_m=(0, None, cnv.dimless),
    SLA_nu=(0, None, cnv.dimless),
    phi_3db=(0, None, apu.deg),
    theta_3db=(0, None, apu.deg),
    strip_input_units=True, output_unit=cnv.dB
    )
def single_element_pattern(
        azim, elev,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db
        ):
    '''
    Single antenna element's pattern according to IMT.MODEL document.

    Parameters
    ----------
    azim - azimuth [deg]
    elev - elevation [deg]
    G_Emax - Single element maximum gain [dBi]
    A_m - front-to-back ratio (horizontal)
    SLA_nu - front-to-back ratio (vertical)
    phi_3db - horizontal 3dB bandwidth of single element [deg]
    theta_3db - vertical 3dB bandwidth of single element [deg]

    Returns
    -------
    A_E - Single antenna element's pattern [dB]
    '''

    return _single_element_pattern(
        azim, elev,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db
        )

def _composite_pattern(
        azim, elev,
        azim_i, elev_i,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db,
        d_H, d_V,
        N_H, N_V,
        ):

    phi = azim
    theta = 90. - elev
    phi_i = azim_i
    theta_i = elev_i  # sic! (tilt angle in imt.model is elevation)

    A_E = _single_element_pattern(
        azim, elev,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db
        )

    def nu(m, n):
        '''m, n zero-based (unlike in IMT.MODEL document, Table 4'''

        return np.exp(
            1j * 2 * np.pi * (
                n * d_V * np.cos(np.radians(theta)) +
                m * d_H * np.sin(np.radians(theta)) * np.sin(np.radians(phi))
                )
            )

    def w(m, n):
        '''m, n zero-based (unlike in IMT.MODEL document, Table 4'''

        return np.exp(
            1j * 2 * np.pi * (
                n * d_V * np.sin(np.radians(theta_i)) +
                m * d_H * np.cos(np.radians(theta_i)) *
                np.sin(np.radians(phi_i))
                )
            ) / np.sqrt(N_H * N_V)

    tmp = np.zeros(np.broadcast(phi, theta).shape, dtype=np.complex64)
    for m in range(N_H):
        for n in range(N_V):
            tmp += w(m, n) * nu(m, n)

    return A_E + 10 * np.log10(np.abs(tmp) ** 2)


@helpers.ranged_quantity_input(
    azim=(-180, 180, apu.deg),
    elev=(-90, 90, apu.deg),
    azim_i=(-180, 180, apu.deg),
    elev_i=(-90, 90, apu.deg),
    G_Emax=(None, None, cnv.dB),
    A_m=(0, None, cnv.dimless),
    SLA_nu=(0, None, cnv.dimless),
    phi_3db=(0, None, apu.deg),
    theta_3db=(0, None, apu.deg),
    d_H=(0, None, cnv.dimless),
    d_V=(0, None, cnv.dimless),
    strip_input_units=True, output_unit=cnv.dB
    )
def composite_pattern(
        azim, elev,
        azim_i, elev_i,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db,
        d_H, d_V,
        N_H, N_V,
        ):
    '''
    Composite (array) antenna pattern according to IMT.MODEL document.

    Parameters
    ----------
    azim, elev - azimuth/elevation [deg]
    azim_i, elev_i - azimuthal/elevation pointing of beam i [deg]
    G_Emax - Single element maximum gain [dBi]
    A_m, SLA_nu - front-to-back ratio (horizontal/vertical)
    phi_3db, theta_3db - horiz/vert 3dB bandwidth of single element [deg]
    d_H, d_V - horiz/vert separation of beams in units of wavelength [dimless]
    N_H, N_V - horiz/vert number of single antenna elements

    Returns
    -------
    A_A - Composite/array antenna pattern of beam i [dB]
    '''

    return _composite_pattern(
        azim, elev,
        azim_i, elev_i,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db,
        d_H, d_V,
        N_H, N_V,
        )


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
