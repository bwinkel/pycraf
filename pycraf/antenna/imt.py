#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

from astropy import units as apu
import numpy as np
from .. import conversions as cnv
from .. import utils


__all__ = [
    'imt2020_single_element_pattern', 'imt2020_composite_pattern',
    ]


def _A_EH(phi, A_m, phi_3db):
    '''
    Antenna element's horizontal pattern according to `IMT.MODEL
    <https://www.itu.int/md/R15-TG5.1-C-0036>`_ document.

    Parameters
    ----------
    phi : np.ndarray, float
        Azimuth [deg]
    A_m : np.ndarray, float
        Front-to-back ratio (horizontal) [dimless]
    phi_3db : np.ndarray, float
        Horizontal 3-dB beam width of single element [deg]

    Returns
    -------
    A_EH : np.ndarray, float
        Antenna element's horizontal radiation pattern [dBi]
    '''

    return -np.minimum(12 * (phi / phi_3db) ** 2, A_m)


def _A_EV(theta, SLA_nu, theta_3db):
    '''
    Antenna element's vertical pattern according to `IMT.MODEL
    <https://www.itu.int/md/R15-TG5.1-C-0036>`_ document.

    Parameters
    ----------
    theta : np.ndarray, float
        Elevation [deg]
    SLA_nu : np.ndarray, float
        Front-to-back ratio (vertical) [dimless]
    theta_3db : np.ndarray, float
        Vertical 3-dB beam width of single element [deg]

    Returns
    -------
    A_EV : np.ndarray, float
        Antenna element's vertical radiation pattern [dBi]
    '''

    return -np.minimum(12 * ((theta - 90.) / theta_3db) ** 2, SLA_nu)


def _imt2020_single_element_pattern(
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


@utils.ranged_quantity_input(
    azim=(-180, 180, apu.deg),
    elev=(-90, 90, apu.deg),
    G_Emax=(None, None, cnv.dB),
    A_m=(0, None, cnv.dimless),
    SLA_nu=(0, None, cnv.dimless),
    phi_3db=(0, None, apu.deg),
    theta_3db=(0, None, apu.deg),
    strip_input_units=True, output_unit=cnv.dBi
    )
def imt2020_single_element_pattern(
        azim, elev,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db
        ):
    '''
    Single antenna element's pattern according to `IMT.MODEL
    <https://www.itu.int/md/R15-TG5.1-C-0036>`_ document.

    Parameters
    ----------
    azim : `~astropy.units.Quantity`
        Azimuth [deg]
    elev : `~astropy.units.Quantity`
        Elevation [deg]
    G_Emax : `~astropy.units.Quantity`
        Single element maximum gain [dBi]
    A_m : `~astropy.units.Quantity`
        Front-to-back ratio (horizontal) [dimless]
    SLA_nu : `~astropy.units.Quantity`
        Front-to-back ratio (vertical) [dimless]
    phi_3db : `~astropy.units.Quantity`
        Horizontal 3dB beam width of single element [deg]
    theta_3db : `~astropy.units.Quantity`
        Vertical 3dB beam width of single element [deg]

    Returns
    -------
    A_E : `~astropy.units.Quantity`
        Single antenna element's pattern [dBi]

    Notes
    -----
    Further information can be found in 3GPP TR 37.840 Section 5.4.4.
    '''

    return _imt2020_single_element_pattern(
        azim, elev,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db
        )


def _imt2020_composite_pattern(
        azim, elev,
        azim_i, elev_i,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db,
        d_H, d_V,
        N_H, N_V,
        rho,
        ):

    phi = azim
    theta = 90. - elev
    phi_i = azim_i
    theta_i = elev_i  # sic! (tilt angle in imt.model is elevation)

    A_E = _imt2020_single_element_pattern(
        azim, elev,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db
        )

    # pre-compute some quantities for speed-up
    _dV_cos_theta = d_V * np.cos(np.radians(theta))
    _dH_sin_theta_sin_phi = (
        d_H * np.sin(np.radians(theta)) * np.sin(np.radians(phi))
        )

    _dV_sin_theta_i = d_V * np.sin(np.radians(theta_i))
    _dH_cos_theta_i_sin_phi_i = (
        d_H * np.cos(np.radians(theta_i)) * np.sin(np.radians(phi_i))
        )

    def nu(m, n):
        '''m, n zero-based (unlike in IMT.MODEL document, Table 4'''

        return np.exp(
            1j * 2 * np.pi * (
                n * _dV_cos_theta +
                m * _dH_sin_theta_sin_phi
                )
            )

    def w(m, n):
        '''m, n zero-based (unlike in IMT.MODEL document, Table 4'''

        return np.exp(
            1j * 2 * np.pi * (
                n * _dV_sin_theta_i +
                m * _dH_cos_theta_i_sin_phi_i
                )
            ) / np.sqrt(N_H * N_V)

    tmp = np.zeros(
        np.broadcast(phi, theta, phi_i, theta_i).shape, dtype=np.complex64
        )
    for m in range(N_H):
        for n in range(N_V):
            tmp += w(m, n) * nu(m, n)

    # account for correlation level
    tmp = 1 + rho * (np.abs(tmp) ** 2 - 1)

    return A_E + 10 * np.log10(tmp)


@utils.ranged_quantity_input(
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
    rho=(0, 1, cnv.dimless),
    strip_input_units=True, output_unit=cnv.dB
    )
def imt2020_composite_pattern(
        azim, elev,
        azim_i, elev_i,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db,
        d_H, d_V,
        N_H, N_V,
        rho=1 * cnv.dimless,
        ):
    '''
    Composite (array) antenna pattern according to `IMT.MODEL
    <https://www.itu.int/md/R15-TG5.1-C-0036>`_ document.

    Parameters
    ----------
    azim, elev : `~astropy.units.Quantity`
        Azimuth/Elevation [deg]
    azim_i, elev_i : `~astropy.units.Quantity`
        Azimuthal/Elevational pointing of beam `i` [deg]
    G_Emax : `~astropy.units.Quantity`
        Single element maximum gain [dBi]
    A_m, SLA_nu : `~astropy.units.Quantity`
        Front-to-back ratio (horizontal/vertical) [dimless]
    phi_3db, theta_3db : `~astropy.units.Quantity`
        Horizontal/Vertical 3dB beam width of single element [deg]
    d_H, d_V : `~astropy.units.Quantity`
        Horizontal/Vertical separation of beams in units of wavelength
        [dimless]
    N_H, N_V : int
        Horizontal/Vertical number of single antenna elements
    rho : `~astropy.units.Quantity`, optional
        Correlation level (see 3GPP TR 37.840, 5.4.4.1.4, default: 1) [dimless]

    Returns
    -------
    A_A : `~astropy.units.Quantity`
        Composite (array) antenna pattern of beam `i` [dB]

    Notes
    -----
    Further information can be found in 3GPP TR 37.840 Section 5.4.4.
    '''

    return _imt2020_composite_pattern(
        azim, elev,
        azim_i, elev_i,
        G_Emax,
        A_m, SLA_nu,
        phi_3db, theta_3db,
        d_H, d_V,
        N_H, N_V,
        rho,
        )


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
