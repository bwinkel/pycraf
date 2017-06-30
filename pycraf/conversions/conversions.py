#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

import numpy as np
from astropy import units as apu
from astropy.units import Quantity, UnitsError
import astropy.constants as con
from .. import helpers


UNITS = [
    'dimless',
    'dB',
    'dBi',
    'dB_W',
    'dB_W_Hz',
    'dB_W_m2',
    'dB_W_m2_Hz',
    'dB_Jy_Hz',
    'dB_mW', 'dBm',
    'dB_mW_MHz', 'dBm_MHz',  # this is often used in engineering (dBm/MHz)
    'dB_1_m',
    'dB_uV_m',
    ]

__all__ = [
    'Aeff_from_Ageom', 'Ageom_from_Aeff',
    'Gain_from_Aeff', 'Aeff_from_Gain',
    'Ant_factor_from_Gain', 'Gain_from_Ant_factor',
    'S_from_E', 'E_from_S',
    'Ptx_from_Erx', 'Erx_from_Ptx',
    'S_from_Ptx', 'Ptx_from_S',
    'Prx_from_S', 'S_from_Prx',
    'Prx_from_Ptx', 'Ptx_from_Prx',
    'free_space_loss',
    'Erx_unit', 'R0', 'E_field_equivalency',
    ] + UNITS



# define some useful dB-Scales
dimless = apu.Unit(1)
dB = dBi = apu.dB(dimless)
dB_W = apu.dB(apu.W)
dB_W_Hz = apu.dB(apu.W / apu.Hz)
dB_W_m2 = apu.dB(apu.W / apu.m ** 2)
dB_W_m2_Hz = apu.dB(apu.W / apu.Hz / apu.m ** 2)
dB_Jy_Hz = apu.dB(apu.Jy * apu.Hz)
dBm = dB_mW = apu.dB(apu.mW)
dBm_MHz = dB_mW_MHz = apu.dB(apu.mW / apu.MHz)
dB_uV_m = apu.dB(apu.uV ** 2 / apu.m ** 2)
dB_1_m = apu.dB(1. / apu.m)  # for antenna factor


# Astropy.unit equivalency between linear and logscale field strength
# this is necessary, because the dB_uV_m is from E ** 2 (dB scale is power)
# one can make use of the equivalency in the .to() function, e.g.:
#     Erx_unit.to(cnv.dB_uV_m, equivalencies=E_field_equivalency)
# this conflicts with apu.logarithmic():
# def E_field_equivalency():
#     return [(
#         apu.uV / apu.m,
#         dB_uV_m,
#         lambda x: 10. * np.log10(x ** 2),
#         lambda x: np.sqrt(10 ** (x / 10.))
#         )]
def E_field_equivalency():
    return [(
        apu.uV / apu.m,
        (apu.uV / apu.m) ** 2,
        lambda x: x ** 2,
        lambda x: x ** 0.5
        )]


# apu.add_enabled_equivalencies(apu.logarithmic())
apu.add_enabled_equivalencies(E_field_equivalency())

# define some useful constants
R0 = (
    1. * (con.mu0 / con.eps0) ** 0.5
    ).to(apu.Ohm)
Erx_unit = (
    (1 * apu.W / 4. / np.pi * R0) ** 0.5 / (1 * apu.km)
    ).to(apu.uV / apu.m)
C_VALUE = con.c.to(apu.m / apu.s).value
R0_VALUE = R0.to(apu.Ohm).value
ERX_VALUE = Erx_unit.to(apu.V / apu.m).value


@helpers.ranged_quantity_input(
    Ageom=(0, None, apu.m ** 2),
    eta_a=(0, 100, apu.percent),
    strip_input_units=True, output_unit=apu.m ** 2
    )
def Aeff_from_Ageom(Ageom, eta_a):
    '''
    Calculate effective ant. area from geometric area, given ant. efficiency.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    Ageom - Geometric antenna area [m**2]
    eta_a - Antenna efficiency [% or dimless]

    Returns
    -------
    Effective antenna area, Aeff [m**2]
    '''

    return Ageom * eta_a / 100.


@helpers.ranged_quantity_input(
    Aeff=(0, None, apu.m ** 2),
    eta_a=(0, 100, apu.percent),
    strip_input_units=True, output_unit=apu.m ** 2
    )
def Ageom_from_Aeff(Aeff, eta_a):
    '''
    Calculate geometric ant. area from effective area, given ant. efficiency.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    Aeff - Effective antenna area [m**2]
    eta_a - Antenna efficiency [% or dimless]

    Returns
    -------
    Geometric antenna area, Ageom [m**2]
    '''

    return Aeff / eta_a * 100.


@helpers.ranged_quantity_input(
    Aeff=(0, None, apu.m ** 2),
    f=(0, None, apu.Hz),
    strip_input_units=True, output_unit=dBi
    )
def Gain_from_Aeff(Aeff, f):
    '''
    Calculate antenna gain from effective antenna area, given frequency.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    Aeff - Effective antenna area [m**2]
    f - Frequency [Hz]

    Returns
    -------
    Antenna gain, G [dBi]
    '''

    return 10 * np.log10(
        4. * np.pi * Aeff * (f / C_VALUE) ** 2
        )


@helpers.ranged_quantity_input(
    G=(1.e-30, None, dimless),
    f=(0, None, apu.Hz),
    strip_input_units=True, output_unit=apu.m ** 2
    )
def Aeff_from_Gain(G, f):
    '''
    Calculate effective antenna area from antenna gain, given frequency.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    G - Antenna gain [dBi, or dimless]
    f - Frequency [Hz]

    Returns
    -------
    Effective antenna area, Aeff [m**2]
    '''

    return G * (C_VALUE / f) ** 2 / 4. / np.pi


@helpers.ranged_quantity_input(
    G=(1.e-30, None, dimless),
    f=(0, None, apu.Hz),
    Zi=(0, None, apu.Ohm),
    strip_input_units=True, output_unit=dB_1_m
    )
def Ant_factor_from_Gain(G, f, Zi):
    '''
    Calculate antenna factor from antenna gain, given frequency and impedance.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    G - Antenna gain [dBi, or dimless]
    f - Frequency [Hz]
    Zi - Receiver impedance [Ohm]

    Returns
    -------
    Antenna factor, Ka [dB(1/m)]
    '''

    return 10 * np.log10(np.sqrt(
        4. * np.pi / G * (f / C_VALUE) ** 2 * R0_VALUE / Zi
        ))


@helpers.ranged_quantity_input(
    Ka=(1.e-30, None, 1. / apu.m),
    f=(0, None, apu.Hz),
    Zi=(0, None, apu.Ohm),
    strip_input_units=True, output_unit=dBi
    )
def Gain_from_Ant_factor(Ka, f, Zi):
    '''
    Calculate antenna gain from antenna factor, given frequency and impedance.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    Ka - Antenna factor, Ka [1/m]
    f - Frequency [Hz]
    Zi - Receiver impedance [Ohm]

    Returns
    -------
    Antenna gain [dBi]
    '''

    return 10 * np.log10(
        4. * np.pi / Ka ** 2 * (f / C_VALUE) ** 2 * R0_VALUE / Zi
        )


# @apu.quantity_input(E=dB_uV_m, equivalencies=E_field_equivalency())
@helpers.ranged_quantity_input(
    E=(1.e-30, None, apu.V / apu.meter),
    strip_input_units=True, output_unit=apu.W / apu.m ** 2
    )
def S_from_E(E):
    '''
    Calculate power flux density, S, from field strength.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    E - Received E-field strength [uV/m]

    Returns
    -------
    Power flux density, S [dB_W_m2 or W/m**2]
    '''

    return E ** 2 / R0_VALUE


@helpers.ranged_quantity_input(
    S=(None, None, apu.W / apu.m ** 2),
    strip_input_units=True, output_unit=apu.uV / apu.meter
    )
def E_from_S(S):
    '''
    Calculate field strength, E, from power flux density.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    S - Power flux density [dB_W_m2 or W/m**2]

    Returns
    -------
    Received E-field strength, E [uV/m]
    '''

    # return np.sqrt(np.power(10., 0.1 * S) * R0_VALUE) * 1.e6
    return np.sqrt(S * R0_VALUE) * 1.e6


@helpers.ranged_quantity_input(
    Erx=(1.e-30, None, apu.V / apu.meter),
    d=(1.e-30, None, apu.m),
    Gtx=(1.e-30, None, dimless),
    strip_input_units=True, output_unit=apu.W
    )
def Ptx_from_Erx(Erx, d, Gtx):
    '''
    Calculate transmitter power, Ptx, from received field strength.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    Erx - Received E-field strength [dB_uV_m, uV/m, or (uV/m)**2]
    d - Distance to transmitter [m]
    Gtx - Gain of transmitter [dBi, or dimless]

    Returns
    -------
    Transmitter power, Ptx [W]
    '''

    return 4. * np.pi * d ** 2 / Gtx * Erx ** 2 / R0_VALUE


@helpers.ranged_quantity_input(
    Ptx=(1.e-30, None, apu.W),
    d=(1.e-30, None, apu.m),
    Gtx=(1.e-30, None, dimless),
    strip_input_units=True, output_unit=apu.uV / apu.meter
    )
def Erx_from_Ptx(Ptx, d, Gtx):
    '''
    Calculate received field strength, Erx, from transmitter power.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    Ptx - Transmitter power [dB_W, W]
    d - Distance to transmitter [m]
    Gtx - Gain of transmitter [dBi, or dimless]

    Returns
    -------
    Received E-field strength, Erx [uV/m]
    '''

    return (Ptx * Gtx / 4. / np.pi * R0_VALUE) ** 0.5 / d * 1.e6


@helpers.ranged_quantity_input(
    Ptx=(1.e-30, None, apu.W),
    d=(1.e-30, None, apu.m),
    Gtx=(1.e-30, None, dimless),
    strip_input_units=True, output_unit=apu.W / apu.m ** 2
    )
def S_from_Ptx(Ptx, d, Gtx):
    '''
    Calculate power flux density, S, from transmitter power.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    Ptx - Transmitter power [dB_W, W]
    d - Distance to transmitter [m]
    Gtx - Gain of transmitter [dBi, or dimless]

    Returns
    -------
    Power flux density, S (at receiver location) [W/m**2]
    '''

    # log-units seem not yet flexible enough to make the simpler
    # statement work:
    # return Gtx * Ptx / 4. / np.pi / d ** 2
    # (would be doable with apu.logarithmic() environment)

    return Gtx * Ptx / 4. / np.pi / d ** 2


@helpers.ranged_quantity_input(
    S=(1.e-30, None, apu.W / apu.m ** 2),
    d=(1.e-30, None, apu.m),
    Gtx=(1.e-30, None, dimless),
    strip_input_units=True, output_unit=apu.W
    )
def Ptx_from_S(S, d, Gtx):
    '''
    Calculate transmitter power, Ptx, from power flux density.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    S - Power flux density (at receiver location) [W/m**2, dB_W_m2]
    d - Distance to transmitter [m]
    Gtx - Gain of transmitter [dBi, or dimless]

    Returns
    -------
    Transmitter power, Ptx [W]
    '''

    return S * 4. * np.pi * d ** 2 / Gtx


@helpers.ranged_quantity_input(
    Prx=(1.e-30, None, apu.W),
    f=(1.e-30, None, apu.Hz),
    Grx=(1.e-30, None, dimless),
    strip_input_units=True, output_unit=apu.W / apu.m ** 2
    )
def S_from_Prx(Prx, f, Grx):
    '''
    Calculate power flux density, S, from received power.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    Prx - Received power [dB_W, W]
    f - Frequency of radiation [Hz]
    Grx - Gain of receiver [dBi, or dimless]

    Returns
    -------
    Power flux density, S (at receiver location) [W]
    '''

    return Prx / Grx * (
        4. * np.pi * f ** 2 / C_VALUE ** 2
        )


@helpers.ranged_quantity_input(
    S=(1.e-30, None, apu.W / apu.m ** 2),
    f=(1.e-30, None, apu.Hz),
    Grx=(1.e-30, None, dimless),
    strip_input_units=True, output_unit=apu.W
    )
def Prx_from_S(S, f, Grx):
    '''
    Calculate received power, Prx, from power flux density.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    S - Power flux density (at receiver location) [W/m**2, dB_W_m2]
    f - Frequency of radiation [Hz]
    Grx - Gain of receiver [dBi, or dimless]

    Returns
    -------
    Received power, Prx [W]
    '''

    return S * Grx * (
        C_VALUE ** 2 / 4. / np.pi / f ** 2
        )


def _free_space_loss(d, f):

    return (C_VALUE / 4. / np.pi / f / d) ** 2


@helpers.ranged_quantity_input(
    d=(1.e-30, None, apu.m),
    f=(1.e-30, None, apu.Hz),
    strip_input_units=True, output_unit=dB
    )
def free_space_loss(d, f):
    '''
    Calculate the free space loss of a propagating radio wave.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    d - Distance between transmitter and receiver [m]
    f - Frequency of radiation [Hz]

    Returns
    -------
    Free-space loss, FSPL [dB]
    '''

    return 10. * np.log10(_free_space_loss(f, d))


@helpers.ranged_quantity_input(
    Ptx=(1.e-30, None, apu.W),
    Gtx=(1.e-30, None, dimless),
    Grx=(1.e-30, None, dimless),
    d=(1.e-30, None, apu.m),
    f=(1.e-30, None, apu.Hz),
    strip_input_units=True, output_unit=apu.W
    )
def Prx_from_Ptx(Ptx, Gtx, Grx, d, f):
    '''
    Calculate received power, Prx, from transmitted power.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    Ptx - Transmitter power [dB_W, W]
    Gtx - Gain of transmitter [dBi, or dimless]
    Grx - Gain of receiver [dBi, or dimless]
    d - Distance between transmitter and receiver [m]
    f - Frequency of radiation [Hz]

    Returns
    -------
    Received power, Prx [W]
    '''

    return Ptx * Gtx * Grx * _free_space_loss(d, f)


@helpers.ranged_quantity_input(
    Prx=(1.e-30, None, apu.W),
    Gtx=(1.e-30, None, dimless),
    Grx=(1.e-30, None, dimless),
    d=(1.e-30, None, apu.m),
    f=(1.e-30, None, apu.Hz),
    strip_input_units=True, output_unit=apu.W
    )
def Ptx_from_Prx(Prx, Gtx, Grx, d, f):
    '''
    Calculate transmitted power, Prx, from received power.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    Prx - Received power [dB_W, W]
    Gtx - Gain of transmitter [dBi, or dimless]
    Grx - Gain of receiver [dBi, or dimless]
    d - Distance between transmitter and receiver [m]
    f - Frequency of radiation [Hz]

    Returns
    -------
    Transmitter power, Ptx [W]
    '''

    return Prx / Gtx / Grx / _free_space_loss(d, f)


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
