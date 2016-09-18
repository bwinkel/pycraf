#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

import numpy as np
from astropy import units as apu
from astropy.units import Quantity, UnitsError
import astropy.constants as con


UNITS = [
    'dimless',
    'dB',
    'dBi',
    'dB_W',
    'dB_W_Hz',
    'dB_W_m2',
    'dB_W_m2_Hz',
    'dB_Jy_Hz',
    'dB_mW',
    'dB_uV_m',
    ]

__all__ = [
    'Aeff_from_Ageom', 'Ageom_from_Aeff',
    'Gain_from_Aeff', 'Aeff_from_Gain',
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
dB_mW = apu.dB(apu.mW)
dB_uV_m = apu.dB(apu.uV ** 2 / apu.m ** 2)


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


@apu.quantity_input(Ageom=apu.m ** 2, eta_a=dimless)
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

    return Ageom * eta_a.to(dimless)


@apu.quantity_input(Aeff=apu.m ** 2, eta_a=dimless)
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

    return Aeff / eta_a.to(dimless)


@apu.quantity_input(Aeff=apu.m ** 2, f=apu.Hz)
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

    return (4. * np.pi * Aeff * (f / con.c) ** 2).to(dBi)


@apu.quantity_input(G=dBi, f=apu.Hz)
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

    return (G.to(dimless) * (con.c / f) ** 2 / 4. / np.pi).to(apu.m ** 2)


# @apu.quantity_input(E=dB_uV_m, equivalencies=E_field_equivalency())
@apu.quantity_input(E=dB_uV_m)
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

    return (E.to(apu.uV / apu.meter) ** 2 / R0).to(apu.W / apu.m ** 2)


@apu.quantity_input(S=dB_W_m2)
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

    return (np.sqrt(S.to(apu.W / apu.m ** 2) * R0)).to(apu.uV / apu.meter)


@apu.quantity_input(Erx=dB_uV_m, d=apu.m, Gtx=dBi)
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

    return (
        4. * np.pi * d ** 2 / Gtx.to(dimless) *
        Erx.to(apu.uV / apu.meter) ** 2 / R0
        ).to(apu.W)


@apu.quantity_input(Ptx=dB_W, d=apu.m, Gtx=dBi)
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

    return (
        (Ptx.to(apu.W) * Gtx.to(dimless) / 4. / np.pi * R0) ** 0.5 / d
        ).to(apu.uV / apu.meter)


@apu.quantity_input(Ptx=dB_W, d=apu.m, Gtx=dBi)
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

    return (
        Gtx.to(dimless) * Ptx.to(apu.W) / 4. / np.pi / d ** 2
        ).to(apu.W / apu.m ** 2)


@apu.quantity_input(S=dB_W_m2, d=apu.m, Gtx=dBi)
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

    return (
        S.to(apu.W / apu.m ** 2) * 4. * np.pi * d ** 2 / Gtx.to(dimless)
        ).to(apu.W)


@apu.quantity_input(Prx=dB_W, f=apu.Hz, Gtx=dBi)
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

    return (
        Prx.to(apu.W) / Grx.to(dimless) * (
            4. * np.pi * f ** 2 / con.c ** 2
            )
        ).to(apu.W / apu.m ** 2)


@apu.quantity_input(S=dB_W_m2, f=apu.Hz, Gtx=dBi)
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

    return (
        S.to(apu.W / apu.m ** 2) * Grx.to(dimless) * (
            con.c ** 2 / 4. / np.pi / f ** 2
            )
        ).to(apu.W)


@apu.quantity_input(d=apu.m, f=apu.Hz)
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

    return ((con.c / 4. / np.pi / f / d) ** 2).to(dB)


@apu.quantity_input(Ptx=dB_W, Gtx=dBi, Grx=dBi, d=apu.m, f=apu.Hz)
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

    return (
        Ptx.to(apu.W) * Gtx.to(dimless) * Grx.to(dimless) *
        free_space_loss(d, f).to(dimless)
        ).to(apu.W)


@apu.quantity_input(Prx=dB_W, Gtx=dBi, Grx=dBi, d=apu.m, f=apu.Hz)
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

    return (
        Prx.to(apu.W) / Gtx.to(dimless) / Grx.to(dimless) /
        free_space_loss(d, f).to(dimless)
        ).to(apu.W)


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
