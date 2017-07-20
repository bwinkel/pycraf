#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

import numpy as np
from astropy import units as apu
from astropy.units import Quantity, UnitsError
import astropy.constants as con
from .. import utils


UNITS = [
    'dimless',
    'dB', 'dBi', 'dBc',
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
    'eff_from_geom_area', 'geom_from_eff_area',
    'eta_a_from_areas',
    'gain_from_eff_area', 'eff_area_from_gain',
    'antfactor_from_gain', 'gain_from_antfactor',
    'powerflux_from_efield', 'efield_from_powerflux',
    'ptx_from_efield', 'efield_from_ptx',
    'powerflux_from_ptx', 'ptx_from_powerflux',
    'prx_from_powerflux', 'powerflux_from_prx',
    'prx_from_ptx', 'ptx_from_prx',
    'free_space_loss',
    'Erx_unit', 'R0', 'efield_equivalency',
    ] + UNITS


# define some useful dB-Scales
# dimless = apu.Unit(1)
dimless = apu.dimensionless_unscaled
dB = dBi = dBc = apu.dB(dimless)
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
#     Erx_unit.to(cnv.dB_uV_m, equivalencies=efield_equivalency)
# this conflicts with apu.logarithmic():
# def efield_equivalency():
#     return [(
#         apu.uV / apu.m,
#         dB_uV_m,
#         lambda x: 10. * np.log10(x ** 2),
#         lambda x: np.sqrt(10 ** (x / 10.))
#         )]


def efield_equivalency():
    '''
    `~astropy.units` equivalency to handle log-scale E-field units.

    For electric fields, the Decibel scale is define via the amplitude
    of the field squared, :math:`{\\vert\\vec E\\vert}^2` which is
    proportional to the power.

    Returns
    -------
    equivalency : list
        The returned list contains one tuple with the equivalency.
    '''
    return [(
        apu.uV / apu.m,
        (apu.uV / apu.m) ** 2,
        lambda x: x ** 2,
        lambda x: x ** 0.5
        )]


# apu.add_enabled_equivalencies(apu.logarithmic())
apu.add_enabled_equivalencies(efield_equivalency())

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


@utils.ranged_quantity_input(
    geom_area=(0, None, apu.m ** 2),
    eta_a=(0, 100, apu.percent),
    strip_input_units=True, output_unit=apu.m ** 2
    )
def eff_from_geom_area(geom_area, eta_a):
    '''
    Effective antenna area from geometric area, given antenna efficiency.

    The effective and geometric antenna areas are linked via the antenna
    efficiency:

        A_eff = eta_a * A_geom.

    Parameters
    ----------
    geom_area : `~astropy.units.Quantity`
        Geometric antenna area, A_geom [m**2]
    eta_a : `~astropy.units.Quantity`
        Antenna efficiency [%, dimless]

    Returns
    -------
    eff_area : `~astropy.units.Quantity`
        Effective antenna area, A_eff [m**2]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return geom_area * eta_a / 100.


@utils.ranged_quantity_input(
    eff_area=(0, None, apu.m ** 2),
    eta_a=(0, 100, apu.percent),
    strip_input_units=True, output_unit=apu.m ** 2
    )
def geom_from_eff_area(eff_area, eta_a):
    '''
    Geometric antenna area from effective area, given antenna efficiency.

    The effective and geometric antenna areas are linked via the antenna
    efficiency:

        A_eff = eta_a * A_geom.

    Parameters
    ----------
    eff_area : `~astropy.units.Quantity`
        Effective antenna area, A_eff [m**2]
    eta_a : `~astropy.units.Quantity`
        Antenna efficiency [%, dimless]

    Returns
    -------
    geom_area : `~astropy.units.Quantity`
        Geometric antenna area, A_geom [m**2]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return eff_area / eta_a * 100.


@utils.ranged_quantity_input(
    geom_area=(0, None, apu.m ** 2),
    eff_area=(0, None, apu.m ** 2),
    strip_input_units=True, output_unit=apu.percent
    )
def eta_a_from_areas(geom_area, eff_area):
    '''
    Antenna efficiency from geometric and effective antenna areas.

    The effective and geometric antenna areas are linked via the antenna
    efficiency:

        A_eff = eta_a * A_geom.

    Parameters
    ----------
    eff_area : `~astropy.units.Quantity`
        Effective antenna area, A_eff [m**2]
    geom_area : `~astropy.units.Quantity`
        Geometric antenna area, A_geom [m**2]

    Returns
    -------
    eta_a : `~astropy.units.Quantity`
        Antenna efficiency [%, dimless]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return eff_area / geom_area * 100.


@utils.ranged_quantity_input(
    eff_area=(0, None, apu.m ** 2),
    freq=(0, None, apu.Hz),
    strip_input_units=True, output_unit=dBi
    )
def gain_from_eff_area(eff_area, freq):
    '''
    Antenna gain from effective antenna area, given frequency.

    Parameters
    ----------
    eff_area : `~astropy.units.Quantity`
        Effective antenna area, A_eff [m**2]
    freq : `~astropy.units.Quantity`
        Frequency [Hz]

    Returns
    -------
    gain : `~astropy.units.Quantity`
        Antenna gain [dBi]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return 10 * np.log10(
        4. * np.pi * eff_area * (freq / C_VALUE) ** 2
        )


@utils.ranged_quantity_input(
    gain=(1.e-30, None, dimless),
    freq=(0, None, apu.Hz),
    strip_input_units=True, output_unit=apu.m ** 2
    )
def eff_area_from_gain(gain, freq):
    '''
    Effective antenna area from antenna gain, given frequency.

    Parameters
    ----------
    gain : `~astropy.units.Quantity`
        Antenna gain [dBi, dimless]
    freq : `~astropy.units.Quantity`
        Frequency [Hz]

    Returns
    -------
    eff_area : `~astropy.units.Quantity`
        Effective antenna area, A_eff [m**2]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return gain * (C_VALUE / freq) ** 2 / 4. / np.pi


@utils.ranged_quantity_input(
    gain=(1.e-30, None, dimless),
    freq=(0, None, apu.Hz),
    zi=(0, None, apu.Ohm),
    strip_input_units=True, output_unit=dB_1_m
    )
def antfactor_from_gain(gain, freq, zi):
    '''
    Antenna factor from antenna gain, given frequency and impedance.

    Parameters
    ----------
    gain : `~astropy.units.Quantity`
        Antenna gain [dBi, or dimless]
    freq : `~astropy.units.Quantity`
        Frequency [Hz]
    zi : `~astropy.units.Quantity`
        Receiver impedance, Zi [Ohm]

    Returns
    -------
    antfactor : `~astropy.units.Quantity`
        Antenna factor, Ka [dB(1/m)]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return 10 * np.log10(np.sqrt(
        4. * np.pi / gain * (freq / C_VALUE) ** 2 * R0_VALUE / zi
        ))


@utils.ranged_quantity_input(
    antfactor=(1.e-30, None, 1. / apu.m),
    freq=(0, None, apu.Hz),
    zi=(0, None, apu.Ohm),
    strip_input_units=True, output_unit=dBi
    )
def gain_from_antfactor(antfactor, freq, zi):
    '''
    Antenna gain from antenna factor, given frequency and impedance.

    Parameters
    ----------
    antfactor : `~astropy.units.Quantity`
        Antenna factor, Ka [1/m]
    freq : `~astropy.units.Quantity`
        Frequency [Hz]
    zi : `~astropy.units.Quantity`
        Receiver impedance, Zi [Ohm]

    Returns
    -------
    gain : `~astropy.units.Quantity`
        Antenna gain [dBi, or dimless]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return 10 * np.log10(
        4. * np.pi / antfactor ** 2 * (freq / C_VALUE) ** 2 * R0_VALUE / zi
        )


# @apu.quantity_input(E=dB_uV_m, equivalencies=efield_equivalency())
@utils.ranged_quantity_input(
    efield=(1.e-30, None, apu.V / apu.meter),
    strip_input_units=True, output_unit=apu.W / apu.m ** 2
    )
def powerflux_from_efield(efield):
    '''
    Power flux density from E-field strength.

    Parameters
    ----------
    efield : `~astropy.units.Quantity`
        E-field strength, E [uV/m]

    Returns
    -------
    powerflux : `~astropy.units.Quantity`
        Power flux density, S [dB_W_m2, W/m**2]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return efield ** 2 / R0_VALUE


@utils.ranged_quantity_input(
    powerflux=(None, None, apu.W / apu.m ** 2),
    strip_input_units=True, output_unit=apu.uV / apu.meter
    )
def efield_from_powerflux(powerflux):
    '''
    E-field strength from power flux density.

    Parameters
    ----------
    powerflux : `~astropy.units.Quantity`
        Power flux density, S [dB_W_m2 or W/m**2]

    Returns
    -------
    efield : `~astropy.units.Quantity`
        E-field strength, E [uV/m]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return np.sqrt(powerflux * R0_VALUE) * 1.e6


@utils.ranged_quantity_input(
    efield=(1.e-30, None, apu.V / apu.meter),
    dist=(1.e-30, None, apu.m),
    gtx=(1.e-30, None, dimless),
    strip_input_units=True, output_unit=apu.W
    )
def ptx_from_efield(efield, dist, gtx):
    '''
    Transmitter power from E-field strength measured at distance.

    Parameters
    ----------
    efield : `~astropy.units.Quantity`
        E-field strength, E [dB_uV_m, uV/m, (uV/m)**2]
    dist : `~astropy.units.Quantity`
        Distance to transmitter [m]
    gtx : `~astropy.units.Quantity`
        Gain of transmitter, Gtx [dBi, or dimless]

    Returns
    -------
    ptx : `~astropy.units.Quantity`
        Transmitter power, Ptx [W]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return 4. * np.pi * dist ** 2 / gtx * efield ** 2 / R0_VALUE


@utils.ranged_quantity_input(
    ptx=(1.e-30, None, apu.W),
    dist=(1.e-30, None, apu.m),
    gtx=(1.e-30, None, dimless),
    strip_input_units=True, output_unit=apu.uV / apu.meter
    )
def efield_from_ptx(ptx, dist, gtx):
    '''
    E-field strength measured at distance from transmitter power.

    Parameters
    ----------
    ptx : `~astropy.units.Quantity`
        Transmitter power, Ptx [dB_W, W]
    dist : `~astropy.units.Quantity`
        Distance to transmitter [m]
    gtx : `~astropy.units.Quantity`
        Gain of transmitter, Gtx [dBi, dimless]

    Returns
    -------
    efield : `~astropy.units.Quantity`
        E-field strength, E [uV/m]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return (ptx * gtx / 4. / np.pi * R0_VALUE) ** 0.5 / dist * 1.e6


@utils.ranged_quantity_input(
    ptx=(1.e-30, None, apu.W),
    dist=(1.e-30, None, apu.m),
    gtx=(1.e-30, None, dimless),
    strip_input_units=True, output_unit=apu.W / apu.m ** 2
    )
def powerflux_from_ptx(ptx, dist, gtx):
    '''
    Power flux density from transmitter power.

    Parameters
    ----------
    ptx : `~astropy.units.Quantity`
        Transmitter power, Ptx [dB_W, W]
    dist : `~astropy.units.Quantity`
        Distance to transmitter [m]
    gtx : `~astropy.units.Quantity`
        Gain of transmitter, Gtx [dBi, dimless]

    Returns
    -------
    powerflux : `~astropy.units.Quantity`
        Power flux density, S (at distance) [W/m**2]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return gtx * ptx / 4. / np.pi / dist ** 2


@utils.ranged_quantity_input(
    powerflux=(1.e-30, None, apu.W / apu.m ** 2),
    dist=(1.e-30, None, apu.m),
    gtx=(1.e-30, None, dimless),
    strip_input_units=True, output_unit=apu.W
    )
def ptx_from_powerflux(powerflux, dist, gtx):
    '''
    Transmitter power from power flux density.

    Parameters
    ----------
    powerflux : `~astropy.units.Quantity`
        Power flux density, S (at distance) [W/m**2, dB_W_m2]
    dist : `~astropy.units.Quantity`
        Distance to transmitter [m]
    gtx : `~astropy.units.Quantity`
        Gain of transmitter, Gtx [dBi, dimless]

    Returns
    -------
    ptx : `~astropy.units.Quantity`
        Transmitter power, Ptx [W]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return powerflux * 4. * np.pi * dist ** 2 / gtx


@utils.ranged_quantity_input(
    prx=(1.e-30, None, apu.W),
    freq=(1.e-30, None, apu.Hz),
    grx=(1.e-30, None, dimless),
    strip_input_units=True, output_unit=apu.W / apu.m ** 2
    )
def powerflux_from_prx(prx, freq, grx):
    '''
    Power flux density from received power.

    Power flux density and received power are linked via effective antenna
    area (which is propotional to receiving antenna gain).

    Parameters
    ----------
    prx : `~astropy.units.Quantity`
        Received power [dB_W, W]
    freq : `~astropy.units.Quantity`
        Frequency of radiation [Hz]
    grx : `~astropy.units.Quantity`
        Gain of receiver, Grx [dBi, dimless]

    Returns
    -------
    powerflux : `~astropy.units.Quantity`
        Power flux density [W/m**2]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return prx / grx * (
        4. * np.pi * freq ** 2 / C_VALUE ** 2
        )


@utils.ranged_quantity_input(
    powerflux=(1.e-30, None, apu.W / apu.m ** 2),
    freq=(1.e-30, None, apu.Hz),
    grx=(1.e-30, None, dimless),
    strip_input_units=True, output_unit=apu.W
    )
def prx_from_powerflux(powerflux, freq, grx):
    '''
    Received power from power flux density.

    Power flux density and received power are linked via effective antenna
    area (which is propotional to receiving antenna gain).

    Parameters
    ----------
    powerflux : `~astropy.units.Quantity`
        Power flux density [W/m**2, dB_W_m2]
    freq : `~astropy.units.Quantity`
        Frequency of radiation [Hz]
    grx : `~astropy.units.Quantity`
        Gain of receiver, Grx [dBi, dimless]

    Returns
    -------
    prx : `~astropy.units.Quantity`
        Received power [W]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return powerflux * grx * (
        C_VALUE ** 2 / 4. / np.pi / freq ** 2
        )


def _free_space_loss(d, f):

    return (C_VALUE / 4. / np.pi / f / d) ** 2


@utils.ranged_quantity_input(
    dist=(1.e-30, None, apu.m),
    freq=(1.e-30, None, apu.Hz),
    strip_input_units=True, output_unit=dB
    )
def free_space_loss(dist, freq):
    '''
    Free-space loss of a propagating radio wave.

    Parameters
    ----------
    dist : `~astropy.units.Quantity`
        Distance between transmitter and receiver [m]
    freq : `~astropy.units.Quantity`
        Frequency of radiation [Hz]

    Returns
    -------
    FSPL : `~astropy.units.Quantity`
        Free-space loss [dB]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return 10. * np.log10(_free_space_loss(freq, dist))


@utils.ranged_quantity_input(
    ptx=(1.e-30, None, apu.W),
    gtx=(1.e-30, None, dimless),
    grx=(1.e-30, None, dimless),
    dist=(1.e-30, None, apu.m),
    freq=(1.e-30, None, apu.Hz),
    strip_input_units=True, output_unit=apu.W
    )
def prx_from_ptx(ptx, gtx, grx, dist, freq):
    '''
    Received power from transmitted power.

    Parameters
    ----------
    ptx : `~astropy.units.Quantity`
        Transmitter power, Ptx [dB_W, W]
    gtx : `~astropy.units.Quantity`
        Gain of transmitter, Gtx [dBi, dimless]
    grx : `~astropy.units.Quantity`
        Gain of receiver, Grx [dBi, dimless]
    dist : `~astropy.units.Quantity`
        Distance between transmitter and receiver [m]
    freq : `~astropy.units.Quantity`
        Frequency of radiation [Hz]

    Returns
    -------
    prx : `~astropy.units.Quantity`
        Received power [W]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return ptx * gtx * grx * _free_space_loss(dist, freq)


@utils.ranged_quantity_input(
    prx=(1.e-30, None, apu.W),
    gtx=(1.e-30, None, dimless),
    grx=(1.e-30, None, dimless),
    dist=(1.e-30, None, apu.m),
    freq=(1.e-30, None, apu.Hz),
    strip_input_units=True, output_unit=apu.W
    )
def ptx_from_prx(prx, gtx, grx, dist, freq):
    '''
    Transmitted power from received power.

    Parameters
    ----------
    prx : `~astropy.units.Quantity`
        Received power, Prx [dB_W, W]
    gtx : `~astropy.units.Quantity`
        Gain of transmitter, Gtx [dBi, dimless]
    grx : `~astropy.units.Quantity`
        Gain of receiver, Grx [dBi, dimless]
    dist : `~astropy.units.Quantity`
        Distance between transmitter and receiver [m]
    freq : `~astropy.units.Quantity`
        Frequency of radiation [Hz]

    Returns
    -------
    ptx : `~astropy.units.Quantity`
        Transmitter power, Ptx [W]

    Notes
    -----
    Because all parameters/returned values are Astropy Quantities (see
    `~astropy.units.Quantity`), unit conversion is automatically performed.
    '''

    return prx / gtx / grx / _free_space_loss(dist, freq)


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
