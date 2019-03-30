#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

import os
from functools import partial, lru_cache
import numbers
import collections
import numpy as np
from astropy import units as apu
from astropy.utils.data import get_pkg_data_filename
from .. import conversions as cnv
from .. import utils
from .atm_helper import path_helper_cython, path_endpoint_cython


__all__ = [
    'EARTH_RADIUS',
    'refractive_index', 'saturation_water_pressure',
    'pressure_water_from_humidity', 'humidity_from_pressure_water',
    'pressure_water_from_rho_water', 'rho_water_from_pressure_water',
    'profile_standard', 'profile_lowlat',
    'profile_midlat_summer', 'profile_midlat_winter',
    'profile_highlat_summer', 'profile_highlat_winter',
    'resonances_oxygen', 'resonances_water',
    # 'atten_linear_from_atten_log', 'atten_log_from_atten_linear',
    'elevation_from_airmass', 'airmass_from_elevation',
    'opacity_from_atten', 'atten_from_opacity',
    'atten_specific_annex1',
    'atten_terrestrial', 'atm_layers', 'atten_slant_annex1',
    'atten_specific_annex2',
    'atten_slant_annex2',
    'equivalent_height_dry', 'equivalent_height_wet',
    # '_prepare_path'
    ]


EARTH_RADIUS = 6371.

fname_oxygen = get_pkg_data_filename(
    '../itudata/p.676-10/R-REC-P.676-10-201309_table1.csv'
    )
fname_water = get_pkg_data_filename(
    '../itudata/p.676-10/R-REC-P.676-10-201309_table2.csv'
    )

oxygen_dtype = np.dtype([
    (str(s), np.float64) for s in ['f0', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6']
    ])
water_dtype = np.dtype([
    (str(s), np.float64) for s in ['f0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6']
    ])
resonances_oxygen = np.genfromtxt(
    fname_oxygen, dtype=oxygen_dtype, delimiter=';'
    )
resonances_water = np.genfromtxt(
    fname_water, dtype=water_dtype, delimiter=';'
    )


def _airmass_from_elevation(elev):

    elev = np.array(elev)
    elev_shape = elev.shape
    elev = np.atleast_1d(elev)
    mask = elev < 32
    elev_rad = np.radians(elev)

    airmass = np.empty_like(elev)
    airmass[~mask] = 1 / np.sin(elev_rad[~mask])
    airmass[mask] = -0.02344 + 1.0140 / np.sin(np.radians(
        elev[mask] + 5.18 / (elev[mask] + 3.35)
        ))

    return airmass.reshape(elev_shape)


@utils.ranged_quantity_input(
    elev=(0, 90, apu.deg),
    strip_input_units=True, output_unit=cnv.dimless
    )
def airmass_from_elevation(elev):
    '''
    Airmass derived from elevation using extrapolation by Maddalena & Johnson.

    Parameters
    ----------
    elev : `~astropy.units.Quantity`
        Elevation [deg]

    Returns
    -------
    airmass : `~astropy.units.Quantity`
        Airmass [dimless]

    Notes
    -----
    For low elevations, the well-known 1/sin-law breaks down. Maddalena &
    Johnson (2006) propose the following extrapolation for El < 32 deg:

    .. math::

        \\textrm{AM} = \\begin{cases}
            -0.02344 + \\frac{1.0140}{
                \\sin\\left(\\mathrm{El} + \\frac{5.18}{\\mathrm{El} + 3.35}\\right)}\\qquad\\mathrm{for~}\\mathrm{El}<32\\\\
            \\frac{1}{\\sin\\mathrm{El}}\\qquad\\mathrm{for~}\\mathrm{El}\\geq32
            \\end{cases}
    '''

    return _airmass_from_elevation(elev)


def _elevation_from_airmass(airmass):

    airmass = np.array(airmass)
    airmass_shape = airmass.shape
    airmass = np.atleast_1d(airmass)

    mask = airmass <= 1.887079914799858  # via airmass_from_elevation(32)

    B = np.degrees(np.arcsin(
        1.0140 / (airmass[~mask] + 0.02344)
        ))

    elev = np.empty_like(airmass)
    elev[~mask] = (
        0.5 * B +
        0.025 * np.sqrt(400.0 * B ** 2 + 2680.0 * B - 3799.0) -
        1.675
        )
    elev[mask] = np.degrees(np.arcsin(1 / airmass[mask]))

    return elev.reshape(airmass_shape)


@utils.ranged_quantity_input(
    airmass=(1, None, cnv.dimless),
    strip_input_units=True, output_unit=apu.deg
    )
def elevation_from_airmass(airmass):
    '''
    Airmass derived from elevation using extrapolation by Maddalena & Johnson.

    Parameters
    ----------
    airmass : `~astropy.units.Quantity`
        Airmass [dimless]

    Returns
    -------
    elev : `~astropy.units.Quantity`
        Elevation [deg]

    Notes
    -----
    For low elevations, the well-known 1/sin-law breaks down. Maddalena &
    Johnson (2006) propose the following extrapolation for El (in degrees):

    .. math::

        \\textrm{AM} = \\begin{cases}
            -0.02344 + \\frac{1.0140}{
                \\sin\\left(\\mathrm{El} + \\frac{5.18}{\\mathrm{El} + 3.35}\\right)}\\qquad\\mathrm{for~}\\mathrm{El}<32\\\\
            \\frac{1}{\\sin\\mathrm{El}}\\qquad\\mathrm{for~}\\mathrm{El}\\geq32
            \\end{cases}

    which was simply inverted:

    .. math::

        \\mathrm{El} = \\begin{cases}
            \\frac{B}{2} +
                \\frac{1}{40}\\sqrt{400 B^2 + 2680 B - 3799} - 1.675\\qquad\\mathrm{for~}\\mathrm{AM}\\leq1.88708 \\\\
            \\sin^{-1}\\frac{1}{\\mathrm{AM}}\\qquad\\mathrm{for~}\\mathrm{AM}>1.88708
            \\end{cases}

    where

    .. math::

        B \\equiv \\sin^{-1}\\left(\\frac{1.0140}{\\mathrm{AM} + 0.02344}\\right)
    '''

    return _elevation_from_airmass(airmass)


def _opacity_from_atten(atten, elev=None):

    if elev is None:

        return np.log(atten)

    else:

        return np.log(atten) / _airmass_from_elevation(elev)


@utils.ranged_quantity_input(
    atten=(1.000000000001, None, cnv.dimless), elev=(0, 90, apu.deg),
    strip_input_units=True, allow_none=True, output_unit=cnv.dimless
    )
def opacity_from_atten(atten, elev=None):
    '''
    Atmospheric opacity derived from attenuation.

    Parameters
    ----------
    atten : `~astropy.units.Quantity`
        Atmospheric attenuation [dB or dimless]
    elev : `~astropy.units.Quantity`, optional
        Elevation [deg]
        If not None, this is used to correct for the Airmass
        (via `~pycraf.atm.airmass_from_elevation`; AM = 1 / sin(elev)),
        which means that the zenith opacity is inferred.

    Returns
    -------
    tau : `~astropy.units.Quantity`
        Atmospheric opacity [dimless aka neper]
    '''

    return _opacity_from_atten(atten, elev=elev)


def _atten_from_opacity(tau, elev=None):

    if elev is None:

        return 10 * np.log10(np.exp(tau))

    else:

        return 10 * np.log10(np.exp(tau * _airmass_from_elevation(elev)))


@utils.ranged_quantity_input(
    tau=(0.000000000001, None, cnv.dimless), elev=(0, 90, apu.deg),
    strip_input_units=True, allow_none=True, output_unit=cnv.dB
    )
def atten_from_opacity(tau, elev=None):
    '''
    Atmospheric attenuation derived from opacity.

    Parameters
    ----------
    tau : `~astropy.units.Quantity`
        Atmospheric opacity [dimless aka neper]
    elev : `~astropy.units.Quantity`, optional
        Elevation [deg]
        If not None, this is used to correct for the Airmass
        (via `~pycraf.atm.airmass_from_elevation`; AM = 1 / sin(elev)),
        which means that the input opacity is treated as zenith opacity.

    Returns
    -------
    atten : `~astropy.units.Quantity`
        Atmospheric attenuation [dB or dimless]
    '''

    return _atten_from_opacity(tau, elev=elev)


def _refractive_index(temp, press, press_w):

    return (
        1 + 1e-6 / temp * (
            77.6 * press - 5.6 * press_w +
            3.75e5 * press_w / temp
            )
        )


@utils.ranged_quantity_input(
    temp=(1.e-30, None, apu.K),
    press=(1.e-30, None, apu.hPa),
    press_w=(1.e-30, None, apu.hPa),
    strip_input_units=True, output_unit=cnv.dimless
    )
def refractive_index(temp, press, press_w):
    '''
    Refractive index according to `ITU-R P.453-12
    <https://www.itu.int/rec/R-REC-P.453-12-201609-I/en>`_.

    Parameters
    ----------
    temp : `~astropy.units.Quantity`
        Air temperature [K]
    press : `~astropy.units.Quantity`
        Total air pressure [hPa]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]

    Returns
    -------
    n_index : `~astropy.units.Quantity`
        Refractive index [dimless]
    '''

    return _refractive_index(temp, press, press_w)


def _saturation_water_pressure(temp, press, wet_type):
    # temp_C is temperature in Celcius
    temp_C = temp - 273.15

    assert wet_type in ['water', 'ice']

    EF = (
        1. + 1.e-4 * (7.2 + press * (0.0320 + 5.9e-6 * temp_C ** 2))
        if wet_type == 'water' else
        1. + 1.e-4 * (2.2 + press * (0.0382 + 6.4e-6 * temp_C ** 2))
        )

    a, b, c, d = (
        (6.1121, 18.678, 257.14, 234.5)
        if wet_type == 'water' else
        (6.1115, 23.036, 279.82, 333.7)
        )
    e_s = EF * a * np.exp((b - temp_C / d) * temp_C / (c + temp_C))

    return e_s


@utils.ranged_quantity_input(
    temp=(1.e-30, None, apu.K),
    press=(1.e-30, None, apu.hPa),
    strip_input_units=True, output_unit=apu.hPa
    )
def saturation_water_pressure(temp, press, wet_type='water'):
    '''
    Saturation water pressure according to `ITU-R P.453-12
    <https://www.itu.int/rec/R-REC-P.453-12-201609-I/en>`_.

    Parameters
    ----------
    temp : `~astropy.units.Quantity`
        Air temperature [K]
    press : `~astropy.units.Quantity`
        Total air pressure [hPa]
    wet_type : str, optional
        Type of wet material: 'water', 'ice'

    Returns
    -------
    press_sat : `~astropy.units.Quantity`
        Saturation water vapor pressure, e_s [hPa]
    '''

    return _saturation_water_pressure(
        temp, press, wet_type=wet_type
        )


def _pressure_water_from_humidity(
        temp, press, humidity, wet_type='water'
        ):

    e_s = _saturation_water_pressure(
        temp, press, wet_type=wet_type
        )

    press_w = humidity / 100. * e_s

    return press_w


@utils.ranged_quantity_input(
    temp=(1.e-30, None, apu.K),
    press=(1.e-30, None, apu.hPa),
    humidity=(0, 100, apu.percent),
    strip_input_units=True, output_unit=apu.hPa
    )
def pressure_water_from_humidity(
        temp, press, humidity, wet_type='water'
        ):
    '''
    Water pressure according to `ITU-R P.453-12
    <https://www.itu.int/rec/R-REC-P.453-12-201609-I/en>`_.

    Parameters
    ----------
    temp : `~astropy.units.Quantity`
        Air temperature [K]
    press : `~astropy.units.Quantity`
        Total air pressure [hPa]
    humidity : `~astropy.units.Quantity`
        Relative humidity [%]
    wet_type : str, optional
        Type of wet material: 'water', 'ice'

    Returns
    -------
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    '''

    return _pressure_water_from_humidity(
        temp, press, humidity, wet_type=wet_type
        )


def _humidity_from_pressure_water(
        temp, press, press_w, wet_type='water'
        ):

    e_s = _saturation_water_pressure(
        temp, press, wet_type=wet_type
        )

    humidity = 100. * press_w / e_s

    return humidity


@utils.ranged_quantity_input(
    temp=(1.e-30, None, apu.K),
    press=(1.e-30, None, apu.hPa),
    press_w=(1.e-30, None, apu.hPa),
    strip_input_units=True, output_unit=apu.percent
    )
def humidity_from_pressure_water(
        temp, press, press_w, wet_type='water'
        ):
    '''
    Relative humidity according to `ITU-R P.453-12
    <https://www.itu.int/rec/R-REC-P.453-12-201609-I/en>`_.

    wet_type - 'water' or 'ice'

    Parameters
    ----------
    temp : `~astropy.units.Quantity`
        Air temperature [K]
    press : `~astropy.units.Quantity`
        Total air pressure [hPa]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    wet_type : str, optional
        Type of wet material: 'water', 'ice'

    Returns
    -------
    humidity : `~astropy.units.Quantity`
        Relative humidity [%]
    '''

    return _humidity_from_pressure_water(
        temp, press, press_w, wet_type=wet_type
        )


def _pressure_water_from_rho_water(temp, rho_w):

    press_w = rho_w * temp / 216.7

    return press_w


@utils.ranged_quantity_input(
    temp=(1.e-30, None, apu.K),
    rho_w=(1.e-30, None, apu.g / apu.m ** 3),
    strip_input_units=True, output_unit=apu.hPa
    )
def pressure_water_from_rho_water(temp, rho_w):
    '''
    Water pressure according to `ITU-R P.453-12
    <https://www.itu.int/rec/R-REC-P.453-12-201609-I/en>`_.

    Parameters
    ----------
    temp : `~astropy.units.Quantity`
        Air temperature [K]
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m**3]

    Returns
    -------
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    '''

    return _pressure_water_from_rho_water(temp, rho_w)


def _rho_water_from_pressure_water(temp, press_w):

    rho_w = press_w * 216.7 / temp

    return rho_w


@utils.ranged_quantity_input(
    temp=(1.e-30, None, apu.K),
    press_w=(1.e-30, None, apu.hPa),
    strip_input_units=True, output_unit=apu.g / apu.m ** 3
    )
def rho_water_from_pressure_water(temp, press_w):
    '''
    Water density according to `ITU-R P.453-12
    <https://www.itu.int/rec/R-REC-P.453-12-201609-I/en>`_.

    Parameters
    ----------
    temp : `~astropy.units.Quantity`
        Air temperature [K]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]

    Returns
    -------
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m**3]
    '''

    return _rho_water_from_pressure_water(temp, press_w)


def _profile_standard(height):

    # height = np.asarray(height)  # this is not sufficient for masking :-(
    height = np.atleast_1d(height)
    _height = height.flatten()

    # to make this work with numpy arrays
    # lets first find the correct index for every height

    layer_heights = np.array([0., 11., 20., 32., 47., 51., 71., 85.])
    indices = np.zeros(_height.size, dtype=np.int32)
    for i, lh in enumerate(layer_heights[0:-1]):
        indices[_height > lh] = i

    T0 = 288.15  # K
    P0 = 1013.25  # hPa
    rho0 = 7.5  # g / m^3
    h0 = 2.  # km
    layer_temp_gradients = np.array([-6.5, 0., 1., 2.8, 0., -2.8, -2.])
    layer_start_temperatures = [T0]
    layer_start_pressures = [P0]
    for i in range(1, len(layer_heights) - 1):

        dh = layer_heights[i] - layer_heights[i - 1]
        Ti = layer_start_temperatures[i - 1]
        Li = layer_temp_gradients[i - 1]
        Pi = layer_start_pressures[i - 1]
        # print(i, Ti, Li, Pi, i - 1 not in [1, 4])
        layer_start_temperatures.append(Ti + dh * Li)
        layer_start_pressures.append(
            Pi * (
                (Ti / (Ti + Li * dh)) ** (34.163 / Li)
                if i - 1 not in [1, 4] else
                np.exp(-34.163 * dh / Ti)
                )
            )
    layer_start_temperatures = np.array(layer_start_temperatures)
    layer_start_pressures = np.array(layer_start_pressures)

    temperatures = (
        layer_start_temperatures[indices] +
        (_height - layer_heights[indices]) * layer_temp_gradients[indices]
        )

    pressures = np.empty_like(temperatures)

    # gradient zero
    mask = np.in1d(indices, [1, 4])
    indm = indices[mask]
    dhm = (_height[mask] - layer_heights[indm])
    pressures[mask] = (
        layer_start_pressures[indm] *
        np.exp(-34.163 * dhm / layer_start_temperatures[indm])
        )

    # gradient non-zero
    mask = np.logical_not(mask)
    indm = indices[mask]
    dhm = (_height[mask] - layer_heights[indm])
    Lim = layer_temp_gradients[indm]
    Tim = layer_start_temperatures[indm]
    pressures[mask] = (
        layer_start_pressures[indm] *
        (Tim / (Tim + Lim * dhm)) ** (34.163 / Lim)
        )

    rho_water = rho0 * np.exp(-_height / h0)
    pressures_water = rho_water * temperatures / 216.7
    mask = (pressures_water / pressures) < 2.e-6
    pressures_water[mask] = pressures[mask] * 2.e-6
    rho_water[mask] = pressures_water[mask] / temperatures[mask] * 216.7

    ref_indices = _refractive_index(temperatures, pressures, pressures_water)
    humidities_water = _humidity_from_pressure_water(
        temperatures, pressures, pressures_water, wet_type='water'
        )
    humidities_ice = _humidity_from_pressure_water(
        temperatures, pressures, pressures_water, wet_type='ice'
        )

    result = (
        temperatures.reshape(height.shape).squeeze(),
        pressures.reshape(height.shape).squeeze(),
        rho_water.reshape(height.shape).squeeze(),
        pressures_water.reshape(height.shape).squeeze(),
        ref_indices.reshape(height.shape).squeeze(),
        humidities_water.reshape(height.shape).squeeze(),
        humidities_ice.reshape(height.shape).squeeze(),
        )

    # return tuple(v.reshape(height.shape) for v in result)
    return result


@utils.ranged_quantity_input(
    height=(0, 84.99999999, apu.km),
    strip_input_units=True,
    output_unit=(
        apu.K, apu.hPa, apu.g / apu.m ** 3, apu.hPa,
        cnv.dimless, apu.percent, apu.percent
        ),
    )
def profile_standard(height):
    '''
    Standard height profiles according to `ITU-R P.835-5
    <https://www.itu.int/rec/R-REC-P.835-5-201202-I/en>`_, Annex 1.

    Parameters
    ----------
    height : `~astropy.units.Quantity`
        Height above ground [km]

    Returns
    -------
    temp : `~astropy.units.Quantity`
        Temperature [K]
    press : `~astropy.units.Quantity`
        Total pressure [hPa]
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m**3]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    n_index : `~astropy.units.Quantity`
        Refractive index [dimless]
    humidity_water : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of liquid water [%]
    humidity_ice : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of ice [%]

    Notes
    -----
    For convenience, derived quantities like water density/pressure
    and refraction indices are also returned.
    '''

    return _profile_standard(height)


def _profile_helper(
        height,
        temp_heights, temp_funcs,
        press_heights, press_funcs,
        rho_heights, rho_funcs,
        ):
    '''
    Helper function for specialized profiles.

    Parameters
    ----------
    height :`~numpy.ndarray` of `~numpy.float`
        Height above ground [km]
    {temp,press,rho}_heights : list of floats
        Height steps for which piece-wise functions are defined
    {temp,press,rho}_funcs - list of functions
        Functions that return the desired quantity for a given height interval

    Returns
    -------
    temp : `~numpy.ndarray` of `~numpy.float`
        Temperature [K]
    press : `~numpy.ndarray` of `~numpy.float`
        Total pressure [hPa]
    rho_w : `~numpy.ndarray` of `~numpy.float`
        Water vapor density [g / m**3]
    press_w : `~numpy.ndarray` of `~numpy.float`
        Water vapor partial pressure [hPa]
    n_index : `~numpy.ndarray` of `~numpy.float`
        Refractive index [dimless]
    humidity_water : `~numpy.ndarray` of `~numpy.float`
        Relative humidity if water vapor was in form of liquid water [%]
    humidity_ice : `~numpy.ndarray` of `~numpy.float`
        Relative humidity if water vapor was in form of ice [%]
    '''

    height = np.atleast_1d(height)

    # assert np.all(height < temp_heights[-1]), (
    #     'profile only defined below {} km height!'.format(temp_heights[-1])
    #     )
    # assert np.all(height >= temp_heights[0]), (
    #     'profile only defined above {} km height!'.format(temp_heights[0])
    #     )

    temperature = np.empty(height.shape, dtype=np.float64)
    pressure = np.empty(height.shape, dtype=np.float64)
    pressure_water = np.empty(height.shape, dtype=np.float64)
    rho_water = np.empty(height.shape, dtype=np.float64)

    Pstarts = [None]
    for i in range(1, len(press_heights) - 1):
        Pstarts.append(press_funcs[i - 1](Pstarts[-1], press_heights[i]))

    # calculate temperature profile
    for i in range(len(temp_heights) - 1):
        hmin, hmax = temp_heights[i], temp_heights[i + 1]
        mask = (height >= hmin) & (height < hmax)
        temperature[mask] = (temp_funcs[i])(height[mask])

    # calculate pressure profile
    for i in range(len(press_heights) - 1):
        hmin, hmax = press_heights[i], press_heights[i + 1]
        mask = (height >= hmin) & (height < hmax)
        pressure[mask] = (press_funcs[i])(Pstarts[i], height[mask])

    # calculate rho profile
    for i in range(len(rho_heights) - 1):
        hmin, hmax = rho_heights[i], rho_heights[i + 1]
        mask = (height >= hmin) & (height < hmax)
        rho_water[mask] = (rho_funcs[i])(height[mask])

    # calculate pressure_water profile
    pressure_water = rho_water * temperature / 216.7
    mask = (pressure_water / pressure) < 2.e-6
    pressure_water[mask] = pressure[mask] * 2.e-6
    rho_water[mask] = pressure_water[mask] / temperature[mask] * 216.7

    ref_index = _refractive_index(temperature, pressure, pressure_water)
    humidity_water = _humidity_from_pressure_water(
        temperature, pressure, pressure_water, wet_type='water'
        )
    humidity_ice = _humidity_from_pressure_water(
        temperature, pressure, pressure_water, wet_type='ice'
        )

    return (
        temperature.reshape(height.shape).squeeze(),
        pressure.reshape(height.shape).squeeze(),
        rho_water.reshape(height.shape).squeeze(),
        pressure_water.reshape(height.shape).squeeze(),
        ref_index.reshape(height.shape).squeeze(),
        humidity_water.reshape(height.shape).squeeze(),
        humidity_ice.reshape(height.shape).squeeze(),
        )


@utils.ranged_quantity_input(
    height=(0, 99.99999999, apu.km),
    strip_input_units=True,
    output_unit=(
        apu.K, apu.hPa, apu.g / apu.m ** 3, apu.hPa,
        cnv.dimless, apu.percent, apu.percent
        ),
    )
def profile_lowlat(height):
    '''
    Low latitude height profiles according to `ITU-R P.835-5
    <https://www.itu.int/rec/R-REC-P.835-5-201202-I/en>`_.

    Valid for geographic latitudes :math:`\\vert \\phi\\vert < 22^\\circ`.

    Parameters
    ----------
    height : `~astropy.units.Quantity`
        Height above ground [km]

    Returns
    -------
    temp : `~astropy.units.Quantity`
        Temperature [K]
    press : `~astropy.units.Quantity`
        Total pressure [hPa]
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m**3]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    n_index : `~astropy.units.Quantity`
        Refractive index [dimless]
    humidity_water : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of liquid water [%]
    humidity_ice : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of ice [%]
    '''

    temp_heights = [0., 17., 47., 52., 80., 100.]
    temp_funcs = [
        lambda h: 300.4222 - 6.3533 * h + 0.005886 * h ** 2,
        lambda h: 194 + (h - 17.) * 2.533,
        lambda h: 270.,
        lambda h: 270. - (h - 52.) * 3.0714,
        lambda h: 184.,
        ]

    press_heights = [0., 10., 72., 100.]
    press_funcs = [
        lambda Pstart, h: 1012.0306 - 109.0338 * h + 3.6316 * h ** 2,
        lambda Pstart, h: Pstart * np.exp(-0.147 * (h - 10)),
        lambda Pstart, h: Pstart * np.exp(-0.165 * (h - 72)),
        ]

    rho_heights = [0., 15., 100.]
    rho_funcs = [
        lambda h: 19.6542 * np.exp(
            -0.2313 * h -
            0.1122 * h ** 2 +
            0.01351 * h ** 3 -
            0.0005923 * h ** 4
            ),
        lambda h: 0.,
        ]

    return _profile_helper(
        height,
        temp_heights, temp_funcs,
        press_heights, press_funcs,
        rho_heights, rho_funcs,
        )


@utils.ranged_quantity_input(
    height=(0, 99.99999999, apu.km),
    strip_input_units=True,
    output_unit=(
        apu.K, apu.hPa, apu.g / apu.m ** 3, apu.hPa,
        cnv.dimless, apu.percent, apu.percent
        ),
    )
def profile_midlat_summer(height):
    '''
    Mid latitude summer height profiles according to `ITU-R P.835-5
    <https://www.itu.int/rec/R-REC-P.835-5-201202-I/en>`_.

    Valid for geographic latitudes :math:`22^\\circ < \\vert \\phi\\vert < 45^\\circ`.

    Parameters
    ----------
    height : `~astropy.units.Quantity`
        Height above ground [km]

    Returns
    -------
    temp : `~astropy.units.Quantity`
        Temperature [K]
    press : `~astropy.units.Quantity`
        Total pressure [hPa]
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m**3]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    n_index : `~astropy.units.Quantity`
        Refractive index [dimless]
    humidity_water : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of liquid water [%]
    humidity_ice : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of ice [%]
    '''

    temp_heights = [0., 13., 17., 47., 53., 80., 100.]
    temp_funcs = [
        lambda h: 294.9838 - 5.2159 * h - 0.07109 * h ** 2,
        lambda h: 215.15,
        lambda h: 215.15 * np.exp(0.008128 * (h - 17.)),
        lambda h: 275.,
        lambda h: 275. + 20. * (1. - np.exp(0.06 * (h - 53.))),
        lambda h: 175.,
        ]

    press_heights = [0., 10., 72., 100.]
    press_funcs = [
        lambda Pstart, h: 1012.8186 - 111.5569 * h + 3.8646 * h ** 2,
        lambda Pstart, h: Pstart * np.exp(-0.147 * (h - 10)),
        lambda Pstart, h: Pstart * np.exp(-0.165 * (h - 72)),
        ]

    rho_heights = [0., 15., 100.]
    rho_funcs = [
        lambda h: 14.3542 * np.exp(
            -0.4174 * h - 0.02290 * h ** 2 + 0.001007 * h ** 3
            ),
        lambda h: 0.,
        ]

    return _profile_helper(
        height,
        temp_heights, temp_funcs,
        press_heights, press_funcs,
        rho_heights, rho_funcs,
        )


@utils.ranged_quantity_input(
    height=(0, 99.99999999, apu.km),
    strip_input_units=True,
    output_unit=(
        apu.K, apu.hPa, apu.g / apu.m ** 3, apu.hPa,
        cnv.dimless, apu.percent, apu.percent
        ),
    )
def profile_midlat_winter(height):
    '''
    Mid latitude winter height profiles according to `ITU-R P.835-5
    <https://www.itu.int/rec/R-REC-P.835-5-201202-I/en>`_.

    Valid for geographic latitudes :math:`22^\\circ < \\vert \\phi\\vert < 45^\\circ`.

    Parameters
    ----------
    height : `~astropy.units.Quantity`
        Height above ground [km]

    Returns
    -------
    temp : `~astropy.units.Quantity`
        Temperature [K]
    press : `~astropy.units.Quantity`
        Total pressure [hPa]
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m**3]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    n_index : `~astropy.units.Quantity`
        Refractive index [dimless]
    humidity_water : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of liquid water [%]
    humidity_ice : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of ice [%]
    '''

    temp_heights = [0., 10., 33., 47., 53., 80., 100.]
    temp_funcs = [
        lambda h: 272.7241 - 3.6517 * h - 0.1759 * h ** 2,
        lambda h: 218.,
        lambda h: 218. + 3.3571 * (h - 33.),
        lambda h: 265.,
        lambda h: 265. - 2.0370 * (h - 53.),
        lambda h: 210.,
        ]

    press_heights = [0., 10., 72., 100.]
    press_funcs = [
        lambda Pstart, h: 1018.8627 - 124.2954 * h + 4.8307 * h ** 2,
        lambda Pstart, h: Pstart * np.exp(-0.147 * (h - 10)),
        lambda Pstart, h: Pstart * np.exp(-0.155 * (h - 72)),
        ]

    rho_heights = [0., 10., 100.]
    rho_funcs = [
        lambda h: 3.4742 * np.exp(
            -0.2697 * h - 0.03604 * h ** 2 + 0.0004489 * h ** 3
            ),
        lambda h: 0.,
        ]

    return _profile_helper(
        height,
        temp_heights, temp_funcs,
        press_heights, press_funcs,
        rho_heights, rho_funcs,
        )


@utils.ranged_quantity_input(
    height=(0, 99.99999999, apu.km),
    strip_input_units=True,
    output_unit=(
        apu.K, apu.hPa, apu.g / apu.m ** 3, apu.hPa,
        cnv.dimless, apu.percent, apu.percent
        ),
    )
def profile_highlat_summer(height):
    '''
    High latitude summer height profiles according to `ITU-R P.835-5
    <https://www.itu.int/rec/R-REC-P.835-5-201202-I/en>`_.

    Valid for geographic latitudes :math:`\\vert \\phi\\vert > 45^\\circ`.

    Parameters
    ----------
    height : `~astropy.units.Quantity`
        Height above ground [km]

    Returns
    -------
    temp : `~astropy.units.Quantity`
        Temperature [K]
    press : `~astropy.units.Quantity`
        Total pressure [hPa]
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m**3]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    n_index : `~astropy.units.Quantity`
        Refractive index [dimless]
    humidity_water : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of liquid water [%]
    humidity_ice : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of ice [%]
    '''

    temp_heights = [0., 10., 23., 48., 53., 79., 100.]
    temp_funcs = [
        lambda h: 286.8374 - 4.7805 * h - 0.1402 * h ** 2,
        lambda h: 225.,
        lambda h: 225. * np.exp(0.008317 * (h - 23.)),
        lambda h: 277.,
        lambda h: 277. - 4.0769 * (h - 53.),
        lambda h: 171.,
        ]

    press_heights = [0., 10., 72., 100.]
    press_funcs = [
        lambda Pstart, h: 1008.0278 - 113.2494 * h + 3.9408 * h ** 2,
        lambda Pstart, h: Pstart * np.exp(-0.147 * (h - 10)),
        lambda Pstart, h: Pstart * np.exp(-0.165 * (h - 72)),
        ]

    rho_heights = [0., 15., 100.]
    rho_funcs = [
        lambda h: 8.988 * np.exp(
            -0.3614 * h - 0.005402 * h ** 2 - 0.001955 * h ** 3
            ),
        lambda h: 0.,
        ]

    return _profile_helper(
        height,
        temp_heights, temp_funcs,
        press_heights, press_funcs,
        rho_heights, rho_funcs,
        )


@utils.ranged_quantity_input(
    height=(0, 99.99999999, apu.km),
    strip_input_units=True,
    output_unit=(
        apu.K, apu.hPa, apu.g / apu.m ** 3, apu.hPa,
        cnv.dimless, apu.percent, apu.percent
        ),
    )
def profile_highlat_winter(height):
    '''
    High latitude winter height profiles according to `ITU-R P.835-5
    <https://www.itu.int/rec/R-REC-P.835-5-201202-I/en>`_.

    Valid for geographic latitudes :math:`\\vert \\phi\\vert > 45^\\circ`.

    Parameters
    ----------
    height : `~astropy.units.Quantity`
        Height above ground [km]

    Returns
    -------
    temp : `~astropy.units.Quantity`
        Temperature [K]
    press : `~astropy.units.Quantity`
        Total pressure [hPa]
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m**3]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    n_index : `~astropy.units.Quantity`
        Refractive index [dimless]
    humidity_water : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of liquid water [%]
    humidity_ice : `~astropy.units.Quantity`
        Relative humidity if water vapor was in form of ice [%]
    '''

    temp_heights = [0., 8.5, 30., 50., 54., 100.]
    temp_funcs = [
        lambda h: 257.4345 + 2.3474 * h - 1.5479 * h ** 2 + 0.08473 * h ** 3,
        lambda h: 217.5,
        lambda h: 217.5 + 2.125 * (h - 30.),
        lambda h: 260.,
        lambda h: 260. - 1.667 * (h - 54.),
        ]

    press_heights = [0., 10., 72., 100.]
    press_funcs = [
        lambda Pstart, h: 1010.8828 - 122.2411 * h + 4.554 * h ** 2,
        lambda Pstart, h: Pstart * np.exp(-0.147 * (h - 10)),
        lambda Pstart, h: Pstart * np.exp(-0.150 * (h - 72)),
        ]

    rho_heights = [0., 10., 100.]
    rho_funcs = [
        lambda h: 1.2319 * np.exp(
            0.07481 * h - 0.0981 * h ** 2 + 0.00281 * h ** 3
            ),
        lambda h: 0.,
        ]

    return _profile_helper(
        height,
        temp_heights, temp_funcs,
        press_heights, press_funcs,
        rho_heights, rho_funcs,
        )


def _S_oxygen(press_dry, temp):
    '''
    Line strengths of all oxygen resonances according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Eq (3).

    Parameters
    ----------
    press_dry : `numpy.ndarray`, float
        Dry air (=Oxygen) pressure [hPa]
    temp : `numpy.ndarray`, float
        Temperature [K]

    Returns
    -------
    I : `numpy.ndarray`, float
        Line strength

    Notes
    -----
    Total pressure: `press = press_dry + press_w`
    '''

    theta = 300. / temp
    factor = 1.e-7 * press_dry * theta ** 3

    return (
        resonances_oxygen['a1'] *
        factor *
        np.exp(resonances_oxygen['a2'] * (1. - theta))
        )


def _S_water(press_w, temp):
    '''
    Line strengths of all water resonances according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Eq (3).

    Parameters
    ----------
    press_w : `numpy.ndarray`, float
        Water vapor partial pressure [hPa]
    temp : `numpy.ndarray`, float
        Temperature [K]

    Returns
    -------
    I : `numpy.ndarray`, float
        Line strength

    Notes
    -----
    Total pressure: `press = press_dry + press_w`
    '''

    theta = 300. / temp
    factor = 1.e-1 * press_w * theta ** 3.5

    return (
        resonances_water['b1'] *
        factor *
        np.exp(resonances_water['b2'] * (1. - theta))
        )


def _Delta_f_oxygen(press_dry, press_w, temp):
    '''
    Line widths for all oxygen resonances according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Eq (6a/b).

    Parameters
    ----------
    press_dry : `numpy.ndarray`, float
        Dry air (=Oxygen) pressure [hPa]
    press_w : `numpy.ndarray`, float
        Water vapor partial pressure [hPa]
    temp : `numpy.ndarray`, float
        Temperature [K]

    Returns
    -------
    W : `numpy.ndarray`, float
        Line widths for all oxygen resonances

    Notes
    -----
    Oxygen resonance line widths also depend on wet-air pressure.
    '''

    theta = 300. / temp
    df = resonances_oxygen['a3'] * 1.e-4 * (
        press_dry * theta ** (0.8 - resonances_oxygen['a4']) +
        1.1 * press_w * theta
        )
    return np.sqrt(df ** 2 + 2.25e-6)


def _Delta_f_water(press_dry, press_w, temp):
    '''
    Line widths for all water resonances according to
    `ITU-R P.676-10 <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_,
    Eq (6a/b).

    Parameters
    ----------
    press_dry : `numpy.ndarray`, float
        Dry air (=Oxygen) pressure [hPa]
    press_w : `numpy.ndarray`, float
        Water vapor partial pressure [hPa]
    temp : `numpy.ndarray`, float
        Temperature [K]

    Returns
    -------
    W : `numpy.ndarray`, float
        Line widths for all water resonances [GHz]

    Notes
    -----
    Water resonance line widths also depend on dry-air pressure.
    '''

    theta = 300. / temp
    f0, b3, b4, b5, b6 = (
        resonances_water[b]
        for b in ['f0', 'b3', 'b4', 'b5', 'b6']
        )

    df = b3 * 1.e-4 * (
        press_dry * theta ** b4 +
        b5 * press_w * theta ** b6
        )
    return 0.535 * df + np.sqrt(
        0.217 * df ** 2 + 2.1316e-12 * f0 ** 2 / theta
        )


def _delta_oxygen(press_dry, press_w, temp):
    '''
    Shape correction for all oxygen resonances according to
    `ITU-R P.676-10 <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_,
    Eq (7).

    Parameters
    ----------
    press_dry : `numpy.ndarray`, float
        Dry air (=Oxygen) pressure [hPa]
    press_w : `numpy.ndarray`, float
        Water vapor partial pressure [hPa]
    temp : `numpy.ndarray`, float
        Temperature [K]

    Returns
    -------
    delta : `numpy.ndarray`, float
        Profile shape correction factors for all oxygen resonances

    Notes
    -----
    This function accounts for interference effects in oxygen lines.
    '''

    theta = 300. / temp
    return (
        (resonances_oxygen['a5'] + resonances_oxygen['a6'] * theta) * 1.e-4 *
        (press_dry + press_w) * theta ** 0.8
        )


def _delta_water():
    '''
    Shape correction factor for all water vapor resonances (all-zero).

    Returns
    -------
    delta : `numpy.ndarray`, float
        Profile shape correction factors for all water resonances.

    Notes
    -----
    This is only introduced to be able to use the same machinery that is
    working with oxygen, in the `_delta_oxygen` function.
    '''

    return np.zeros(len(resonances_water['f0']), dtype=np.float64)


def _F(freq_grid, f_i, Delta_f, delta):
    '''
    Line-profiles for all resonances at the freq_grid positions according to
    `ITU-R P.676-10 <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_,
    Eq (5).

    Parameters
    ----------
    freq_grid : `numpy.ndarray` of float
        Frequencies at which to calculate line-width shapes [GHz]
    f_i : `numpy.ndarray` of float
        Resonance line frequencies [GHz]
    Delta_f : `numpy.ndarray` of float
        Line widths of all resonances [GHz]
    delta : `numpy.ndarray` of float
        Correction factors to account for interference effects in oxygen
        lines

    Returns
    -------
    S : `numpy.ndarray` of float (m, n)
        Line-shape values, (`n = len(freq_grid)`, `m = len(Delta_f)`)

    Notes
    -----
    No integration is done between `freq_grid` positions, so if you're
    interested in high accuracy near resonance lines, make your `freq_grid`
    sufficiently fine.
    '''

    _freq_grid = freq_grid[np.newaxis]
    _f_i = f_i[:, np.newaxis]
    _Delta_f = Delta_f[:, np.newaxis]
    _delta = delta[:, np.newaxis]

    _df_plus, _df_minus = _f_i + _freq_grid, _f_i - _freq_grid

    sum_1 = (_Delta_f - _delta * _df_minus) / (_df_minus ** 2 + _Delta_f ** 2)
    sum_2 = (_Delta_f - _delta * _df_plus) / (_df_plus ** 2 + _Delta_f ** 2)

    return _freq_grid / _f_i * (sum_1 + sum_2)


def _N_D_prime2(freq_grid, press_dry, press_w, temp):
    '''
    Dry air continuum absorption (Debye spectrum) according to
    `ITU-R P.676-10 <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_,
    Eq (8/9).

    Parameters
    ----------
    freq_grid : `numpy.ndarray`, float
        Frequencies at which to calculate line-width shapes [GHz]
    press_dry : `numpy.ndarray`, float
        Dry air (=Oxygen) pressure [hPa]
    press_w : `numpy.ndarray`, float
        Water vapor partial pressure [hPa]
    temp : `numpy.ndarray`, float
        Temperature [K]

    Returns
    -------
    deb_spec : `numpy.ndarray`, float
       Debye absorption spectrum [dB / km]
    '''

    theta = 300. / temp
    d = 5.6e-4 * (press_dry + press_w) * theta ** 0.8

    sum_1 = 6.14e-5 / d / (1 + (freq_grid / d) ** 2)
    sum_2 = 1.4e-12 * press_dry * theta ** 1.5 / (
        1 + 1.9e-5 * freq_grid ** 1.5
        )

    return freq_grid * press_dry * theta ** 2 * (sum_1 + sum_2)


def _atten_specific_annex1(
        freq_grid, press_dry, press_w, temp
        ):

    freq_grid = np.atleast_1d(freq_grid)

    if not isinstance(press_dry, numbers.Real):
        raise TypeError('press_dry must be a scalar float')
    if not isinstance(press_w, numbers.Real):
        raise TypeError('press_w must be a scalar float')
    if not isinstance(temp, numbers.Real):
        raise TypeError('temp must be a scalar float')

    # first calculate dry attenuation (oxygen lines + N_D_prime2)
    S_o2 = _S_oxygen(press_dry, temp)
    f_i = resonances_oxygen['f0']
    Delta_f = _Delta_f_oxygen(press_dry, press_w, temp)
    delta = _delta_oxygen(press_dry, press_w, temp)
    F_o2 = _F(freq_grid, f_i, Delta_f, delta)

    atten_o2 = np.sum(S_o2[:, np.newaxis] * F_o2, axis=0)
    atten_o2 += _N_D_prime2(
        freq_grid, press_dry, press_w, temp
        )

    # now, wet contribution
    S_h2o = _S_water(press_w, temp)
    f_i = resonances_water['f0']
    Delta_f = _Delta_f_water(press_dry, press_w, temp)
    delta = _delta_water()
    F_h2o = _F(freq_grid, f_i, Delta_f, delta)

    atten_h2o = np.sum(S_h2o[:, np.newaxis] * F_h2o, axis=0)

    return atten_o2 * 0.182 * freq_grid, atten_h2o * 0.182 * freq_grid


@utils.ranged_quantity_input(
    freq_grid=(1.e-30, 1000, apu.GHz),
    press_dry=(1.e-30, None, apu.hPa),
    press_w=(1.e-30, None, apu.hPa),
    temp=(1.e-30, None, apu.K),
    strip_input_units=True, output_unit=(cnv.dB / apu.km, cnv.dB / apu.km)
    )
def atten_specific_annex1(
        freq_grid, press_dry, press_w, temp
        ):
    '''
    Specific (one layer) atmospheric attenuation according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Annex 1.

    Parameters
    ----------
    freq_grid : `~astropy.units.Quantity`
        Frequencies at which to calculate line-width shapes [GHz]
    press_dry : `~astropy.units.Quantity`
        Dry air (=Oxygen) pressure [hPa]
    press_w : `~astropy.units.Quantity`
        Water vapor partial pressure [hPa]
    temp : `~astropy.units.Quantity`
        Temperature [K]

    Returns
    -------
    atten_dry : `~astropy.units.Quantity`
        Dry-air specific attenuation [dB / km]
    atten_wet : `~astropy.units.Quantity`
        Wet-air specific attenuation [dB / km]

    Notes
    -----
    No integration is done between `freq_grid` positions, so if you're
    interested in high accuracy near resonance lines, make your `freq_grid`
    sufficiently fine.
    '''

    return _atten_specific_annex1(
        freq_grid, press_dry, press_w, temp
        )


@utils.ranged_quantity_input(
    specific_atten=(1.e-30, None, cnv.dB / apu.km),
    path_length=(1.e-30, None, apu.km),
    strip_input_units=True, output_unit=cnv.dB
    )
def atten_terrestrial(specific_atten, path_length):
    '''
    Total path attenuation for a path close to the ground (i.e., one layer),
    according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Annex 1 + 2.

    Parameters
    ----------
    specific_atten : `~astropy.units.Quantity`
        Specific attenuation (dry + wet) [dB / km]
    path_length : `~astropy.units.Quantity`
        Length of path [km]

    Returns
    -------
    total_atten : `~astropy.units.Quantity`
        Total attenuation along path [dB]
    '''

    return specific_atten * path_length


@utils.ranged_quantity_input(
    freq_grid=(1.e-30, 1000, apu.GHz),
    heights=(0, 80, apu.km),
    strip_input_units=True, allow_none=True, output_unit=None
    )
def atm_layers(freq_grid, profile_func, heights=None):
    '''
    Calculate physical parameters for atmospheric layers to be used with
    `~pycraf.atm.atten_slant_annex1`.

    This can be used to cache layer-profile data. Since it is only dependent
    on frequency, one can re-use it to save computing time when doing batch
    jobs (e.g., atmospheric dampening for each pixel in a map).

    Parameters
    ----------
    freq_grid : `~astropy.units.Quantity`
        Frequencies at which to calculate line-width shapes [GHz]
    profile_func : func
        A height profile function having the same signature as
        `~pycraf.atm.profile_standard`

    Returns
    -------
    '''

    freq_grid = np.atleast_1d(freq_grid)
    if freq_grid.ndim != 1:
        raise ValueError("'freq_grid' must be a 1D array or scalar")

    if heights is None:
        deltas = 0.0001 * np.exp(np.arange(900) / 100.)
        heights = np.hstack([0., np.cumsum(deltas)])
    else:
        heights = np.asarray(heights)
        if heights.ndim != 1:
            raise ValueError("'heights' must be a 1D array")

    layer_mids = np.hstack([
        0.,
        0.5 * (heights[1:] + heights[:-1])
        ])

    (
        temperature,
        pressure,
        _,
        pressure_water,
        ref_index,
        _,
        _
        ) = profile_func(apu.Quantity(layer_mids, apu.km))

    adict = {}
    adict['freq_grid'] = freq_grid
    adict['heights'] = heights
    adict['radii'] = EARTH_RADIUS + heights  # distance Earth-center to layers

    # handle units
    adict['temp'] = temp = temperature.to(apu.K).value
    adict['press'] = press = pressure.to(apu.hPa).value
    adict['press_w'] = press_w = pressure_water.to(apu.hPa).value
    ref_index = ref_index.to(cnv.dimless).value
    # need to append a value (1.0) to ref_index, aka outer space
    adict['ref_index'] = np.hstack([ref_index, 1.])

    atten_dry_db = np.zeros((heights.size, freq_grid.size), dtype=np.float64)
    atten_wet_db = np.zeros((heights.size, freq_grid.size), dtype=np.float64)

    for idx in range(len(heights)):
        atten_dry_db[idx], atten_wet_db[idx] = _atten_specific_annex1(
            freq_grid, press[idx], press_w[idx], temp[idx]
            )

    adict['atten_dry_db'] = atten_dry_db
    adict['atten_wet_db'] = atten_wet_db
    adict['atten_db'] = atten_dry_db + atten_wet_db

    return adict


def _prepare_path(
        elev, obs_alt,
        radii, heights, ref_index,
        max_path_length=1000., max_arc_length=180.
        ):

    # the algorithm below will fail, if observer is *on* the smallest height
    obs_alt = max([1.e-9, obs_alt])

    start_i = np.searchsorted(heights, obs_alt)
    max_i = len(heights) - 1

    path_params, refraction, is_space_path = path_helper_cython(
        start_i,
        max_i,
        elev,  # deg
        obs_alt,  # km
        max_path_length,  # km
        max_arc_length,  # deg
        radii,
        ref_index,
        )

    return path_params, refraction, is_space_path


def _path_endpoint(
        elev, obs_alt,
        radii, heights, ref_index,
        max_path_length=1000., max_arc_length=180.
        ):

    # the algorithm below will fail, if observer is *on* the smallest height
    obs_alt = max([1.e-9, obs_alt])

    start_i = np.searchsorted(heights, obs_alt)
    max_i = len(heights) - 1

    ret = path_endpoint_cython(
        start_i,
        max_i,
        elev,  # deg
        obs_alt,  # km
        max_path_length,  # km
        max_arc_length,  # deg
        radii,
        ref_index,
        )

    # (
    #     a_n, r_n, h_n, x_n, y_n, alpha_n, beta_n, delta_n, layer_idx,
    #     refraction,
    #     is_space_path,
    #     ) = ret

    return ret


def _find_elevation(
        elev_init, obs_alt, target_alt, arc_length,
        radii, heights, ref_index, niter=200, interval=50, stepsize=2.
        ):

    from scipy.optimize import basinhopping

    def func(x):
        elev = x[0]
        ret = _path_endpoint(
            elev, obs_alt, radii, heights, ref_index,
            max_arc_length=arc_length,
            )
        h_n = ret[2]
        arc_len = ret[7]
        a_tot = ret[9]
        return h_n, arc_len, a_tot

    def opt_func(x):
        elev = x[0]
        h_n, arc_len, a_tot = func(x)
        mmin = (
            # primary optimization aim
            abs(h_n - target_alt) +
            # make sure, arc length is compatible with condition
            abs(np.degrees(arc_len) - arc_length) +
            # add a penalty term:
            # path length must be > projected earth surface length
            # (
            #     0.
            #     if a_tot > (EARTH_RADIUS * arc_len) else
            #     abs(a_tot - (EARTH_RADIUS * arc_len))
            #     )
            (0 if elev >= -90 else -90 - elev) +
            (0 if elev <= 90 else elev - 90)
            )
        # print(h_n, arc_len, a_tot, abs(h_n - target_alt), abs(np.degrees(arc_len) - arc_length), mmin)
        return mmin

    # need to avoid h_n == 0 and elevations below or above -90 or 90
    class MyBounds(object):

        def __init__(self, xmax=90., xmin=-90):
            self.xmax = xmax
            self.xmin = xmin

        def __call__(self, **kwargs):
            x = kwargs["x_new"][0]
            h_n, arc_len, a_tot = func(kwargs["x_new"])
            tmax = x <= self.xmax
            tmin = x >= self.xmin
            hmin = h_n >= 0
            # pmin = a_tot > EARTH_RADIUS * arc_len
            # print(x, tmax, tmin, hmin, pmin)
            # x = kwargs["x_new"]
            # print(x)
            # h_n, arc_len, a_tot = np.array([func(x_i) for x_i in x]).T
            # tmax = bool(np.all(x <= self.xmax))
            # tmin = bool(np.all(x >= self.xmin))
            # hmin = bool(np.all(h_n >= 0))
            # pmin = bool(np.all(a_tot > (EARTH_RADIUS * arc_len)))
            # print(x, tmax, tmin, hmin, pmin)
            return tmax and tmin and hmin  # and pmin

    x0 = np.array([elev_init])
    minimizer_kwargs = {'method': 'BFGS'}
    mybounds = MyBounds()
    res = basinhopping(
        opt_func, x0,
        T=0.05, minimizer_kwargs=minimizer_kwargs,
        accept_test=mybounds,
        niter=niter, interval=interval, stepsize=stepsize,
        )

    elev_final = res['x'][0]
    h_final = func(res['x'])[0]

    return elev_final, h_final


@utils.ranged_quantity_input(
    elevation=(-90, 90, apu.deg),
    obs_alt=(0, None, apu.km),
    t_bg=(1.e-30, None, apu.K),
    max_arc_length=(1.e-30, 180., apu.deg),
    max_path_length=(1.e-30, None, apu.km),
    strip_input_units=True, output_unit=(cnv.dB, apu.deg, apu.K)
    )
def atten_slant_annex1(
        elevation, obs_alt, atm_layers_dict, do_tebb=True,
        t_bg=2.73 * apu.K,
        max_arc_length=180. * apu.deg,
        max_path_length=1000. * apu.km,
        ):
    '''
    Path attenuation for a slant path through full atmosphere according to
    `ITU-R P.676-10 <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_
    Eq (17-20).

    Parameters
    ----------
    elevation : `~astropy.units.Quantity`, scalar
        (Apparent) elevation of source as seen from observer [deg]
    obs_alt : `~astropy.units.Quantity`, scalar
        Height of observer above sea-level [km]
    atm_layers_dict : dict
        TODO
    do_tebb : boolean
        TODO
    t_bg : `~astropy.units.Quantity`, scalar, optional
        Background temperature, i.e. temperature just after the outermost
        layer (default: 2.73 K)

        This is needed for accurate `t_ebb` calculation, usually this is the
        temperature of the CMB (if Earth-Space path), but at lower
        frequencies, Galactic foreground contribution might play a role.
    max_path_length : `~astropy.units.Quantity`, scalar
        Maximal length of path before stopping iteration [km]
        (default: 1000 km; useful for terrestrial paths)
    max_arc_length : `~astropy.units.Quantity`, scalar
        Maximal arc of path before stopping iteration [deg]
        (default: 180 deg; useful for terrestrial paths)

    Returns
    -------
    total_atten : `~astropy.units.Quantity`
        Total attenuation along path [dB]
    Refraction : `~astropy.units.Quantity`
        Offset with respect to a hypothetical straight path, i.e., the
        correction between real and apparent source elevation [deg]
    t_ebb (K) : `~astropy.units.Quantity`
        Equivalent black body temperature of the atmosphere (accounting
        for any outside contribution, e.g., from CMB) [K]

    Notes
    -----
    '''

    if not isinstance(elevation, numbers.Real):
        raise TypeError('elevation must be a scalar float')
    if not isinstance(obs_alt, numbers.Real):
        raise TypeError('obs_alt must be a scalar float')
    if not isinstance(t_bg, numbers.Real):
        raise TypeError('t_bg must be a scalar float')
    if not isinstance(max_path_length, numbers.Real):
        raise TypeError('max_path_length must be a scalar float')
    if not isinstance(max_arc_length, numbers.Real):
        raise TypeError('max_arc_length must be a scalar float')

    adict = atm_layers_dict

    tebb = np.ones(adict['freq_grid'].shape, dtype=np.float64) * t_bg

    path_params, refraction, is_space_path = _prepare_path(
        elevation, obs_alt,
        adict['radii'], adict['heights'], adict['ref_index'],
        max_path_length=max_path_length, max_arc_length=max_arc_length,
        )

    # do backward raytracing (to allow tebb calculation); this makes only
    # sense if we have a path that goes to space!

    atten_db = adict['atten_db']
    temp = adict['temp']
    total_atten_db = np.sum(
        atten_db[path_params.layer_idx] *
        path_params.a_n[:, np.newaxis],
        axis=0,
        )

    # TODO: do the following in cython?
    if is_space_path and do_tebb:
        for idx, lidx in list(enumerate(path_params.layer_idx))[::-1]:

            # need to calculate (linear) atten per layer for tebb
            atten_tot_lin = 10 ** (
                -atten_db[lidx] * path_params.a_n[idx] / 10.
                )
            tebb *= atten_tot_lin
            tebb += (1. - atten_tot_lin) * temp[lidx]

    else:
        tebb[...] = np.nan

    return total_atten_db, refraction, tebb


def _phi_helper(r_p, r_t, args):
    '''
    Helper function according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_ Eq (22u).
    '''

    phi0, a, b, c, d = args
    return phi0 * r_p ** a * r_t ** b * np.exp(
        c * (1. - r_p) + d * (1. - r_t)
        )


_HELPER_PARAMS = {
    'xi1': (1., 0.0717, -1.8132, 0.0156, -1.6515),
    'xi2': (1., 0.5146, -4.6368, -0.1921, -5.7416),
    'xi3': (1., 0.3414, -6.5851, 0.2130, -8.5854),
    'xi4': (1., -0.0112, 0.0092, -0.1033, -0.0009),
    'xi5': (1., 0.2705, -2.7192, -0.3016, -4.1033),
    'xi6': (1., 0.2445, -5.9191, 0.0422, -8.0719),
    'xi7': (1., -0.1833, 6.5589, -0.2402, 6.131),
    'gamma54': (2.192, 1.8286, -1.9487, 0.4051, -2.8509),
    'gamma58': (12.59, 1.0045, 3.5610, 0.1588, 1.2834),
    'gamma60': (15., 0.9003, 4.1335, 0.0427, 1.6088),
    'gamma62': (14.28, 0.9886, 3.4176, 0.1827, 1.3429),
    'gamma64': (6.819, 1.4320, 0.6258, 0.3177, -0.5914),
    'gamma66': (1.908, 2.0717, -4.1404, 0.4910, -4.8718),
    'delta': (-0.00306, 3.211, -14.94, 1.583, -16.37),
    }


_helper_funcs = dict(
    (k, partial(_phi_helper, args=v))
    for k, v in _HELPER_PARAMS.items()
    )


def _atten_specific_annex2(freq_grid, press, rho_w, temp):

    freq_grid = np.atleast_1d(freq_grid)

    if not isinstance(press, numbers.Real):
        raise TypeError('press must be a scalar float')
    if not isinstance(rho_w, numbers.Real):
        raise TypeError('rho_w must be a scalar float')
    if not isinstance(temp, numbers.Real):
        raise TypeError('temp must be a scalar float')

    _freq = freq_grid
    _press = press
    _rho_w = rho_w
    _temp = temp

    r_p = _press / 1013.
    r_t = 288. / (_temp - 0.15)

    atten_dry = np.empty_like(_freq)
    atten_wet = np.zeros_like(_freq)

    h = dict(
        (k, func(r_p, r_t))
        for k, func in _helper_funcs.items()
        )

    # calculate dry attenuation, depending on frequency
    mask = _freq <= 54
    f = _freq[mask]
    atten_dry[mask] = f ** 2 * r_p ** 2 * 1.e-3 * (
        7.2 * r_t ** 2.8 / (f ** 2 + 0.34 * r_p ** 2 * r_t ** 1.6) +
        0.62 * h['xi3'] / ((54. - f) ** (1.16 * h['xi1']) + 0.83 * h['xi2'])
        )

    mask = (_freq > 54) & (_freq <= 60)
    f = _freq[mask]
    atten_dry[mask] = np.exp(
        np.log(h['gamma54']) / 24. * (f - 58.) * (f - 60.) -
        np.log(h['gamma58']) / 8. * (f - 54.) * (f - 60.) +
        np.log(h['gamma60']) / 12. * (f - 54.) * (f - 58.)
        )

    mask = (_freq > 60) & (_freq <= 62)
    f = _freq[mask]
    atten_dry[mask] = (
        h['gamma60'] + (h['gamma62'] - h['gamma60']) * (f - 60.) / 2.
        )

    mask = (_freq > 62) & (_freq <= 66)
    f = _freq[mask]
    atten_dry[mask] = np.exp(
        np.log(h['gamma62']) / 8. * (f - 64.) * (f - 66.) -
        np.log(h['gamma64']) / 4. * (f - 62.) * (f - 66.) +
        np.log(h['gamma66']) / 8. * (f - 62.) * (f - 64.)
        )

    mask = (_freq > 66) & (_freq <= 120)
    f = _freq[mask]
    atten_dry[mask] = f ** 2 * r_p ** 2 * 1.e-3 * (
        3.02e-4 * r_t ** 3.5 +
        0.283 * r_t ** 3.8 / (
            (f - 118.75) ** 2 + 2.91 * r_p ** 2 * r_t ** 1.6
            ) +
        0.502 * h['xi6'] * (1. - 0.0163 * h['xi7'] * (f - 66.)) / (
            (f - 66.) ** (1.4346 * h['xi4']) + 1.15 * h['xi5']
            )
        )

    mask = (_freq > 120) & (_freq <= 350)
    f = _freq[mask]
    atten_dry[mask] = h['delta'] + f ** 2 * r_p ** 3.5 * 1.e-3 * (
        3.02e-4 / (1. + 1.9e-5 * f ** 1.5) +
        0.283 * r_t ** 0.3 / (
            (f - 118.75) ** 2 + 2.91 * r_p ** 2 * r_t ** 1.6
            )
        )

    # calculate wet attenuation, depending on frequency

    eta_1 = 0.955 * r_p * r_t ** 0.68 + 0.006 * _rho_w
    eta_2 = 0.735 * r_p * r_t ** 0.5 + 0.0353 * r_t ** 4 * _rho_w

    f = _freq

    def g(f, f_i):

        return 1. + ((f - f_i) / (f + f_i)) ** 2

    def _helper(a, eta, b, c, d, do_g):

        return (
            a * eta * np.exp(b * (1 - r_t)) /
            ((f - c) ** 2 + d * eta ** 2) *
            (g(f, int(c + 0.5)) if do_g else 1.)
            )

    for args in [
            (3.98, eta_1, 2.23, 22.235, 9.42, True),
            (11.96, eta_1, 0.7, 183.31, 11.14, False),
            (0.081, eta_1, 6.44, 321.226, 6.29, False),
            (3.66, eta_1, 1.6, 325.153, 9.22, False),
            (25.37, eta_1, 1.09, 380, 0, False),
            (17.4, eta_1, 1.46, 448, 0, False),
            (844.6, eta_1, 0.17, 557, 0, True),
            (290., eta_1, 0.41, 752, 0, True),
            (83328., eta_2, 0.99, 1780, 0, True),
            ]:

        atten_wet += _helper(*args)

    atten_wet *= f ** 2 * r_t ** 2.5 * _rho_w * 1.e-4

    return atten_dry, atten_wet


@utils.ranged_quantity_input(
    freq_grid=(1.e-30, 350., apu.GHz),
    press=(1.e-30, None, apu.hPa),
    rho_w=(1.e-30, None, apu.g / apu.m ** 3),
    temp=(1.e-30, None, apu.K),
    strip_input_units=True, output_unit=(cnv.dB / apu.km, cnv.dB / apu.km)
    )
def atten_specific_annex2(freq_grid, press, rho_w, temp):
    '''
    Specific (one layer) atmospheric attenuation based on a simplified
    algorithm according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Annex 2.1.

    Parameters
    ----------
    freq_grid : `~astropy.units.Quantity`
        Frequencies at which to calculate line-width shapes [GHz]
    press : `~astropy.units.Quantity`
        Total air pressure (dry + wet) [hPa]
    rho_w : `~astropy.units.Quantity`
        Water vapor density [g / m^3]
    temp : `~astropy.units.Quantity`
        Temperature [K]

    Returns
    -------
    atten_dry : `~astropy.units.Quantity`
        Dry-air specific attenuation [dB / km]
    atten_wet : `~astropy.units.Quantity`
        Wet-air specific attenuation [dB / km]

    Notes
    -----
    In contrast to Annex 1, the method in Annex 2 is only valid below 350 GHz.
    '''

    return _atten_specific_annex2(
        freq_grid, press, rho_w, temp
        )


@utils.ranged_quantity_input(
    freq_grid=(1.e-30, 350., apu.GHz),
    press=(1.e-30, None, apu.hPa),
    strip_input_units=True, output_unit=apu.km
    )
def equivalent_height_dry(freq_grid, press):
    '''
    Equivalent height for dry air according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Annex 2.2.

    Parameters
    ----------
    freq_grid : `~astropy.units.Quantity`
        Frequenciesat which to calculate line-width shapes [GHz]
    press : `~astropy.units.Quantity`
        Total air pressure (dry + wet) [hPa]

    Returns
    -------
    h_dry : `~astropy.units.Quantity`
        Equivalent height for dry air [km]
    '''

    r_p = press / 1013.

    f = np.atleast_1d(freq_grid).astype(dtype=np.float64, copy=False)

    t_1 = 4.64 / (1. + 0.066 * r_p ** -2.3) * np.exp(
        - ((f - 59.7) / (2.87 + 12.4 * np.exp(-7.9 * r_p))) ** 2
        )

    t_2 = 0.14 * np.exp(2.12 * r_p) / (
        (f - 118.75) ** 2 + 0.031 * np.exp(2.2 * r_p)
        )

    t_3 = 0.0114 * f / (1. + 0.14 * r_p ** -2.6) * (
        (-0.0247 + 0.0001 * f + 1.61e-6 * f ** 2) /
        (1. - 0.0169 * f + 4.1e-5 * f ** 2 + 3.2e-7 * f ** 3)
        )

    h_0 = 6.1 * (1. + t_1 + t_2 + t_3) / (1. + 0.17 * r_p ** -1.1)

    h_0[(h_0 > 10.8 * r_p ** 0.3) & (f < 70.)] = 10.8 * r_p ** 0.3

    return h_0.squeeze()


@utils.ranged_quantity_input(
    freq_grid=(1.e-30, 350., apu.GHz),
    press=(1.e-30, None, apu.hPa),
    strip_input_units=True, output_unit=apu.km
    )
def equivalent_height_wet(freq_grid, press):
    '''
    Equivalent height for wet air according to `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Annex 2.2.

    Parameters
    ----------
    freq_grid : `~astropy.units.Quantity`
        Frequenciesat which to calculate line-width shapes [GHz]
    press : `~astropy.units.Quantity`
        Total air pressure (dry + wet) [hPa]

    Returns
    -------
    h_wet : `~astropy.units.Quantity`
        Equivalent height for wet air [km]
    '''

    r_p = press / 1013.

    f = np.atleast_1d(freq_grid).astype(dtype=np.float64, copy=False)

    s_w = 1.013 / (1. + np.exp(-8.6 * (r_p - 0.57)))

    def _helper(a, b, c):
        return a * s_w / ((f - b) ** 2 + c * s_w)

    h_w = 1.66 * (
        1. +
        _helper(1.39, 22.235, 2.56) +
        _helper(3.37, 183.31, 4.69) +
        _helper(1.58, 325.1, 2.89)
        )

    return h_w.squeeze()


@utils.ranged_quantity_input(
    atten_dry=(1.e-30, None, cnv.dB / apu.km),
    atten_wet=(1.e-30, None, cnv.dB / apu.km),
    h_dry=(1.e-30, None, apu.km),
    h_wet=(1.e-30, None, apu.km),
    elev=(-90, 90, apu.deg),
    strip_input_units=True, output_unit=cnv.dB
    )
def atten_slant_annex2(atten_dry, atten_wet, h_dry, h_wet, elev):
    '''
    Simple path attenuation for slant path through full atmosphere according to
    `ITU-R P.676-10 <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_,
    Eq (28).

    Parameters
    ----------
    atten_dry : `~astropy.units.Quantity`
        Specific attenuation for dry air [dB / km]
    atten_wet : `~astropy.units.Quantity`
        Specific attenuation for wet air [dB / km]
    h_dry : `~astropy.units.Quantity`
        Equivalent height for dry air [km]
    h_wet : `~astropy.units.Quantity`
        Equivalent height for wet air [km]
    elev : `~astropy.units.Quantity`
        (Apparent) elevation of source as seen from observer [deg]

    Returns
    -------
    total_atten : `~astropy.units.Quantity`
        Total attenuation along path [dB]

    Notes
    -----
    You can use the helper functions `~pycraf.atm.equivalent_height_dry` and
    `~pycraf.atm.equivalent_height_wet` to infer the equivalent heights from
    the total (wet+dry) air pressure.
    '''

    AM = 1. / np.sin(np.radians(elev))

    return AM * (atten_dry * h_dry + atten_wet * h_wet)
