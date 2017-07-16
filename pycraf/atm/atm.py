#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

import os
from functools import partial
import numbers
import collections
import numpy as np
from astropy import units as apu
from .. import conversions as cnv
from .. import utils


__all__ = [
    'refractive_index', 'saturation_water_pressure',
    'pressure_water_from_humidity', 'humidity_from_pressure_water',
    'pressure_water_from_rho_water', 'rho_water_from_pressure_water',
    'profile_standard', 'profile_lowlat',
    'profile_midlat_summer', 'profile_midlat_winter',
    'profile_highlat_summer', 'profile_highlat_winter',
    'resonances_oxygen', 'resonances_water',
    # 'atten_linear_from_atten_log', 'atten_log_from_atten_linear',
    'opacity_from_atten', 'atten_from_opacity',
    'atten_specific_annex1',
    'atten_terrestrial', 'atten_slant_annex1',
    'atten_specific_annex2',
    'atten_slant_annex2',
    'equivalent_height_dry', 'equivalent_height_wet',
    # '_prepare_path'
    ]


this_dir, this_filename = os.path.split(__file__)
fname_oxygen = os.path.join(
    this_dir, '../itudata/p.676-10', 'R-REC-P.676-10-201309_table1.csv'
    )
fname_water = os.path.join(
    this_dir, '../itudata/p.676-10', 'R-REC-P.676-10-201309_table2.csv'
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


@utils.ranged_quantity_input(
    atten=(1.000000000001, None, cnv.dimless), elev=(-90, 90, apu.deg),
    strip_input_units=True, output_unit=cnv.dimless
    )
def opacity_from_atten(atten, elev):
    '''
    Atmospheric opacity derived from attenuation.

    Parameters
    ----------
    atten : `~astropy.units.Quantity`
        Atmospheric attenuation [dB or dimless]
    elev : `~astropy.units.Quantity`
        Elevation [deg]
        This is used to calculate the so-called Airmass, AM = 1 / sin(elev)

    Returns
    -------
    tau : `~astropy.units.Quantity`
        Atmospheric opacity [dimless aka neper]
    '''

    AM_inv = np.sin(np.radians(elev))

    return AM_inv * np.log(atten)


@utils.ranged_quantity_input(
    tau=(0.000000000001, None, cnv.dimless), elev=(-90, 90, apu.deg),
    strip_input_units=True, output_unit=cnv.dB
    )
def atten_from_opacity(tau, elev):
    '''
    Atmospheric attenuation derived from opacity.

    Parameters
    ----------
    tau : `~astropy.units.Quantity`
        Atmospheric opacity [dimless aka neper]
    elev : `~astropy.units.Quantity`
        Elevation [deg]

        This is used to calculate the so-called Airmass, AM = 1 / sin(elev)

    Returns
    -------
    atten : `~astropy.units.Quantity`
        Atmospheric attenuation [dB or dimless]
    '''

    AM = 1. / np.sin(np.radians(elev))

    return 10 * np.log10(np.exp(tau * AM))


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

    return (
        1 + 1e-6 / temp * (
            77.6 * press - 5.6 * press_w +
            3.75e5 * press_w / temp
            )
        )


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

    e_s = _saturation_water_pressure(
        temp, press, wet_type=wet_type
        )

    press_w = humidity / 100. * e_s

    return press_w


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

    e_s = _saturation_water_pressure(
        temp, press, wet_type=wet_type
        )

    humidity = 100. * press_w / e_s

    return humidity


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

    press_w = rho_w * temp / 216.7

    return press_w


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

    rho_w = press_w * 216.7 / temp

    return rho_w


@utils.ranged_quantity_input(
    height=(0, 84.99999999, apu.km),
    strip_input_units=True, output_unit=None
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

    temperatures = apu.Quantity(temperatures.reshape(height.shape), apu.K)
    pressures = apu.Quantity(pressures.reshape(height.shape), apu.hPa)
    pressures_water = apu.Quantity(
        pressures_water.reshape(height.shape), apu.hPa
        )
    rho_water = apu.Quantity(
        rho_water.reshape(height.shape), apu.g / apu.m ** 3
        )

    ref_indices = refractive_index(temperatures, pressures, pressures_water)
    humidities_water = humidity_from_pressure_water(
        temperatures, pressures, pressures_water, wet_type='water'
        )
    humidities_ice = humidity_from_pressure_water(
        temperatures, pressures, pressures_water, wet_type='ice'
        )

    result = (
        temperatures.squeeze(),
        pressures.squeeze(),
        rho_water.squeeze(),
        pressures_water.squeeze(),
        ref_indices.squeeze(),
        humidities_water.squeeze(),
        humidities_ice.squeeze(),
        )

    # return tuple(v.reshape(height.shape) for v in result)
    return result


@apu.quantity_input(height=apu.km)
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
    height : `~astropy.units.Quantity`
        Height above ground [km]
    {temp,press,rho}_heights : list of floats
        Height steps for which piece-wise functions are defined
    {temp,press,rho}_funcs - list of functions
        Functions that return the desired quantity for a given height interval

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

    height = np.atleast_1d(height)
    _height = height.to(apu.km).value

    assert np.all(_height < temp_heights[-1]), (
        'profile only defined below {} km height!'.format(temp_heights[-1])
        )
    assert np.all(_height >= temp_heights[0]), (
        'profile only defined above {} km height!'.format(temp_heights[0])
        )

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
        mask = (_height >= hmin) & (_height < hmax)
        temperature[mask] = (temp_funcs[i])(_height[mask])

    # calculate pressure profile
    for i in range(len(press_heights) - 1):
        hmin, hmax = press_heights[i], press_heights[i + 1]
        mask = (_height >= hmin) & (_height < hmax)
        pressure[mask] = (press_funcs[i])(Pstarts[i], _height[mask])

    # calculate rho profile
    for i in range(len(rho_heights) - 1):
        hmin, hmax = rho_heights[i], rho_heights[i + 1]
        mask = (_height >= hmin) & (_height < hmax)
        rho_water[mask] = (rho_funcs[i])(_height[mask])

    # calculate pressure_water profile
    pressure_water = rho_water * temperature / 216.7
    mask = (pressure_water / pressure) < 2.e-6
    pressure_water[mask] = pressure[mask] * 2.e-6
    rho_water[mask] = pressure_water[mask] / temperature[mask] * 216.7

    temperature = apu.Quantity(temperature.reshape(height.shape), apu.K)
    pressure = apu.Quantity(pressure.reshape(height.shape), apu.hPa)
    pressure_water = apu.Quantity(
        pressure_water.reshape(height.shape), apu.hPa
        )
    rho_water = apu.Quantity(
        rho_water.reshape(height.shape), apu.g / apu.m ** 3
        )

    ref_index = refractive_index(temperature, pressure, pressure_water)
    humidity_water = humidity_from_pressure_water(
        temperature, pressure, pressure_water, wet_type='water'
        )
    humidity_ice = humidity_from_pressure_water(
        temperature, pressure, pressure_water, wet_type='ice'
        )

    return (
        temperature.squeeze(),
        pressure.squeeze(),
        rho_water.squeeze(),
        pressure_water.squeeze(),
        ref_index.squeeze(),
        humidity_water.squeeze(),
        humidity_ice.squeeze(),
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


def _prepare_path(elev, obs_alt, profile_func, max_path_length=1000.):
    '''
    Helper function to construct the path parameters; see `ITU-R P.676-10
    <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, Annex 1.

    Parameters
    ----------
    elev : float
        (Apparent) elevation of source as seen from observer [deg]
    obs_alt : float
        Height of observer above sea-level [m]
    profile_func : func
        A height profile function having the same signature as
        `~pycraf.atm.profile_standard`
    max_path_length : float, optional
        Maximal length of path before stopping iteration [km]
        (default: 1000 km; useful for terrestrial paths)

    Returns
    -------
    layer_params : list
        List of tuples with the following quantities for each height layer `n`:

        0) press_n - Total pressure [hPa]
        1) press_w_n - Water vapor partial pressure [hPa]
        2) temp_n - Temperature [K]
        3) a_n - Path length [km]
        4) r_n - Radius (i.e., distance to Earth center) [km]
        5) alpha_n - exit angle [rad]
        6) delta_n - angle between current normal vector and first normal
           vector (aka projected angular distance to starting point) [rad]
        7) beta_n - entry angle [rad]
        8) h_n - height above sea-level [km]

    Refraction : float
        Offset with respect to a hypothetical straight path, i.e., the
        correction between real and apparent source elevation [deg]

    Notes
    -----
    Although the `profile_func` must have the same signature as
    `~pycraf.atm.profile_standard`, which is one of the standardized
    atmospheric height profiles, only temperature, total pressure and
    water vapor pressure are needed here, i.e., `profile_func` may return
    dummy values for the rest.
    '''

    # construct height layers
    # deltas = 0.0001 * np.exp(np.arange(922) / 100.)
    # atm profiles only up to 80 km...
    deltas = 0.0001 * np.exp(np.arange(899) / 100.)
    heights = np.cumsum(deltas)

    # radius calculation
    # TODO: do we need to account for non-spherical Earth?
    # probably not - some tests suggest that the relative error is < 1e-6
    earth_radius = 6371. + obs_alt / 1000.
    radii = earth_radius + heights  # distance Earth-center to layers

    (
        temperature,
        pressure,
        rho_water,
        pressure_water,
        ref_index,
        humidity_water,
        humidity_ice
        ) = profile_func(apu.Quantity(heights, apu.km))
    # handle units
    temperature = temperature.to(apu.K).value
    pressure = pressure.to(apu.hPa).value
    rho_water = rho_water.to(apu.g / apu.m ** 3).value
    pressure_water = pressure_water.to(apu.hPa).value
    ref_index = ref_index.to(cnv.dimless).value
    humidity_water = humidity_water.to(apu.percent).value
    humidity_ice = humidity_ice.to(apu.percent).value

    def fix_arg(arg):
        '''
        Ensure argument is in [-1., +1.] for arcsin, arccos functions.
        '''

        if arg < -1.:
            return -1.
        elif arg > 1.:
            return 1.
        else:
            return arg

    # calculate layer path lengths (Equation 17 to 19)
    # all angles in rad
    beta_n = beta_0 = np.radians(90. - elev)  # initial value

    # we will store a_n, gamma_n, and temperature for each layer, to allow
    # Tebb calculation
    path_params = []
    # angle of the normal vector (r_n) at current layer w.r.t. zenith (r_1):
    delta_n = 0
    path_length = 0

    # TODO: this is certainly a case for cython
    for i in range(len(heights) - 1):

        r_n = radii[i]
        d_n = deltas[i]
        a_n = -r_n * np.cos(beta_n) + 0.5 * np.sqrt(
            4 * r_n ** 2 * np.cos(beta_n) ** 2 + 8 * r_n * d_n + 4 * d_n ** 2
            )
        alpha_n = np.pi - np.arccos(fix_arg(
            (-a_n ** 2 - 2 * r_n * d_n - d_n ** 2) / 2. / a_n / (r_n + d_n)
            ))
        delta_n += beta_n - alpha_n
        beta_n = np.arcsin(
            fix_arg(ref_index[i] / ref_index[i + 1] * np.sin(alpha_n))
            )

        h_n = 0.5 * (heights[i] + heights[i + 1])
        press_n = 0.5 * (pressure[i] + pressure[i + 1])
        press_w_n = 0.5 * (pressure_water[i] + pressure_water[i + 1])
        temp_n = 0.5 * (temperature[i] + temperature[i + 1])

        path_length += a_n
        if path_length > max_path_length:
            break

        path_params.append((
            press_n, press_w_n, temp_n,
            a_n, r_n, alpha_n, delta_n, beta_n, h_n
            ))

    refraction = - np.degrees(beta_n + delta_n - beta_0)

    return path_params, refraction


@utils.ranged_quantity_input(
    freq_grid=(1.e-30, 1000, apu.GHz),
    elevation=(-90, 90, apu.deg),
    obs_alt=(1.e-30, None, apu.m),
    t_bg=(1.e-30, None, apu.K),
    max_path_length=(1.e-30, None, apu.km),
    strip_input_units=True, output_unit=(cnv.dB, apu.deg, apu.K)
    )
def atten_slant_annex1(
        freq_grid, elevation, obs_alt, profile_func,
        t_bg=2.73 * apu.K, max_path_length=1000. * apu.km
        ):
    '''
    Path attenuation for a slant path through full atmosphere according to
    `ITU-R P.676-10 <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_
    Eq (17-20).

    Parameters
    ----------
    freq_grid : `~astropy.units.Quantity`
        Frequencies at which to calculate line-width shapes [GHz]
    elev : `~astropy.units.Quantity`, scalar
        (Apparent) elevation of source as seen from observer [deg]
    obs_alt : `~astropy.units.Quantity`, scalar
        Height of observer above sea-level [m]
    profile_func : func
        A height profile function having the same signature as
        `~pycraf.atm.profile_standard`
    t_bg : `~astropy.units.Quantity`, scalar, optional
        Background temperature, i.e. temperature just after the outermost
        layer (default: 2.73 K)

        This is needed for accurate `t_ebb` calculation, usually this is the
        temperature of the CMB (if Earth-Space path), but at lower
        frequencies, Galactic foreground contribution might play a role.
    max_path_length : `~astropy.units.Quantity`, scalar
        Maximal length of path before stopping iteration [km]
        (default: 1000 km; useful for terrestrial paths)

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
    Although the `profile_func` must have the same signature as
    `~pycraf.atm.profile_standard`, which is one of the standardized
    atmospheric height profiles, only temperature, total pressure and
    water vapor pressure are needed here, i.e., `profile_func` may return
    dummy values for the rest.
    '''

    if not isinstance(elevation, numbers.Real):
        raise TypeError('elevation must be a scalar float')
    if not isinstance(obs_alt, numbers.Real):
        raise TypeError('obs_alt must be a scalar float')
    if not isinstance(t_bg, numbers.Real):
        raise TypeError('t_bg must be a scalar float')
    if not isinstance(max_path_length, numbers.Real):
        raise TypeError('max_path_length must be a scalar float')

    freq_grid = np.atleast_1d(freq_grid)
    _freq = freq_grid
    _elev = elevation
    _alt = obs_alt
    _t_bg = t_bg
    _max_plen = max_path_length

    total_atten_db = np.zeros(freq_grid.shape, dtype=np.float64)
    tebb = np.ones(freq_grid.shape, dtype=np.float64) * _t_bg

    path_params, refraction = _prepare_path(
        _elev, _alt, profile_func, max_path_length=_max_plen
        )

    # do backward raytracing (to allow tebb calculation)

    for press_n, press_w_n, temp_n, a_n, _, _, _, _, _ in path_params[::-1]:

        atten_dry, atten_wet = _atten_specific_annex1(
            _freq, press_n, press_w_n, temp_n
        )
        gamma_n = atten_dry + atten_wet
        total_atten_db += gamma_n * a_n

        # need to calculate (linear) atten per layer for tebb
        gamma_n_lin = 10 ** (-gamma_n * a_n / 10.)
        tebb *= gamma_n_lin
        tebb += (1. - gamma_n_lin) * temp_n

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
