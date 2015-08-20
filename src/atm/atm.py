#!/usr/bin/python
# -*- coding: utf-8 -*-
# Licensed under GPL v2 - see LICENSE

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

import os
from functools import partial
import numpy as np


__all__ = [
    'refractive_index', 'saturation_water_pressure',
    'pressure_water_from_humidity', 'humidity_from_pressure_water',
    'pressure_water_from_rho_water', 'rho_water_from_pressure_water',
    'standard_profile', 'low_latitude_profile',
    'mid_latitude_summer_profile', 'mid_latitude_winter_profile',
    'high_latitude_summer_profile', 'high_latitude_winter_profile',
    'resonances_oxygen', 'resonances_water',
    'atten_linear_from_atten_log', 'atten_log_from_atten_linear',
    'opacity_from_atten_linear', 'atten_linear_from_opacity',
    'specific_attenuation_annex1',
    'terrestrial_attenuation', 'slant_attenuation_annex1',
    'specific_attenuation_annex2',
    'slant_attenuation_annex2',
    '_equivalent_height_dry', '_equivalent_height_wet',
    '_prepare_path'
    ]


this_dir, this_filename = os.path.split(__file__)
fname_oxygen = os.path.join(this_dir, 'data', 'R-REC-P.676-10-201309_table1.csv')
fname_water = os.path.join(this_dir, 'data', 'R-REC-P.676-10-201309_table2.csv')

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


def atten_linear_from_atten_log(atten_db):

    return 10 ** (atten_db / 10.)


def atten_log_from_atten_linear(atten):

    return 10 * np.log10(atten)


def opacity_from_atten_linear(atten, elevation):

    AM_inv = np.sin(np.radians(elevation))
    return -AM_inv * np.log(atten)


def atten_linear_from_opacity(opacity, elevation):

    AM = 1. / np.sin(np.radians(elevation))
    return np.exp(-opacity * AM)


def refractive_index(temperature, pressure, pressure_water):
    '''
    Calculate refractive index according to ITU-R P.453-10.

    Parameters
    ----------
    temperature - air temperature (K)
    pressure - total air pressure (hPa)
    pressure_water - water vapor partial pressure (hPa)

    Returns
    -------
    Refractive index (dimensionless)
    '''

    return 1 + 77.6e-6 / temperature * (
        pressure + 4810. * pressure_water / temperature
        )


def saturation_water_pressure(temperature, pressure, wet_type='water'):
    '''
    Calculate saturation water pressure according to ITU-R P.453-10.

    Parameters
    ----------
    temperature - air temperature (K)
    pressure - total air pressure (hPa)
    wet_type - 'water' or 'ice'

    Returns
    -------
    Saturation water vapor pressure, e_s (hPa)
    '''

    assert wet_type in ['water', 'ice']

    t = temperature - 273.15  # t is temperature in Celcius

    EF = (
        1. + 1.e-4 * (7.2 + pressure * (0.00320 + 5.9e-7 * t ** 2))
        if wet_type == 'water' else
        1. + 1.e-4 * (2.2 + pressure * (0.00382 + 6.4e-7 * t ** 2))
        )

    a, b, c, d = (
        (6.1121, 18.678, 257.14, 234.5)
        if wet_type == 'water' else
        (6.1115, 23.036, 279.82, 333.7)
        )
    e_s = EF * a * np.exp((b - t / d) * t / (c + t))

    return e_s


def pressure_water_from_humidity(
        temperature, pressure, humidity, wet_type='water'
        ):
    '''
    Calculate p_water according to ITU-R P.453-10.

    Parameters
    ----------
    temperature - air temperature (K)
    pressure - total air pressure (hPa)
    humidity - relative humidity (%)
    wet_type - 'water' or 'ice'

    Returns
    -------
    Water vapor partial pressure (hPa)
    '''

    e_s = saturation_water_pressure(
        temperature, pressure, wet_type=wet_type
        )

    pressure_water = humidity / 100. * e_s

    return pressure_water


def humidity_from_pressure_water(
        temperature, pressure, pressure_water, wet_type='water'
        ):
    '''
    Calculate relative humidity according to ITU-R P.453-10.

    wet_type - 'water' or 'ice'

    Parameters
    ----------
    temperature - air temperature (K)
    pressure - total air pressure (hPa)
    pressure_water - water vapor partial pressure (hPa)
    wet_type - 'water' or 'ice'

    Returns
    -------
    Relative humidity %)
    '''

    e_s = saturation_water_pressure(
        temperature, pressure, wet_type=wet_type
        )

    humidity  = 100. * pressure_water / e_s

    return humidity


def pressure_water_from_rho_water(temperature, rho_water):
    '''
    Calculate water pressure according to ITU-R P.453-10.

    Parameters
    ----------
    temperature - air temperature (K)
    rho_water - water vapor density (g / m**3)

    Returns
    -------
    Water vapor partial pressure (hPa)
    '''

    pressure_water = rho_water * temperature / 216.7
    return pressure_water


def rho_water_from_pressure_water(temperature, pressure_water):
    '''
    Calculate water density according to ITU-R P.453-10.

    Parameters
    ----------
    temperature - air temperature (K)
    pressure_water - water vapor partial pressure (hPa)

    Returns
    -------
    Water vapor density (g / m**3)
    '''

    rho_water = pressure_water * 216.7 / temperature
    return rho_water


def standard_profile(height):
    '''
    Compute temperature and pressure according to ITU R-P.835-5, Annex 1.

    Note, for convenience, derived quantities like water density/pressure
    and refraction indices are also returned.

    Parameters
    ----------
    height - height above ground (km, array or scalar)

    Returns
    -------
    Temperature (K)
    Total pressure (hPa)
    Water vapor density (g / m**3)
    Water vapor partial pressure (hPa)
    Refractive index (dimensionless)
    Relative humidity (%) if water vapour was in form of liquid water
    Relative humidity (%) if water vapour was in form of ice
    '''

    height = np.asarray(height)  # this is not sufficient for masking :-(
    _height = np.atleast_1d(height).astype(np.float64, copy=False)

    assert np.all(_height < 85), 'profile only defined below 85 km height!'
    assert np.all(_height >= 0), 'profile only defined above 0 km height!'

    # to make this work with numpy arrays
    # lets first find the correct index for every height


    layer_heights = np.array([0., 11., 20., 32., 47., 51., 71., 85.])
    indices = np.zeros(_height.shape, dtype=np.int32)
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

        dh = layer_heights[i] - layer_heights[i-1]
        Ti = layer_start_temperatures[i-1]
        Li = layer_temp_gradients[i-1]
        Pi = layer_start_pressures[i-1]
        #print i, Ti, Li, Pi, i - 1 not in [1, 4]
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

    ref_indices = refractive_index(temperatures, pressures, pressures_water)
    humidities_water = humidity_from_pressure_water(
        temperatures, pressures, pressures_water, wet_type='water'
        )
    humidities_ice = humidity_from_pressure_water(
        temperatures, pressures, pressures_water, wet_type='ice'
        )

    result = (
        temperatures,
        pressures,
        rho_water,
        pressures_water,
        ref_indices,
        humidities_water,
        humidities_ice,
        )
    return tuple(v.reshape(height.shape) for v in result)


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
    height - height above ground (km, array or scalar)
    [temp,press,rho]_heights - list of height steps for which piece-wise functions are defined
    [temp,press,rho]_funcs - list of functions valid for the according height interval

    Returns
    -------
    Temperature (K)
    Total pressure (hPa)
    Water vapor density (g / m**3)
    Water vapor partial pressure (hPa)
    Refractive index (dimensionless)
    Relative humidity (%) if water vapour was in form of liquid water
    Relative humidity (%) if water vapour was in form of ice
    '''

    height = np.atleast_1d(height)

    assert np.all(height < temp_heights[-1]), (
        'profile only defined below {} km height!'.format(temp_heights[-1])
        )
    assert np.all(height >= temp_heights[0]), (
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

    ref_index = refractive_index(temperature, pressure, pressure_water)
    humidity_water = humidity_from_pressure_water(
        temperature, pressure, pressure_water, wet_type='water'
        )
    humidity_ice = humidity_from_pressure_water(
        temperature, pressure, pressure_water, wet_type='ice'
        )

    return (
        temperature,
        pressure,
        rho_water,
        pressure_water,
        ref_index,
        humidity_water,
        humidity_ice
        )


def low_latitude_profile(height):
    '''
    Compute "low latitude" height profile functions according to ITU R-P.835-5.

    Valid for geographic latitudes |phi| < 22 deg.

    Parameters
    ----------
    height - height above ground (km, array or scalar)

    Returns
    -------
    Temperature (K)
    Total pressure (hPa)
    Water vapor density (g / m**3)
    Water vapor partial pressure (hPa)
    Refractive index (dimensionless)
    Relative humidity (%) if water vapour was in form of liquid water
    Relative humidity (%) if water vapour was in form of ice
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
            -0.2313 * h - 0.1122 * h ** 2 + 0.01351 * h ** 3 - 0.0005923 * h ** 4
            ),
        lambda h: 0.,
        ]

    return _profile_helper(
        height,
        temp_heights, temp_funcs,
        press_heights, press_funcs,
        rho_heights, rho_funcs,
        )


def mid_latitude_summer_profile(height):
    '''
    Compute "mid latitude summer" height profile functions according to ITU R-P.835-5.

    Valid for geographic latitudes 22 deg < |phi| < 45 deg.

    Parameters
    ----------
    height - height above ground (km, array or scalar)

    Returns
    -------
    Temperature (K)
    Total pressure (hPa)
    Water vapor density (g / m**3)
    Water vapor partial pressure (hPa)
    Refractive index (dimensionless)
    Relative humidity (%) if water vapour was in form of liquid water
    Relative humidity (%) if water vapour was in form of ice
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


def mid_latitude_winter_profile(height):
    '''
    Compute "mid latitude winter" height profile functions according to ITU R-P.835-5.

    Valid for geographic latitudes 22 deg < |phi| < 45 deg.

    Parameters
    ----------
    height - height above ground (km, array or scalar)

    Returns
    -------
    Temperature (K)
    Total pressure (hPa)
    Water vapor density (g / m**3)
    Water vapor partial pressure (hPa)
    Refractive index (dimensionless)
    Relative humidity (%) if water vapour was in form of liquid water
    Relative humidity (%) if water vapour was in form of ice
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


def high_latitude_summer_profile(height):
    '''
    Compute "high latitude summer" height profile functions according to ITU R-P.835-5.

    Valid for geographic latitudes |phi| > 45 deg.

    Parameters
    ----------
    height - height above ground (km, array or scalar)

    Returns
    -------
    Temperature (K)
    Total pressure (hPa)
    Water vapor density (g / m**3)
    Water vapor partial pressure (hPa)
    Refractive index (dimensionless)
    Relative humidity (%) if water vapour was in form of liquid water
    Relative humidity (%) if water vapour was in form of ice
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


def high_latitude_winter_profile(height):
    '''
    Compute "high latitude winter" height profile functions according to ITU R-P.835-5.

    Valid for geographic latitudes |phi| > 45 deg.

    Parameters
    ----------
    height - height above ground (km, array or scalar)

    Returns
    -------
    Temperature (K)
    Total pressure (hPa)
    Water vapor density (g / m**3)
    Water vapor partial pressure (hPa)
    Refractive index (dimensionless)
    Relative humidity (%) if water vapour was in form of liquid water
    Relative humidity (%) if water vapour was in form of ice
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


def _S_oxygen(pressure_dry, temperature):
    '''
    Compute line strength of all oxygen resonances (Equation [P.676-10: 3]).

    Parameters
    ----------
    pressure_dry - dry air pressure (~oxygen) (hPa)
    temperature - temperature (K)

    Returns
    -------
    Line strength

    Note:
    pressure_total = pressure_dry + pressure_water
    '''

    theta = 300. / temperature
    factor = 1.e-7 * pressure_dry * theta ** 3

    return resonances_oxygen['a1'] * factor * np.exp(resonances_oxygen['a2'] * (1. - theta))


def _S_water(pressure_water, temperature):
    '''
    Compute line strength of all water resonances (Equation [P.676-10: 3]).

    Parameters
    ----------
    pressure_water - water vapor partial pressure (hPa)
    temperature - temperature (K)

    Returns
    -------
    Line strength

    Note:
    pressure_total = pressure_dry + pressure_water
    '''

    theta = 300. / temperature
    factor = 1.e-1 * pressure_water * theta ** 3.5

    return resonances_water['b1'] * factor * np.exp(resonances_water['b2'] * (1. - theta))


def _Delta_f_oxygen(pressure_dry, pressure_water, temperature):
    '''
    Calculate line width for all oxygen resonances (Equation [P.676-10: 6a/b]).

    Parameters
    ----------
    pressure_dry - dry air pressure (~oxygen) (hPa)
    pressure_water - water vapor partial pressure (hPa)
    temperature - temperature (K)

    Returns
    -------
    Line widths for all oxygen resonances.

    Note: oxygen resonance line widths also depend on wet-air pressure.
    '''

    theta = 300. / temperature
    df = resonances_oxygen['a3'] * 1.e-4 * (
        pressure_dry * theta ** (0.8 - resonances_oxygen['a4']) +
        1.1 * pressure_water * theta
        )
    return np.sqrt(df ** 2 + 2.25e-6)


def _Delta_f_water(pressure_dry, pressure_water, temperature):
    '''
    Calculate line width for all water vapor resonances (Equation [P.676-10: 6a/b]).

    Parameters
    ----------
    pressure_dry - dry air pressure (~oxygen) (hPa)
    pressure_water - water vapor partial pressure (hPa)
    temperature - temperature (K)

    Returns
    -------
    Line widths for all oxygen resonances.

    Note: water resonance line widths also depend on dry-air pressure.
    '''

    theta = 300. / temperature
    f0, b3, b4, b5, b6 = (resonances_water[b] for b in ['f0', 'b3', 'b4', 'b5', 'b6'])

    df = b3 * 1.e-4 * (
        pressure_dry * theta ** b4 +
        b5 * pressure_water * theta ** b6
        )
    return 0.535 * df + np.sqrt(
        0.217 * df ** 2 + 2.1316e-12 * f0 ** 2 / theta
        )


def _delta_oxygen(pressure_dry, pressure_water, temperature):
    '''
    Calculate shape correction factor for all oxygen resonances (Equation [P.676-10: 7]).

    Parameters
    ----------
    pressure_dry - dry air pressure (~oxygen) (hPa)
    pressure_water - water vapor partial pressure (hPa)
    temperature - temperature (K)

    Returns
    -------
    Profile shape correction factors for all oxygen resonances.

    Note, this function accounts for interference effects in oxygen lines.
    '''

    theta = 300. / temperature
    return (
        (resonances_oxygen['a5'] + resonances_oxygen['a6'] * theta) * 1.e-4 *
        (pressure_dry + pressure_water) * theta ** 0.8
        )


def _delta_water():
    '''
    Return shape correction factor for all water vapor resonances (all-zero).
    '''

    return np.zeros(len(resonances_water['f0']), dtype=np.float64)


def _F(freq_grid, f_i, Delta_f, delta):
    '''
    Calculate line-profile shapes for all resonances at the freq_grid positions.
    (Equation [P.676-10: 5])


    Parameters
    ----------
    freq_grid - Frequencies (GHz) at which to calculate line-width shapes
    f_i - Resonance line frequencies (GHz)
    Delta_f - line widths of all resonances
    delta - correction factors to account for interference effects in oxygen lines

    Returns
    -------
    m x n Array with the line-shape values (n = len(freq_grid), m = len(Delta_f))

    Note: no integration is done between freq_grid positions, so if you're interested
    in high accuracy near resonance lines, make your freq_grid sufficiently fine.
    '''

    _freq_grid = freq_grid[np.newaxis]
    _f_i = f_i[:, np.newaxis]
    _Delta_f = Delta_f[:, np.newaxis]
    _delta = delta[:, np.newaxis]

    _df_plus, _df_minus = _f_i + _freq_grid, _f_i - _freq_grid

    sum_1 = (_Delta_f - _delta * _df_minus) / (_df_minus ** 2 + _Delta_f ** 2)
    sum_2 = (_Delta_f - _delta * _df_plus) / (_df_plus ** 2 + _Delta_f ** 2)

    return _freq_grid / _f_i * (sum_1 + sum_2)


def _N_D_prime2(freq_grid, pressure_dry, pressure_water, temperature):
    '''
    Compute dry air continuum absorption, aka Debye spectrum (Equation [P.676-10: 8/9])

    Parameters
    ----------
    freq_grid - Frequencies (GHz) at which to calculate line-width shapes
    pressure_dry - dry air pressure (~oxygen) (hPa)
    pressure_water - water vapor partial pressure (hPa)
    temperature - temperature (K)

    Returns
    -------
    Debye absorption spectrum
    '''

    theta = 300. / temperature
    d = 5.6e-4 * (pressure_dry + pressure_water) * theta ** 0.8

    sum_1 = 6.14e-5 / d / (1 + (freq_grid / d) ** 2)
    sum_2 = 1.4e-12 * pressure_dry * theta ** 1.5 / (1 + 1.9e-5 * freq_grid ** 1.5)

    return freq_grid * pressure_dry * theta ** 2 * (sum_1 + sum_2)


def specific_attenuation_annex1(freq_grid, pressure_dry, pressure_water, temperature):
    '''
    Compute specific (one layer) atmospheric attenuation according to ITU-R P.676-10, annex 1.

    Parameters
    ----------
    freq_grid - Frequencies (GHz) at which to calculate specific attenuation
    pressure_dry - dry air pressure (~oxygen) (hPa)
    pressure_water - water vapor partial pressure (hPa)
    temperature - temperature (K)

    Returns
    -------
    dry_attenuation, wet_attenuation (dB / km)
    '''

    # first calculate dry attenuation (oxygen lines + N_D_prime2)
    S_o2 = _S_oxygen(pressure_dry, temperature)
    f_i = resonances_oxygen['f0']
    Delta_f = _Delta_f_oxygen(pressure_dry, pressure_water, temperature)
    delta = _delta_oxygen(pressure_dry, pressure_water, temperature)
    F_o2 = _F(freq_grid, f_i, Delta_f, delta)

    atten_o2 = np.sum(S_o2[:, np.newaxis] * F_o2, axis=0)
    atten_o2 += _N_D_prime2(freq_grid, pressure_dry, pressure_water, temperature)

    # now, wet contribution
    S_h2o = _S_water(pressure_water, temperature)
    f_i = resonances_water['f0']
    Delta_f = _Delta_f_water(pressure_dry, pressure_water, temperature)
    delta = _delta_water()
    F_h2o = _F(freq_grid, f_i, Delta_f, delta)

    atten_h2o = np.sum(S_h2o[:, np.newaxis] * F_h2o, axis=0)

    return atten_o2 * 0.182 * freq_grid, atten_h2o * 0.182 * freq_grid


def terrestrial_attenuation(specific_atten, path_length):
    '''
    Calculate total path attenuation for a path close to the ground (i.e., one layer).
    According to ITU-R P.676-10, annex 1.

    Parameters
    ----------
    specific_atten - Specific attenuation (dry + wet, dB / km)
    path_length - Length of path (km)

    Returns
    -------
    Total attenuation along path (dB)
    '''

    return specific_atten * path_length


def _prepare_path(elevation, obs_alt, profile_func, max_path_length=1000.):
    '''
    Helper function to construct the path parameters. See ITU-R P.676-10, annex 1.

    Parameters
    ----------
    elevation - (Apparent) elevation of source as seen from observer (degrees)
    obs_alt - Height of observer above sea-level (m)
    profile_func - function having height (above sea-level in km) as parameter, and
        that returns
            - Temperature (K)
            - Total pressure (hPa)
            - Water vapor density (g / m**3)
            - Water vapor partial pressure (hPa)
            - Refractive index (dimensionless)
            - Relative humidity (%) if water vapour was in form of liquid water
            - Relative humidity (%) if water vapour was in form of ice
        for that height. Note, this function must have the same signature as the
        standardized atmospheric height profiles, but since only temperature,
        total pressure and water vapor pressure are needed here, you can return
        dummy values for the rest.
    max_path_length - Maximal length of path (km) before stopping iteration (default: 1000 km)
        (useful for terrestrial paths)

    Returns
    -------
    List of tuples with the following quantities for each height layer:
        press_n - Total pressure (hPa)
        press_w_n - Water vapor partial pressure (hPa)
        temp_n - Temperature (K)
        a_n - Path length (km)
        r_n - Radius (i.e., distance to Earth center, km)
        alpha_n - exit angle
        delta_n - angle between current normal vector and first normal vector
            (aka projected angular distance to starting point)
        beta_n - entry angle
        h_n - height above sea-level (km)
    Refraction - Offset w.r.t. to a hypothetical straight path, i.e., the correction
        between real and apparent source elevation (degrees)
    '''

    # construct height layers
    # deltas = 0.0001 * np.exp(np.arange(922) / 100.)  # atm profiles only up to 80 km...
    deltas = 0.0001 * np.exp(np.arange(899) / 100.)
    heights = np.cumsum(deltas)

    # radius calculation
    # TODO: do we need to account for non-spherical Earth?
    #       probably not - some tests suggest that the relative error is < 1e-6
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
        ) = profile_func(heights)

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
    beta_n = beta_0 = np.radians(90. - elevation)  # initial value

    # we will store a_n, gamma_n, and temperature for each layer, to allow Tebb calculation
    path_params = []
    delta_n = 0  # angle of the normal vector (r_n) at current layer w.r.t. zenith (r_1)
    path_length = 0

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
        beta_n = np.arcsin(fix_arg(ref_index[i] / ref_index[i+1] * np.sin(alpha_n)))

        h_n = 0.5 * (heights[i] + heights[i+1])
        press_n = 0.5 * (pressure[i] + pressure[i+1])
        press_w_n = 0.5 * (pressure_water[i] + pressure_water[i+1])
        temp_n = 0.5 * (temperature[i] + temperature[i+1])

        path_length += a_n
        if path_length > max_path_length:
            break

        path_params.append(
            (press_n, press_w_n, temp_n, a_n, r_n, alpha_n, delta_n, beta_n, h_n)
            )

    refraction = - np.degrees(beta_n + delta_n - beta_0)

    return path_params, refraction


def slant_attenuation_annex1(
        freq_grid, elevation, obs_alt, profile_func, t_bg=2.73, max_path_length=1000.
        ):
    '''
    Calculate path attenuation for a slant path through full atmosphere.
    (Equation [P.676-10: 17-20])

    Parameters
    ----------
    freq_grid - Frequencies (GHz) at which to calculate line-width shapes
    elevation - (Apparent) elevation of source as seen from observer (degrees)
    obs_alt - Height of observer above sea-level (m)
    profile_func - function having height (above sea-level in km) as parameter, and
        that returns
            - Temperature (K)
            - Total pressure (hPa)
            - Water vapor density (g / m**3)
            - Water vapor partial pressure (hPa)
            - Refractive index (dimensionless)
            - Relative humidity (%) if water vapour was in form of liquid water
            - Relative humidity (%) if water vapour was in form of ice
        for that height. Note, this function must have the same signature as the
        standardized atmospheric height profiles, but since only temperature,
        total pressure and water vapor pressure are needed here, you can return
        dummy values for the rest.
    t_bg - background temperature (or temperature just after the last layer, default: 2.73 K)
        This is needed for accurate Tebb calculation, usually this is the temperature
        of the CMB (if Earth-Space path), but at lower frequencies, Galactic
        foreground contribution might play a role.
    max_path_length - Maximal length of path (km) before stopping iteration (default: 1000 km)
        (useful for terrestrial paths)

    Returns
    -------
    Total attenuation along path (dB)
    Refraction - Offset w.r.t. to a hypothetical straight path, i.e., the correction
        between real and apparent source elevation (degrees)
    Tebb (K) - Equivalent black body temperature of the atmosphere (accounting
        for any outside contribution, e.g., from CMB)
    '''

    total_atten_db = np.zeros(freq_grid.shape, dtype=np.float64)
    tebb = np.ones(freq_grid.shape, dtype=np.float64) * t_bg

    path_params, refraction = _prepare_path(
        elevation, obs_alt, profile_func, max_path_length=max_path_length
        )

    # do backward raytracing (to allow tebb calculation)

    for press_n, press_w_n, temp_n, a_n, _, _, _, _, _ in path_params[::-1]:

        atten_dry, atten_wet = specific_attenuation_annex1(
            freq_grid, press_n, press_w_n, temp_n
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
    Helper function (Equation [P.676-10: 22u])
    '''

    phi0, a, b, c, d = args
    return phi0 * r_p ** a * r_t ** b * np.exp(
        c * (1. - r_p) + d * (1. - r_t)
        )

_helper_params = {
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

_helper_funcs = dict((k, partial(_phi_helper, args=v)) for k, v in _helper_params.iteritems())


def specific_attenuation_annex2(freq_grid, temperature, pressure, rho_water):
    '''
    Calculate specific attenuation using a simplified algorithm (ITU-R P.676-10 Annex 2.1).

    Parameters
    ----------
    freq_grid - Frequencies (GHz)
    temperature - temperature (K)
    pressure - total air pressure (dry + wet) (hPa)
    rho_water - water vapor density (g / m^3)

    Returns
    -------
    dry_attenuation, wet_attenuation (dB / km)
    '''

    r_p = pressure / 1013.
    r_t = 288. / (temperature - 0.15)

    assert np.all(freq_grid > 0.)
    assert np.all(freq_grid <= 350.)

    freq_grid = np.atleast_1d(freq_grid).astype(np.float64, copy=False)
    atten_dry = np.empty_like(freq_grid)
    atten_wet = np.zeros_like(freq_grid)

    h = dict((k, func(r_p, r_t)) for k, func in _helper_funcs.iteritems())

    # calculate dry attenuation, depending on frequency
    mask = freq_grid <= 54
    f = freq_grid[mask]
    atten_dry[mask] = f ** 2 * r_p ** 2 * 1.e-3 * (
        7.2 * r_t ** 2.8 / (f ** 2 + 0.34 * r_p ** 2 * r_t ** 1.6) +
        0.62 * h['xi3'] / ((54. - f) ** (1.16 * h['xi1']) + 0.83 * h['xi2'])
        )

    mask = (freq_grid > 54) & (freq_grid <= 60)
    f = freq_grid[mask]
    atten_dry[mask] = np.exp(
        np.log(h['gamma54']) / 24. * (f - 58.) * (f - 60.) -
        np.log(h['gamma58']) / 8. * (f - 54.) * (f - 60.) +
        np.log(h['gamma60']) / 12. * (f - 54.) * (f - 58.)
        )

    mask = (freq_grid > 60) & (freq_grid <= 62)
    f = freq_grid[mask]
    atten_dry[mask] = h['gamma60'] + (h['gamma62'] - h['gamma60']) * (f - 60.) / 2.

    mask = (freq_grid > 62) & (freq_grid <= 66)
    f = freq_grid[mask]
    atten_dry[mask] = np.exp(
        np.log(h['gamma62']) / 8. * (f - 64.) * (f - 66.) -
        np.log(h['gamma64']) / 4. * (f - 62.) * (f - 66.) +
        np.log(h['gamma66']) / 8. * (f - 62.) * (f - 64.)
        )

    mask = (freq_grid > 66) & (freq_grid <= 120)
    f = freq_grid[mask]
    atten_dry[mask] = f ** 2 * r_p ** 2 * 1.e-3 * (
        3.02e-4 * r_t ** 3.5 +
        0.283 * r_t ** 3.8 / ((f - 118.75) ** 2 + 2.91 * r_p ** 2 * r_t ** 1.6) +
        0.502 * h['xi6'] * (1. - 0.0163 * h['xi7'] * (f - 66.)) /
        ((f - 66.) ** (1.4346 * h['xi4']) + 1.15 * h['xi5'])
        )

    mask = (freq_grid > 120) & (freq_grid <= 350)
    f = freq_grid[mask]
    atten_dry[mask] = h['delta'] + f ** 2 * r_p ** 3.5 * 1.e-3 * (
        3.02e-4 / (1. + 1.9e-5 * f ** 1.5) +
        0.283 * r_t ** 0.3 / ((f - 118.75) ** 2 + 2.91 * r_p ** 2 * r_t ** 1.6)
        )

    # calculate wet attenuation, depending on frequency

    eta_1 = 0.955 * r_p * r_t ** 0.68 + 0.006 * rho_water
    eta_2 = 0.735 * r_p * r_t ** 0.5 + 0.0353 * r_t ** 4 * rho_water

    f = freq_grid

    def g(f, f_i):

        return 1. + ((f - f_i) / (f + f_i)) ** 2

    def _helper(a, b, c, d, do_g):

        return (
            a * eta_1 * np.exp(b * (1 - r_t)) /
            ((f - c) ** 2 + d * eta_1 ** 2) *
            (g(f, int(c + 0.5)) if do_g else 1.)
            )

    for a, b, c, d, do_g in [
        (3.98, 2.23, 22.235, 9.42, True),
        (11.96, 0.7, 183.31, 11.14, False),
        (0.081, 6.44, 321.226, 6.29, False),
        (3.66, 1.6, 325.153, 9.22, False),
        (25.37, 1.09, 380, 0, False),
        (17.4, 1.46, 448, 0, False),
        (844.6, 0.17, 557, 0, True),
        (290., 0.41, 752, 0, True),
        (83328., 0.99, 1780, 0, True),
        ]:

        atten_wet += _helper(a, b, c, d, do_g)

    atten_wet *= f ** 2 * r_t ** 2.5 * rho_water * 1.e-4

    return atten_dry, atten_wet


def _equivalent_height_dry(freq_grid, pressure):
    '''
    Calculate equivalent height for dry air (ITU-R P.676-10 Annex 2.2).

    Parameters
    ----------
    freq_grid - Frequencies (GHz)
    pressure - total air pressure (dry + wet) (hPa)

    Returns
    -------
    Equivalent height for dry air (km)
    '''

    r_p = pressure / 1013.

    assert np.all(freq_grid > 0.)
    assert np.all(freq_grid <= 350.)

    f = np.atleast_1d(freq_grid).astype(dtype=np.float64, copy=False)

    t_1 = 4.64 / (1. + 0.066 * r_p ** -2.3) *np.exp(
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

    return h_0


def _equivalent_height_wet(freq_grid, pressure):
    '''
    Calculate equivalent height for wet air (ITU-R P.676-10 Annex 2.2).

    Parameters
    ----------
    freq_grid - Frequencies (GHz)
    pressure - total air pressure (dry + wet) (hPa)

    Returns
    -------
    Equivalent height for wet air (km)
    '''

    r_p = pressure / 1013.

    assert np.all(freq_grid > 0.)
    assert np.all(freq_grid <= 350.)

    f = np.atleast_1d(freq_grid).astype(dtype=np.float64, copy=False)

    s_w = 1.013 / (1. + np.exp(-8.6 * (r_p - 0.57)))

    _helper = lambda a, b, c: a * s_w / ((f - b) ** 2 + c * s_w)

    h_w = 1.66 * (
        1. +
        _helper(1.39, 22.235, 2.56) +
        _helper(3.37, 183.31, 4.69) +
        _helper(1.58, 325.1, 2.89)
        )

    return h_w


def slant_attenuation_annex2(gamma_dry, gamma_wet, h_0, h_w, elevation):
    '''
    Calculate simple path attenuation for a slant path through full atmosphere.
    (P.676-10: 28])

    Parameters
    ----------
    freq_grid - Frequencies (GHz)
    gamma_dry - Specific attenuation for dry air (dB / km)
    gamma_wet - Specific attenuation for wet air (dB / km)
    h_0 - Equivalent height for dry air (km)
    h_w - Equivalent height for wet air (km)
    elevation - Elevation of source as seen from observer (degrees)

    Returns
    -------
    Total attenuation along path (dB)
    '''

    AM = 1. / np.sin(np.radians(elevation))
    return AM * (gamma_dry * h_0 + gamma_wet * h_w)
