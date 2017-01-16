#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

# from functools import partial, lru_cache
import os
from collections import namedtuple
from functools import lru_cache
from astropy import units as apu
import numpy as np
from . import heightprofile
from . import helper
from .. import conversions as cnv
from .. import atm
from .. import helpers
import ipdb


__all__ = [
    'PathProps', 'path_properties', 'path_properties_with_units',
    'free_space_loss_bfsg',
    'tropospheric_scatter_loss_bs',
    'ducting_loss_ba',
    ]


@lru_cache(maxsize=1000, typed=False)
def _cached_specific_attenuation_annex1(
        freq, pressure_dry, pressure_water, temperature
        ):
    '''Helper to allow faster vectorization'''

    # Note: freq has to be a scalar!
    return atm.atm._specific_attenuation_annex1(
        freq, pressure_dry, pressure_water, temperature
        )


@lru_cache(maxsize=1000, typed=False)
def _cached_specific_attenuation_annex2(
        freq, pressure, rho_water, temperature
        ):
    '''Helper to allow faster vectorization'''

    # Note: freq has to be a scalar!
    return atm.atm._specific_attenuation_annex2(
        freq, pressure, rho_water, temperature
        )


_vectorized_specific_attenuation_annex1 = np.vectorize(
    _cached_specific_attenuation_annex1
    )


_vectorized_specific_attenuation_annex2 = np.vectorize(
    _cached_specific_attenuation_annex2
    )


def J_bull(nu):

    return 6.9 + 20 * np.log10(
        np.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1
        )


_PATH_PROPS = (  # pc = path center
    ('lon_mid', apu.deg),
    ('lat_mid', apu.deg),
    ('delta_N', cnv.dimless / apu.km),
    ('N0', cnv.dimless),  # Sea-level surface refractivity at pc [N-units]
    ('distance', apu.km),
    ('bearing', apu.deg),
    ('back_bearing', apu.deg),
    ('a_e', apu.km),
    ('h0', apu.m),
    ('hn', apu.m),
    ('h_ts', apu.m),
    ('h_rs', apu.m),
    ('h_st', apu.m),
    ('h_sr', apu.m),
    ('h_std', apu.m),
    ('h_srd', apu.m),
    ('h_te', apu.m),
    ('h_re', apu.m),
    ('h_m', apu.m),
    ('d_lt', apu.km),
    ('d_lr', apu.km),
    ('theta_t', apu.mrad),
    ('theta_r', apu.mrad),
    ('theta', apu.mrad),
    ('nu_bull', cnv.dimless),
    # ('hprof_dist', apu.km),  # distances of height profile
    # ('hprof_heights', apu.m),  # heights of height profile
    ('path_type', None),  # (0 - LOS, 1 - transhorizon)
    )

PathProps = namedtuple('PathProps', (tup[0] for tup in _PATH_PROPS))
PathPropsUnits = tuple(tup[1] for tup in _PATH_PROPS)
# PathProps.__new__.__defaults__ = (None,) * len(PathProps._fields)

PathProps.__doc__ = '''
    lon_mid - Path center longitude [deg]
    lat_mid - Path center latitude [deg]
    delta_N - Average radio-refractive index lapse-rate through the
        lowest 1 km of the atmosphere [N-units/km]
    N0 - Sea-level surface refractivity at path center [N-units]
    distance - Distance between transmitter and receiver [km]
    bearing - Bearing from transmitter to receiver [deg]
    back_bearing - Bearing from receiver to transmitter [deg]
    a_e - Median effective Earth radius at path center [km]
    h0 - Profile height at transmitter position [m]
    hn - Profile height at receiver position [m]
    h_ts - Transmitter antenna center height amsl [m]
    h_rs - Receiver antenna center height amsl [m]
    h_st - Height amsl, of the smooth-Earth surface at the transmitter [m]
    h_sr - Height amsl, of the smooth-Earth surface at the receiver [m]
    h_std - Smooth-Earth height amsl, at the transmitter [m]
    h_srd - Smooth-Earth height amsl, at the receiver [m]
    h_te - Effective heights of transmitter antennas above terrain [m]
    h_re - Effective heights of receiver antennas above terrain [m]
    h_m - Terrain roughness [m]
    d_lt - Distance to horizon (for transmitter) [km]
    d_lr - Distance to horizon (for receiver) [km]
        (For a LoS path, use distance to Bullington point, inferred from
        diffraction method for 50% time.)
    theta_t - For a transhorizon path, transmit horizon elevation angle;
        for a LoS path, the elevation angle to the receiver terminal [mrad]
    theta_r - For a transhorizon path, receiver horizon elevation angle;
        for a LoS path, the elevation angle to the transmitter terminal [mrad]
    theta - Path angular distance [mrad]
    nu_bull - Bullington-point diffraction parameter (for transhorizon)
        or highest diffraction parameter of the profile (for LOS) [dimless]
    path_type - 0: LOS; 1: transhorizon

    (amsl - above mean sea level)
    '''


def _smooth_earth_heights(distances, heights):

    d = distances[-1]
    d_i = distances[1:]
    d_im1 = distances[:-1]
    h_i = heights[1:]
    h_im1 = heights[:-1]

    nu_1 = np.sum(
        (d_i - d_im1) * (h_i + h_im1)
        )
    nu_2 = np.sum(
        (d_i - d_im1) * (
            h_i * (2 * d_i + d_im1) + h_im1 * (d_i + 2 * d_im1)
            )
        )

    h_st = (2 * nu_1 * d - nu_2) / d ** 2
    h_sr = (nu_2 - nu_1 * d) / d ** 2

    h_si = ((d - distances) * h_st + distances * h_sr) / d

    return h_st, h_sr, h_si


def _effective_antenna_heights(distances, heights, h_ts, h_rs, h_st, h_sr):

    h0 = heights[0]
    hn = heights[-1]
    d = distances[-1]
    d_i = distances[1:-1]
    h_i = heights[1:-1]

    H_i = h_i - (h_ts * (d - d_i) + h_rs * d_i) / d
    h_obs = np.max(H_i)
    alpha_obt = np.max(H_i / d_i)
    alpha_obr = np.max(H_i / (d - d_i))

    if h_obs < 0.:
        h_stp = h_st
        h_srp = h_sr
    else:
        g_t = alpha_obt / (alpha_obt + alpha_obr)
        g_r = alpha_obr / (alpha_obt + alpha_obr)

        h_stp = h_st - h_obs * g_t
        h_srp = h_sr - h_obs * g_r

    if h_stp > h0:
        h_std = h0
    else:
        h_std = h_stp

    if h_srp > hn:
        h_srd = hn
    else:
        h_srd = h_srp

    return h_std, h_srd


@helpers.ranged_quantity_input(
    freq=(1.e-30, None, apu.GHz),
    lon_t=(0, 360, apu.deg),
    lat_t=(-90, 90, apu.deg),
    lon_r=(0, 360, apu.deg),
    lat_r=(-90, 90, apu.deg),
    h_tg=(0, None, apu.m),
    h_rg=(0, None, apu.m),
    hprof_step=(0, None, apu.m),
    strip_input_units=True,
    output_unit=None,
    )
def path_properties(
        freq,
        lon_t, lat_t,
        lon_r, lat_r,
        h_tg, h_rg,
        hprof_step,
        ):
    pathprops = {}

    lam = 299792458.0 / freq  # wavelength in meter

    (
        lons,
        lats,
        distances,
        heights,
        bearing,
        back_bearing,
        distance,
        ) = heightprofile._srtm_height_profile(
            lon_t, lat_t, lon_r, lat_r, hprof_step
            )

    pathprops['distance'] = distance
    pathprops['bearing'] = bearing
    pathprops['back_bearing'] = back_bearing

    mid_idx = lons.size // 2
    # Note, for *very* few profile points, this is somewhat inaccurate
    # but, for real-world applications, this won't matter
    # (for even lenght, one would need to calculate average (non-trivial on
    # sphere!))
    lon_mid, lat_mid = lons[mid_idx], lats[mid_idx]
    pathprops['lon_mid'] = lon_mid
    pathprops['lat_mid'] = lat_mid

    delta_N, N0 = helper._N_from_map(lon_mid, lat_mid)
    pathprops['delta_N'] = delta_N
    pathprops['N0'] = N0

    h0 = heights[0]
    hn = heights[-1]
    pathprops['h0'] = h0
    pathprops['hn'] = hn
    h_ts = h0 + h_tg
    h_rs = hn + h_rg
    pathprops['h_ts'] = h_ts
    pathprops['h_rs'] = h_rs

    a_e = helper._median_effective_earth_radius(lon_mid, lat_mid)
    pathprops['a_e'] = a_e

    d = distance
    d_i = distances[1:-1]
    h_i = heights[1:-1]

    theta_i = 1000. * np.arctan(
        (h_i - h_ts) / 1.e3 / d_i - d_i / 2. / a_e
        )
    lt_idx = np.argmax(theta_i)
    theta_max = theta_i[lt_idx]
    theta_td = 1000. * np.arctan(
        (h_rs - h_ts) / 1.e3 / d - d / 2. / a_e
        )

    path_type = 1 if theta_max > theta_td else 0
    pathprops['path_type'] = path_type

    # alternative method from Diffraction analysis (Bullington point)
    # are they consistent???
    C_e500 = 500. / a_e
    # not quite sure, why we don't use theta_i here?
    slope_i = (
        h_i - h_ts +
        C_e500 * d_i * (d - d_i)
        ) / d_i
    S_tim = np.max(slope_i)
    S_tr = slope_i[-1]

    # For LOS this leads to inconsistency???
    # assert (theta_max > theta_td) == (S_tim >= S_tr), (
    #     'whoops, inconsistency in P.452???'
    #     )
    # works, if we modify the latter
    assert (theta_max > theta_td) == (S_tim > S_tr), (
        'whoops, inconsistency in P.452???'
        )

    if path_type == 1:
        # transhorizon

        theta_t = theta_max
        d_lt = d_i[lt_idx]

        theta_j = 1000. * np.arctan(
            (h_i - h_rs) / 1.e3 / (d - d_i) -
            (d - d_i) / 2. / a_e
            )
        lr_idx = np.argmax(theta_j)
        theta_r = theta_j[lr_idx]
        d_lr = d - d_i[lr_idx]

        theta = 1.e3 * d / a_e + theta_t + theta_r

        # find Bullington point, etc.
        slope_j = (
            h_i - h_rs +
            C_e500 * (d - d_i) * d_i
            ) / (d - d_i)
        S_rim = np.max(slope_j)
        d_bp = (h_rs - h_ts + S_rim * d) / (S_tim + S_rim)

        nu_bull = (
            h_ts + S_tim * d_bp - (
                h_ts * (d - d_bp) + h_rs * d_bp
                ) / d
            ) * np.sqrt(
                0.002 * d / lam / d_bp / (d - d_bp)
                )

    else:
        # LOS

        theta_t = theta_td

        theta_r = 1000. * np.arctan(
            # h_rs <-> h_ts
            (h_ts - h_rs) / 1.e3 / d - d / 2. / a_e
            )

        theta = 1.e3 * d / a_e + theta_t + theta_r  # is this correct?

        # find Bullington point, etc.

        # diffraction parameter
        nu_i = (
            h_i - h_ts +
            C_e500 * d_i * (d - d_i) -
            (h_ts * (d - d_i) + h_rs * d_i) / d
            ) * np.sqrt(
                0.002 * d / lam / d_i / (d - d_i)
                )

        bull_idx = np.argmax(nu_i)
        nu_bull = nu_i[bull_idx]

        # horizon distance for LOS paths has to be set to distance to
        # Bullington point in diffraction method
        d_lt = d_i[bull_idx]
        d_lr = d - d_i[bull_idx]

    pathprops['theta_t'] = theta_t
    pathprops['theta_r'] = theta_r
    pathprops['theta'] = theta
    pathprops['d_lt'] = d_lt
    pathprops['d_lr'] = d_lr
    pathprops['nu_bull'] = nu_bull

    # smooth-earth height profile
    h_st, h_sr, h_si = _smooth_earth_heights(distances, heights)

    # effective antenna heights for diffraction model
    h_std, h_srd = _effective_antenna_heights(
        distances, heights, h_ts, h_rs, h_st, h_sr
        )
    pathprops['h_std'] = h_std
    pathprops['h_srd'] = h_srd

    # parameters for ducting/layer-reflection model
    h_st = min(h_st, h0)  # use these only for ducting
    h_sr = min(h_sr, hn)  # or also for smooth-earth?
    pathprops['h_st'] = h_st
    pathprops['h_sr'] = h_sr

    m = (h_sr - h_st) / d

    h_te = h_tg + h0 - h_st
    h_re = h_rg + hn - h_sr
    pathprops['h_te'] = h_te
    pathprops['h_re'] = h_re

    if path_type == 1:
        # transhorizon
        _sl = slice(lt_idx, lr_idx+1)
        h_m = np.max(h_i[_sl] - (h_st + m * d_i[_sl]))

    else:
        # LOS
        # it seems, that h_m is calculated just from the profile height
        # at the Bullington point???
        h_m = h_i[bull_idx] - (h_st + m * d_i[bull_idx])

    pathprops['h_m'] = h_m

    return PathProps(**pathprops)


@helpers.ranged_quantity_input(
    freq=(1.e-30, None, apu.GHz),
    lon_t=(0, 360, apu.deg),
    lat_t=(-90, 90, apu.deg),
    lon_r=(0, 360, apu.deg),
    lat_r=(-90, 90, apu.deg),
    h_tg=(0, None, apu.m),
    h_rg=(0, None, apu.m),
    hprof_step=(0, None, apu.m),
    strip_input_units=False,
    output_unit=PathPropsUnits
    )
def path_properties_with_units(
        freq,
        lon_t, lat_t,
        lon_r, lat_r,
        h_tg, h_rg,
        hprof_step,
        ):
    '''
    TODO
    '''

    return path_properties(
        freq,
        lon_t, lat_t,
        lon_r, lat_r,
        h_tg, h_rg,
        hprof_step,
        )


@helpers.ranged_quantity_input(
    freq=(1.e-30, None, apu.GHz),
    omega=(0, 100, apu.percent),
    temperature=(0, None, apu.K),
    pressure=(0, None, apu.hPa),
    time_percent=(0, 100, apu.percent),
    strip_input_units=True, output_unit=cnv.dB
    )
def free_space_loss_bfsg(
        pathprop,
        freq,
        omega,
        temperature,
        pressure,
        time_percent,
        atm_method='annex2',
        ):
    '''
    Calculate the free space loss, L_bfsg, of a propagating radio wave
    according to ITU-R P.452-16 Eq. (8-12).

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    pathprop - PathProps object, obtained from path_properties function.
        (See PathProps documentation.)
    freq - Frequency of radiation [GHz]
    omega - Fraction of the path over water [%] (see Table 3)
    temperature - Ambient temperature in relevant layer [K]
    pressure - Total air pressure (dry + wet) in relevant layer [hPa]
    atm_method - Which annex to use for atm model P.676, ['annex1'/'annex2']
    time_percent - Time percentage [%]
        (for average month, this is just p, for worst-month, this is β_0)

    Returns
    -------
    L_bfsg - Free-space loss [dB]
        This contains either E_sp or E_sβ, depending on which quantity was
        used for time_percent.

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        path_properties helper function.
    - This is similar to conversions.free_space_loss function but additionally
      accounts for athmospheric absorption and corrects for focusing and
      multipath effects.
      '''

    dist = pathprop.distance
    d_lt = pathprop.d_lt
    d_lr = pathprop.d_lr

    assert atm_method in ['annex1', 'annex2'], (
        'atm_method must be one of "annex1" or "annex2"'
        )

    # bin omega to improve specific_attenuation caching
    omega_b = np.int32(omega + 0.5)

    rho_water = 7.5 + 2.5 * omega_b / 100.
    pressure_water = rho_water * temperature / 216.7
    pressure_dry = pressure - pressure_water

    if atm_method == 'annex1':
        atten_dry_dB, atten_wet_dB = _vectorized_specific_attenuation_annex1(
            freq, pressure_dry, pressure_water, temperature
            )
    else:
        atten_dry_dB, atten_wet_dB = _vectorized_specific_attenuation_annex2(
            freq, pressure, rho_water, temperature
            )

    A_g = (atten_dry_dB + atten_wet_dB) * dist

    # better use Eq. (8) for full consistency
    # L_bfsg = cnv.conversions._free_space_loss(freq, dist)  # negative dB
    # L_bfsg = - 10. * np.log10(L_bfsg)  # positive dB
    L_bfsg = 92.5 + 20 * np.log10(freq) + 20 * np.log10(dist)
    L_bfsg += A_g

    L_bfsg += 2.6 * (
        1. - np.exp(-0.1 * (d_lt + d_lr))
        ) * np.log10(time_percent / 50.)

    return L_bfsg


@helpers.ranged_quantity_input(
    freq=(1.e-30, None, apu.GHz),
    temperature=(0, None, apu.K),
    pressure=(0, None, apu.hPa),
    G_t=(0, None, cnv.dBi),
    G_r=(0, None, cnv.dBi),
    time_percent=(0, 100, apu.percent),
    strip_input_units=True, output_unit=cnv.dB
    )
def tropospheric_scatter_loss_bs(
        pathprop,
        freq,
        temperature,
        pressure,
        G_t,
        G_r,
        time_percent,
        atm_method='annex2',
        ):
    '''
    Calculate the tropospheric scatter loss, L_bs, of a propagating radio wave
    according to ITU-R P.452-16 Eq. (45).

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    pathprop - PathProps object, obtained from path_properties function.
        (See PathProps documentation.)
    freq - Frequency of radiation [GHz]
    temperature - Ambient temperature in relevant layer [K]
    pressure - Total air pressure (dry + wet) in relevant layer [hPa]
    G_t, G_r - Antenna gain (transmitter, receiver) in the direction of the
        horizon(!) along the great-circle interference path [dBi]
    time_percent - Time percentage [%]
        (for average month, this is just p, for worst-month, this is β_0)
    atm_method - Which annex to use for atm model P.676, ['annex1'/'annex2']

    Returns
    -------
    L_bs - Tropospheric scatter loss [dB]

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        path_properties helper function.
    '''

    dist = pathprop.distance
    N_0 = pathprop.N0
    a_e = pathprop.a_e
    theta_t = pathprop.theta_t
    theta_r = pathprop.theta_r

    assert atm_method in ['annex1', 'annex2'], (
        'atm_method must be one of "annex1" or "annex2"'
        )

    rho_water = 3.
    pressure_water = rho_water * temperature / 216.7
    pressure_dry = pressure - pressure_water

    if atm_method == 'annex1':
        atten_dry_dB, atten_wet_dB = _vectorized_specific_attenuation_annex1(
            freq, pressure_dry, pressure_water, temperature
            )
    else:
        atten_dry_dB, atten_wet_dB = _vectorized_specific_attenuation_annex2(
            freq, pressure, rho_water, temperature
            )

    A_g = (atten_dry_dB + atten_wet_dB) * dist
    L_f = 25 * np.log10(freq) - 2.5 * np.log10(0.5 * freq) ** 2

    # TODO: why is toposcatter depending on gains towards horizon???
    L_c = 0.051 * np.exp(0.055 * (G_t + G_r))

    # theta is in mrad
    theta = 1e3 * dist / a_e + theta_t + theta_r

    L_bs = (
        190. + L_f + 20 * np.log10(dist) + 0.573 * theta -
        0.15 * N_0 + L_c + A_g - 10.1 * (-np.log10(time_percent / 50.)) ** 0.7
        )

    return L_bs


@helpers.ranged_quantity_input(
    freq=(1.e-30, None, apu.GHz),
    omega=(0, 100, apu.percent),
    temperature=(0, None, apu.K),
    pressure=(0, None, apu.hPa),
    d_ct=(0, None, apu.km),
    d_cr=(0, None, apu.km),
    d_lm=(0, None, apu.km),
    G_t=(0, None, cnv.dBi),
    G_r=(0, None, cnv.dBi),
    beta_0=(0, 100, apu.percent),
    time_percent=(0, 100, apu.percent),
    strip_input_units=True, output_unit=cnv.dB
    )
def ducting_loss_ba(
        pathprop,
        freq,
        omega,
        temperature,
        pressure,
        d_ct,
        d_cr,
        d_lm,
        G_t,
        G_r,
        beta_0,
        time_percent,
        atm_method='annex2',
        ):
    '''
    Calculate the ducting/layer reflection loss, L_ba, of a propagating radio
    wave according to ITU-R P.452-16 Eq. (46-56).

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    pathprop - PathProps object, obtained from path_properties function.
        (See PathProps documentation.)
    freq - Frequency of radiation [GHz]
    omega - Fraction of the path over water [%] (see Table 3)
    temperature - Ambient temperature in relevant layer [K]
    pressure - Total air pressure (dry + wet) in relevant layer [hPa]
    d_ct, d_cr - Distance over land from transmit/receive antenna to the coast
        along great circle interference path [km]
        (set to zero for terminal on ship/sea platform; only relevant if less
        than 5 km)
    d_lm - Longest continuous inland section of the great-circle path [km]
    G_t, G_r - Antenna gain (transmitter, receiver) in the direction of the
        horizon(!) along the great-circle interference path [dBi]
    beta_0 - The time percentage for which refractive index lapse-rates
        exceeding 100 N-units/km can be expected in the first 100 m
        of the lower atmosphere [%]
    time_percent - Time percentage [%]
        (for average month, this is just p, for worst-month, this is β_0)
    atm_method - Which annex to use for atm model P.676, ['annex1'/'annex2']

    Returns
    -------
    L_ba - Ducting/layer reflection loss [dB]

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        path_properties helper function.
    '''

    dist = pathprop.distance
    # N_0 = pathprop.N0
    a_e = pathprop.a_e
    h_ts = pathprop.h_ts
    h_rs = pathprop.h_rs
    h_te = pathprop.h_te
    h_re = pathprop.h_re
    h_m = pathprop.h_m
    d_lt = pathprop.d_lt
    d_lr = pathprop.d_lr
    theta_t = pathprop.theta_t
    theta_r = pathprop.theta_r

    assert atm_method in ['annex1', 'annex2'], (
        'atm_method must be one of "annex1" or "annex2"'
        )

    # bin omega to improve specific_attenuation caching
    omega_b = np.int32(omega + 0.5)

    rho_water = 7.5 + 2.5 * omega_b / 100.
    pressure_water = rho_water * temperature / 216.7
    pressure_dry = pressure - pressure_water

    if atm_method == 'annex1':
        atten_dry_dB, atten_wet_dB = _vectorized_specific_attenuation_annex1(
            freq, pressure_dry, pressure_water, temperature
            )
    else:
        atten_dry_dB, atten_wet_dB = _vectorized_specific_attenuation_annex2(
            freq, pressure, rho_water, temperature
            )

    theta_t_prime = np.where(theta_t <= 0.1 * d_lt, theta_t, 0.1 * d_lt)
    theta_r_prime = np.where(theta_r <= 0.1 * d_lr, theta_r, 0.1 * d_lr)

    theta_t_prime2 = theta_t - 0.1 * d_lt
    theta_r_prime2 = theta_r - 0.1 * d_lr

    theta_prime = 1e3 * dist / a_e + theta_t_prime + theta_r_prime

    gamma_d = 5.e-5 * a_e * np.power(freq, 1. / 3.)

    tau = 1. - np.exp(-4.12e-4 * np.power(d_lm, 2.41))
    eps = 3.5
    alpha = -0.6 - eps * 1.e-9 * np.power(dist, 3.1) * tau
    alpha = np.where(alpha < -3.4, -3.4, alpha)

    mu_2 = np.power(
        500. * dist ** 2 / a_e / (np.sqrt(h_te) + np.sqrt(h_re)) ** 2,
        alpha
        )
    mu_2 = np.where(mu_2 > 1., 1., mu_2)

    d_I = dist - d_lt - d_lr
    d_I = np.where(d_I > 40, 40, d_I)

    mu_3 = np.where(
        h_m <= 10.,
        1.,
        np.exp(-4.6e-5 * (h_m - 10.) * (43. + 6. * d_I))

        )

    beta = beta_0 * mu_2 * mu_3

    Gamma = 1.076 / np.power(2.0058 - np.log10(beta), 1.012) * np.exp(
        -(9.51 - 4.8 * np.log10(beta) + 0.198 * np.log10(beta) ** 2) *
        1.e-6 * np.power(dist, 1.13)
        )

    A_lf = np.where(
        freq < 0.5,
        45.375 - 137. * freq + 92.5 * freq ** 2,
        0.,
        )

    def A_s(f, d_l, theta_prime2):

        return np.where(
            theta_prime2 <= 0,
            0.,
            20 * np.log10(1 + 0.361 * theta_prime2 * np.sqrt(f * d_l)) +
            0.264 * theta_prime2 * np.power(f, 1. / 3.)
            )

    A_st = A_s(freq, d_lt, theta_t_prime2)
    A_sr = A_s(freq, d_lr, theta_r_prime2)

    def A_c(d_l, d_c, h_s, om):

        return np.where(
            (om >= 0.75) & (d_c <= d_l) & (d_c <= 5.),
            -3 * np.exp(-0.25 * d_c ** 2) * (1. + np.tanh(3.5 - 0.07 * h_s)),
            0.
            )

    A_ct = A_c(d_lt, d_ct, h_ts, omega)
    A_cr = A_c(d_lr, d_cr, h_rs, omega)

    A_f = (
        102.45 + 20 * np.log10(freq) + 20 * np.log10(d_lt + d_lr) +
        A_lf + A_st + A_sr + A_ct + A_cr
        )

    A_p = (
        -12 +
        (1.2 + 3.7e-3 * dist) * np.log10(time_percent / beta) +
        12. * np.power(time_percent / beta, Gamma)
        )

    A_d = gamma_d * theta_prime + A_p

    A_g = (atten_dry_dB + atten_wet_dB) * dist

    return A_f + A_d + A_g


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
