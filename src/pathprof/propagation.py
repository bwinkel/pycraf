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

    if nu < -0.78:
        return 0.
    else:
        return (
            6.9 + 20 * np.log10(
                np.sqrt((nu - 0.1) ** 2 + 1) + nu - 0.1
                )
            )


_PATH_PROPS = (  # pc = path center

    ('freq', apu.GHz),
    ('wavelen', apu.m),
    ('p', apu.percent),
    ('beta0', apu.percent),
    ('omega', apu.percent),
    ('lon_mid', apu.deg),
    ('lat_mid', apu.deg),
    ('delta_N', cnv.dimless / apu.km),
    ('N0', cnv.dimless),  # Sea-level surface refractivity at pc [N-units]
    ('distance', apu.km),
    ('bearing', apu.deg),
    ('back_bearing', apu.deg),
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
    ('d_lm', apu.km),
    ('d_tm', apu.km),
    ('d_ct', apu.km),
    ('d_cr', apu.km),
    # ('hprof_dist', apu.km),  # distances of height profile
    # ('hprof_heights', apu.m),  # heights of height profile
    ('a_e', apu.km),
    ('path_type', None),  # (0 - LOS, 1 - transhorizon)
    ('theta_t', apu.mrad),
    ('theta_r', apu.mrad),
    ('theta', apu.mrad),
    ('d_lt', apu.km),
    ('d_lr', apu.km),
    ('nu_bull', cnv.dimless),
    ('h_m', apu.m),
    ('a_e_b0', apu.km),
    ('path_type_b0', None),  # (0 - LOS, 1 - transhorizon)
    ('theta_t_b0', apu.mrad),
    ('theta_r_b0', apu.mrad),
    ('theta_b0', apu.mrad),
    ('d_lt_b0', apu.km),
    ('d_lr_b0', apu.km),
    ('nu_bull_b0', cnv.dimless),
    ('h_m_b0', apu.m),
    )

PathProps = namedtuple('PathProps', (tup[0] for tup in _PATH_PROPS))
PathPropsUnits = tuple(tup[1] for tup in _PATH_PROPS)
# PathProps.__new__.__defaults__ = (None,) * len(PathProps._fields)

PathProps.__doc__ = '''
    freq - Frequency [GHz]
    wavelen - Wavelength [m]
    p - Time percent [%]
    beta0 - the time percentage for which refractive index lapse-rates
        exceeding 100 N-units/km can be expected in the first 100 m
        of the lower atmosphere [%]
    omega - Fraction of the path over water [%] (see Table 3)
    lon_mid - Path center longitude [deg]
    lat_mid - Path center latitude [deg]
    delta_N - Average radio-refractive index lapse-rate through the
        lowest 1 km of the atmosphere [N-units/km]
    N0 - Sea-level surface refractivity at path center [N-units]
    distance - Distance between transmitter and receiver [km]
    bearing - Bearing from transmitter to receiver [deg]
    back_bearing - Bearing from receiver to transmitter [deg]
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
    d_tm - longest continuous land (inland + coastal) section of the
        great-circle path [km]
    d_lm - longest continuous inland section of the great-circle path [km]
    d_ct, d_cr - Distance over land from transmit/receive antenna to the coast
        along great circle interference path [km]
        (set to zero for terminal on ship/sea platform; only relevant if less
        than 5 km)

    --- The values below are present twice, once without subscript
    --- for the median Earth radius and once with subscript "_b0" for
    --- the beta_0 "version" of the Earth radius
    a_e[_beta0] - Median effective Earth radius at path center [km]
    path_type[_beta0] - 0: LOS; 1: transhorizon
    theta_t[_beta0] - For a transhorizon path, transmit horizon elevation
        angle; for a LoS path, the elevation angle to the receiver
        terminal [mrad]
    theta_r[_beta0] - For a transhorizon path, receiver horizon elevation
        angle; for a LoS path, the elevation angle to the transmitter
        terminal [mrad]
    theta[_beta0] - Path angular distance [mrad]
    d_lt[_beta0] - Distance to horizon (for transmitter) [km]
    d_lr[_beta0] - Distance to horizon (for receiver) [km]
    nu_bull[_beta0] - Bullington-point diffraction parameter (for
        transhorizon) or highest diffraction parameter of the profile (for
        LOS) [dimless]
    h_m[_beta0] - Terrain roughness [m]
        (For a LoS path, use distance to Bullington point, inferred from
        diffraction method for 50% time.)

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


def _diffraction_helper(
        a_p,
        distances,
        heights,
        distance,
        h_ts,
        h_rs,
        h_st,
        duct_slope,
        wavelen,
        ):

    lam = wavelen
    d = distance
    d_i = distances[1:-1]
    h_i = heights[1:-1]
    m = duct_slope

    theta_i = 1000. * np.arctan(
        (h_i - h_ts) / 1.e3 / d_i - d_i / 2. / a_p
        )
    lt_idx = np.argmax(theta_i)
    theta_max = theta_i[lt_idx]
    theta_td = 1000. * np.arctan(
        (h_rs - h_ts) / 1.e3 / d - d / 2. / a_p
        )

    path_type = 1 if theta_max > theta_td else 0

    # alternative method from Diffraction analysis (Bullington point)
    # are they consistent???
    C_e500 = 500. / a_p
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
            (d - d_i) / 2. / a_p
            )
        lr_idx = np.argmax(theta_j)
        theta_r = theta_j[lr_idx]
        d_lr = d - d_i[lr_idx]

        theta = 1.e3 * d / a_p + theta_t + theta_r

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
            (h_ts - h_rs) / 1.e3 / d - d / 2. / a_p
            )

        theta = 1.e3 * d / a_p + theta_t + theta_r  # is this correct?

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

    if path_type == 1:
        # transhorizon
        _sl = slice(lt_idx, lr_idx + 1)
        h_m = np.max(h_i[_sl] - (h_st + m * d_i[_sl]))

    else:
        # LOS
        # it seems, that h_m is calculated just from the profile height
        # at the Bullington point???
        h_m = h_i[bull_idx] - (h_st + m * d_i[bull_idx])

    subprops = {}
    subprops['a_e'] = a_p
    subprops['path_type'] = path_type
    subprops['theta_t'] = theta_t
    subprops['theta_r'] = theta_r
    subprops['theta'] = theta
    subprops['d_lt'] = d_lt
    subprops['d_lr'] = d_lr
    subprops['nu_bull'] = nu_bull
    subprops['h_m'] = h_m

    return subprops


def path_properties(
        freq,
        lon_t, lat_t,
        lon_r, lat_r,
        h_tg, h_rg,
        hprof_step,
        time_percent,
        omega=0,
        d_tm=-1, d_lm=-1,
        d_ct=50000, d_cr=50000,
        ):
    '''
    Calculate path profile properties.

    Note: This is the unit-less version (for speed). Use
    path_properties_with_units() if you want astropy-units interface.

    Parameters
    ----------
    freq - Frequency of radiation [GHz]
    lon_t, lat_t - Transmitter coordinates [deg]
    lon_r, lat_r - Receiver coordinates [deg]
    h_tg, h_rg - Transmitter/receiver heights over ground [m]
    hprof_step - Distance resolution of height profile along path [m]
    time_percent - Time percentage [%]
    omega - Fraction of the path over water [%] (see Table 3)
    d_tm - longest continuous land (inland + coastal) section of the
        great-circle path [km]
    d_lm - longest continuous inland section of the great-circle path [km]
    d_ct, d_cr - Distance over land from transmit/receive antenna to the coast
        along great circle interference path [km]
        (set to zero for terminal on ship/sea platform; only relevant if less
        than 5 km)

    Returns
    -------
    Path Properties (as a namedtuple)
    '''

    pathprops = {}
    pathprops['freq'] = freq
    pathprops['p'] = time_percent

    lam = 0.299792458 / freq   # wavelength in meter
    pathprops['wavelen'] = lam

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

    if d_tm < 0:
        d_tm = distance
    if d_lm < 0:
        d_lm = distance
    # TODO: add functionality to produce next 5 parameters programmatically
    pathprops['d_tm'] = d_tm
    pathprops['d_lm'] = d_lm
    pathprops['omega'] = omega
    pathprops['d_ct'] = d_ct
    pathprops['d_cr'] = d_cr

    mid_idx = lons.size // 2
    # Note, for *very* few profile points, this is somewhat inaccurate
    # but, for real-world applications, this won't matter
    # (for even lenght, one would need to calculate average (non-trivial on
    # sphere!))
    lon_mid, lat_mid = lons[mid_idx], lats[mid_idx]
    pathprops['lon_mid'] = lon_mid
    pathprops['lat_mid'] = lat_mid

    delta_N, beta_0, N0 = helper._radiomet_data_for_pathcenter(
        lon_mid, lat_mid, d_tm, d_lm
        )
    pathprops['delta_N'] = delta_N
    pathprops['beta0'] = beta_0
    pathprops['N0'] = N0

    d = distance
    # d_i = distances[1:-1]
    # h_i = heights[1:-1]
    h0 = heights[0]
    hn = heights[-1]
    pathprops['h0'] = h0
    pathprops['hn'] = hn
    h_ts = h0 + h_tg
    h_rs = hn + h_rg
    pathprops['h_ts'] = h_ts
    pathprops['h_rs'] = h_rs

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

    duct_slope = (h_sr - h_st) / d  # == m

    h_te = h_tg + h0 - h_st
    h_re = h_rg + hn - h_sr
    pathprops['h_te'] = h_te
    pathprops['h_re'] = h_re

    # the next part depends on whether the median or beta_0 Earth radius
    # is to be used; to avoid running the whole function twice
    # we just add all related quantities twice with subscript

    args = (
        distances,
        heights,
        distance,
        h_ts,
        h_rs,
        h_st,
        duct_slope,
        lam,
        )

    a_e_50 = helper._median_effective_earth_radius(lon_mid, lat_mid)
    subprops_50 = _diffraction_helper(a_e_50, *args)
    a_e_b0 = helper.A_BETA_VALUE
    subprops_b0 = _diffraction_helper(a_e_b0, *args)

    # subprops_50 = dict((k + '_50', v) for k, v in subprops_50.items())
    subprops_b0 = dict((k + '_b0', v) for k, v in subprops_b0.items())
    pathprops.update(subprops_50)
    pathprops.update(subprops_b0)

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
    d_tm=(0, None, apu.km),
    d_lm=(0, None, apu.km),
    d_ct=(0, None, apu.km),
    d_cr=(0, None, apu.km),
    time_percent=(0, 100, apu.percent),
    strip_input_units=True,
    output_unit=PathPropsUnits
    )
def path_properties_with_units(
        freq,
        lon_t, lat_t,
        lon_r, lat_r,
        h_tg, h_rg,
        hprof_step,
        time_percent,
        omega=0. * apu.percent,
        d_tm=-1 * apu.km, d_lm=-1 * apu.km,
        d_ct=50000 * apu.km, d_cr=50000 * apu.km,
        ):
    '''
    Calculate path profile properties.

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    freq - Frequency of radiation [GHz]
    lon_t, lat_t - Transmitter coordinates [deg]
    lon_r, lat_r - Receiver coordinates [deg]
    h_tg, h_rg - Transmitter/receiver heights over ground [m]
    hprof_step - Distance resolution of height profile along path [m]
    time_percent - Time percentage [%]
    omega - Fraction of the path over water [%] (see Table 3)
    d_tm - longest continuous land (inland + coastal) section of the
        great-circle path [km]
    d_lm - longest continuous inland section of the great-circle path [km]
    d_ct, d_cr - Distance over land from transmit/receive antenna to the coast
        along great circle interference path [km]
        (set to zero for terminal on ship/sea platform; only relevant if less
        than 5 km)

    Returns
    -------
    Path Properties (as a namedtuple)
    '''

    return path_properties(
        freq,
        lon_t, lat_t,
        lon_r, lat_r,
        h_tg, h_rg,
        hprof_step,
        time_percent,
        omega,
        d_tm, d_lm,
        )


def _free_space_loss_bfsg(
        pathprop,
        temperature,
        pressure,
        atm_method='annex2',
        ):

    freq = pathprop.freq
    dist = pathprop.distance
    d_lt = pathprop.d_lt
    d_lr = pathprop.d_lr
    time_percent = pathprop.p
    beta0 = pathprop.beta0
    omega = pathprop.omega

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

    E_sp = 2.6 * (
        1. - np.exp(-0.1 * (d_lt + d_lr))
        ) * np.log10(time_percent / 50.)
    E_beta = 2.6 * (
        1. - np.exp(-0.1 * (d_lt + d_lr))
        ) * np.log10(beta0 / 50.)

    return L_bfsg, E_sp, E_beta


@helpers.ranged_quantity_input(
    temperature=(0, None, apu.K),
    pressure=(0, None, apu.hPa),
    strip_input_units=True, output_unit=(cnv.dB, cnv.dB, cnv.dB),
    )
def free_space_loss_bfsg(
        pathprop,
        temperature,
        pressure,
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
    temperature - Ambient temperature in relevant layer [K]
    pressure - Total air pressure (dry + wet) in relevant layer [hPa]
    atm_method - Which annex to use for atm model P.676, ['annex1'/'annex2']

    Returns
    -------
    (L_bfsg, E_sp, E_sβ) - tuple
        L_bfsg - Free-space loss [dB]
        E_sp - focussing/multipath correction for p% [dB]
        E_sβ - focussing/multipath correction for β0% [dB]

        with these, one can form
        L_b0p = L_bfsg + E_sp [dB]
        L_b0beta = L_bfsg + E_sβ [dB]

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        path_properties helper function.
    - This is similar to conversions.free_space_loss function but additionally
      accounts for athmospheric absorption and corrects for focusing and
      multipath effects.
      '''

    return _free_space_loss_bfsg(
        pathprop,
        temperature,
        pressure,
        atm_method=atm_method,
        )


def _tropospheric_scatter_loss_bs(
        pathprop,
        temperature,
        pressure,
        G_t,
        G_r,
        atm_method='annex2',
        ):

    freq = pathprop.freq
    dist = pathprop.distance
    N_0 = pathprop.N0
    a_e = pathprop.a_e
    theta_t = pathprop.theta_t
    theta_r = pathprop.theta_r
    time_percent = pathprop.p

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
    temperature=(0, None, apu.K),
    pressure=(0, None, apu.hPa),
    G_t=(0, None, cnv.dBi),
    G_r=(0, None, cnv.dBi),
    strip_input_units=True, output_unit=cnv.dB
    )
def tropospheric_scatter_loss_bs(
        pathprop,
        temperature,
        pressure,
        G_t,
        G_r,
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
    temperature - Ambient temperature in relevant layer [K]
    pressure - Total air pressure (dry + wet) in relevant layer [hPa]
    G_t, G_r - Antenna gain (transmitter, receiver) in the direction of the
        horizon(!) along the great-circle interference path [dBi]
    atm_method - Which annex to use for atm model P.676, ['annex1'/'annex2']

    Returns
    -------
    L_bs - Tropospheric scatter loss [dB]

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        path_properties helper function.
    '''

    return _tropospheric_scatter_loss_bs(
        pathprop,
        temperature,
        pressure,
        G_t,
        G_r,
        atm_method=atm_method,
        )


def _ducting_loss_ba(
        pathprop,
        temperature,
        pressure,
        G_t,
        G_r,
        atm_method='annex2',
        ):

    freq = pathprop.freq
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
    d_ct = pathprop.d_ct
    d_cr = pathprop.d_cr
    d_lm = pathprop.d_lm
    omega = pathprop.omega

    theta_t = pathprop.theta_t
    theta_r = pathprop.theta_r
    time_percent = pathprop.p
    beta_0 = pathprop.beta0

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


@helpers.ranged_quantity_input(
    temperature=(0, None, apu.K),
    pressure=(0, None, apu.hPa),
    G_t=(0, None, cnv.dBi),
    G_r=(0, None, cnv.dBi),
    strip_input_units=True, output_unit=cnv.dB
    )
def ducting_loss_ba(
        pathprop,
        temperature,
        pressure,
        G_t,
        G_r,
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
    temperature - Ambient temperature in relevant layer [K]
    pressure - Total air pressure (dry + wet) in relevant layer [hPa]
    G_t, G_r - Antenna gain (transmitter, receiver) in the direction of the
        horizon(!) along the great-circle interference path [dBi]
    atm_method - Which annex to use for atm model P.676, ['annex1'/'annex2']

    Returns
    -------
    L_ba - Ducting/layer reflection loss [dB]

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        path_properties helper function.
    '''

    return _ducting_loss_ba(
        pathprop,
        temperature,
        pressure,
        G_t,
        G_r,
        atm_method=atm_method,
        )


def _diffraction_bullington_helper(
        dist, nu_bull
        ):

    L_uc = J_bull(nu_bull)
    L_bull = L_uc + (1 - np.exp(-L_uc / 6.)) * (10 + 0.02 * dist)

    return L_bull


def _L_dft(a_dft, dist, freq, h_te, h_re, omega, pol):

    def _func(eps_r, sigma):

        assert pol in ['horizontal', 'vertical']

        K = 0.036 * np.power(a_dft * freq, -1. / 3.) * np.power(
            (eps_r - 1) ** 2 + (18. * sigma / freq) ** 2, -0.25
            )

        if pol == 'vertical':

            K *= np.power(eps_r ** 2 + (18. * sigma / freq) ** 2, 0.5)

        K2 = K * K
        K4 = K2 * K2
        beta_dft = (1. + 1.6 * K2 + 0.67 * K4) / (1. + 4.5 * K2 + 1.53 * K4)

        X = 21.88 * beta_dft * np.power(freq / a_dft ** 2, 1. / 3.) * dist
        Y = 0.9575 * beta_dft * np.power(freq ** 2 / a_dft, 1. / 3.)
        Y_t = Y * h_te
        Y_r = Y * h_re

        if X >= 1.6:
            F_X = 11. + 10 * np.log10(X) - 17.6 * X
        else:
            F_X = -20 * np.log10(X) - 5.6488 * np.power(X, 1.425)

        def G(Y):

            B = beta_dft * Y
            if B > 2:
                res = 17.6 * np.sqrt(B - 1.1) - 5 * np.log10(B - 1.1) - 8
            else:
                res = 20 * np.log10(B + 0.1 * B ** 3)

            return max(2 + 20 * np.log10(K), res)

        return -F_X - G(Y_t) - G(Y_r)

    L_dft_land = _func(22., 0.003)
    L_dft_sea = _func(80., 5.0)

    L_dft = omega * L_dft_sea + (1. - omega) * L_dft_land

    return L_dft


def _diffraction_spherical_earth_loss_helper(
        dist, freq, a_p, h_te, h_re, omega, pol
        ):

    wavelen = 0.299792458 / freq
    d_los = np.sqrt(2 * a_p) * (np.sqrt(0.001 * h_te) + np.sqrt(0.001 * h_re))

    if dist >= d_los:

        a_dft = a_p
        return _L_dft(a_dft, dist, freq, h_te, h_re, omega, pol)

    else:

        c = (h_te - h_re) / (h_te + h_re)
        m = 250. * dist ** 2 / a_p / (h_te + h_re)

        b = 2 * np.sqrt((m + 1.) / 3. / m) * np.cos(
            np.pi / 3. +
            1. / 3. * np.arccos(3. * c / 2. * np.sqrt(3. * m / (m + 1.) ** 3))
            )

        d_se1 = 0.5 * dist * (1. + b)
        d_se2 = dist - d_se1

        h_se = (
            (h_te - 500 * d_se1 ** 2 / a_p) * d_se2 +
            (h_re - 500 * d_se2 ** 2 / a_p) * d_se1
            ) / dist

        h_req = 17.456 * np.sqrt(d_se1 * d_se2 * wavelen / dist)

        if h_se > h_req:

            return 0.

        a_em = 500. * (dist / (np.sqrt(h_te) + np.sqrt(h_re))) ** 2
        a_dft = a_em

        L_dft = _L_dft(a_dft, dist, freq, h_te, h_re, omega, pol)

        if L_dft < 0:
            return 0.

        L_dsph = (1. - h_se / h_req) * L_dft
        return L_dsph


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
