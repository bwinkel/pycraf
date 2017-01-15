#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

# from functools import partial, lru_cache
import os
from functools import lru_cache
from astropy import units as apu
import numpy as np
from . import heightprofile
from .. import conversions as cnv
from .. import atm
from .. import helpers
import ipdb


__all__ = [
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
    output_unit=(
        apu.deg,  # lon_mid
        apu.deg,  # lat_mid
        apu.km,  # distance
        apu.deg,  # bearing
        apu.deg,  # backbearing
        apu.km,  # a_e
        apu.m,  # h_ts
        apu.m,  # h_rs
        apu.m,  # h_st
        apu.m,  # h_sr
        apu.m,  # h_te
        apu.m,  # h_re
        apu.m,  # h_m
        apu.km,  # d_lt
        apu.km,  # d_lr
        apu.mrad,  # theta_t
        apu.mrad,  # theta_r
        apu.mrad,  # theta
        apu.dimless,  # nu_b
        apu.km,  # distances of height profile
        apu.m,  # heights of height profile
        None,  # path type (0 - LOS, 1 - transhorizon)
        )
    )
def path_properties(
        freq,
        lon_t, lat_t,
        lon_r, lat_r,
        h_tg, h_rg,
        hprof_step,
        ):

    lam = 299792458.0 / freq  # wavelength in meter

    (
        lons,
        lats,
        distances,
        heights,
        bearing,
        back_bearing,
        distance,
        ) = heightprofile.srtm_height_profile(
            lon_t, lat_t, lon_r, lat_r, hprof_step
            )

    mid_idx = lons.size // 2
    # Note, for *very* few profile points, this is somewhat inaccurate
    # but, for real-world applications, this won't matter
    # (for even lenght, one would need to calculate average (non-trivial on
    # sphere!))
    lon_mid, lat_mid = lons[mid_idx], lats[mid_idx]

    h_ts = heights[0] + h_tg
    h_rs = heights[-1] + h_rg

    a_e = helpers.median_effective_earth_radius(lon_mid, lat_mid)

    d = distance
    d_i = distances[1:-1]
    h_i = heights[1:-1]

    theta_i = 1000. * np.arctan(
        (h_i - h_ts) / 1.e3 / d_i - d_i / 2. / a_e
        )
    max_idx = np.argmax(theta_i)
    theta_max = theta_i[max_idx]
    theta_td = 1000. * np.arctan(
        (h_rs - h_ts) / 1.e3 / d - d / 2. / a_e
        )

    path_type = 1 if theta_max > theta_td else 0

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

    assert (theta_max > theta_td) and (S_tim >= S_tr), (
        'whoops, inconsistency in P.452???'
        )

    if path_type == 1:
        # transhorizon

        theta_t = theta_max
        d_lt = d_i[max_idx]

        theta_j = 1000. * np.arctan(
            (h_i - h_rs) / 1.e3 / (d - d_i) -
            (d - d_i) / 2. / a_e
            )
        max_idx = np.argmax(theta_j)
        theta_r = theta_j[max_idx]
        d_lr = d - d_i[max_idx]

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
        # d_lt = distance to bullington point

        theta_r = 1000. * np.arctan(
            # h_rs <-> h_ts; distance --> -distance
            (h_rs - h_ts) / 1.e3 / d + d / 2. / a_e
            )

        theta = 1.e3 * d / a_e + theta_t + theta_r  # is this correct?

        # find Bullington point, etc.

        # diffraction parameter
        nu_i = (
            h_i - h_ts +
            C_e500 * d_i * (d - d_i) -
            (h_ts * (d - d_i) + h_rs * d_i ) / d
            ) * np.sqrt(
                0.002 * d / lam / d_i / (d - d_i)
                )

        nu_bull_idx = np.argmax(nu_i)
        nu_bull = nu_i(nu_bull_idx)

        # horizon distance for LOS paths has to be set to distance to
        # Bullington point in diffraction method
        d_lt = d_i[nu_bull_idx]
        d_lr = d - d_i[nu_bull_idx]






@helpers.ranged_quantity_input(
    dist=(1.e-30, None, apu.km),
    freq=(1.e-30, None, apu.GHz),
    omega=(0, 100, apu.percent),
    temperature=(0, None, apu.K),
    pressure=(0, None, apu.hPa),
    d_lt=(0, None, apu.km),
    d_lr=(0, None, apu.km),
    time_percent=(0, 100, apu.percent),
    strip_input_units=True, output_unit=cnv.dB
    )
def free_space_loss_bfsg(
        dist,
        freq,
        omega,
        temperature,
        pressure,
        atm_method='annex1',
        do_corrections=False,
        d_lt=0. * apu.km,  # Note: this gets stripped by decorator
        d_lr=0. * apu.km,  # but is still necessary for unit checking
        time_percent=0. * apu.percent,
        ):
    '''
    Calculate the free space loss, L_bfsg, of a propagating radio wave
    according to ITU-R P.452-16 Eq. (8-12).

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    dist - Distance between transmitter and receiver [km]
    freq - Frequency of radiation [GHz]
    omega - Fraction of the path over water [%] (see Table 3)
    temperature - Ambient temperature in relevant layer [K]
    pressure - Total air pressure (dry + wet) in relevant layer [hPa]
    atm_method - Which annex to use for atm model P.676, ['annex1'/'annex2']
    do_corrections - Whether to apply focusing/multipath corrections
        Default: False; if set to True, you need to provide the following:
    d_lt, d_lr - Distance to horizon (for transmitter/receiver) [km]
        (For a LoS path, use distance to Bullington point, inferred from
        diffraction method for 50% time.)
    time_percent - Time percentage [%]
        (for average month, this is just p, for worst-month, this is β_0)

    Returns
    -------
    L_bfsg - Free-space loss [dB]
        If do_corrections is set to True, this contains either E_sp or E_sβ,
        depending on which quantity was used for time_percent.

    Notes
    -----
    - This is similar to conversions.free_space_loss function but additionally
      accounts for athmospheric absorption and allows to correct
      for focusing and multipath effects.
    '''

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

    if do_corrections:
        L_bfsg += 2.6 * (
            1. - np.exp(-0.1 * (d_lt + d_lr))
            ) * np.log10(time_percent / 50.)

    return L_bfsg


@helpers.ranged_quantity_input(
    dist=(1.e-30, None, apu.km),
    freq=(1.e-30, None, apu.GHz),
    temperature=(0, None, apu.K),
    pressure=(0, None, apu.hPa),
    N_0=(1.e-30, None, cnv.dimless),
    a_e=(1.e-30, None, apu.km),
    theta_t=(None, None, apu.mrad),  # TODO: proper ranges
    theta_r=(None, None, apu.mrad),  # TODO: proper ranges
    G_t=(0, None, cnv.dBi),
    G_r=(0, None, cnv.dBi),
    time_percent=(0, 100, apu.percent),
    strip_input_units=True, output_unit=cnv.dB
    )
def tropospheric_scatter_loss_bs(
        dist,
        freq,
        temperature,
        pressure,
        N_0,
        a_e,
        theta_t,
        theta_r,
        G_t,
        G_r,
        time_percent,
        atm_method='annex1',
        ):
    '''
    Calculate the tropospheric scatter loss, L_bs, of a propagating radio wave
    according to ITU-R P.452-16 Eq. (45).

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    dist - Distance between transmitter and receiver [km]
    freq - Frequency of radiation [GHz]
    temperature - Ambient temperature in relevant layer [K]
    pressure - Total air pressure (dry + wet) in relevant layer [hPa]
    N_0 - Sea-level surface refractivity at path center [N-units]
    a_e - Median effective Earth radius at path center [km]
    theta_t, theta_r - For a transhorizon path, transmit and receive horizon
        elevation angles; for a LoS path, the elevation angle to the other
        terminal [mrad]
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
    - Path profile parameters (theta_{t,r}, N_0, a_e) can be derived using
      the [TODO] helper functions.
    '''

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
    dist=(1.e-30, None, apu.km),
    freq=(1.e-30, None, apu.GHz),
    omega=(0, 100, apu.percent),
    temperature=(0, None, apu.K),
    pressure=(0, None, apu.hPa),
    N_0=(1.e-30, None, cnv.dimless),
    a_e=(1.e-30, None, apu.km),
    h_ts=(0, None, apu.m),
    h_rs=(0, None, apu.m),
    h_te=(0, None, apu.m),
    h_re=(0, None, apu.m),
    h_m=(0, None, apu.m),
    d_lt=(0, None, apu.km),
    d_lr=(0, None, apu.km),
    d_ct=(0, None, apu.km),
    d_cr=(0, None, apu.km),
    d_lm=(0, None, apu.km),
    theta_t=(None, None, apu.mrad),  # TODO: proper ranges
    theta_r=(None, None, apu.mrad),  # TODO: proper ranges
    G_t=(0, None, cnv.dBi),
    G_r=(0, None, cnv.dBi),
    beta_0=(0, 100, apu.percent),
    time_percent=(0, 100, apu.percent),
    strip_input_units=True, output_unit=cnv.dB
    )
def ducting_loss_ba(
        dist,
        freq,
        omega,
        temperature,
        pressure,
        N_0,
        a_e,
        h_ts,
        h_rs,
        h_te,
        h_re,
        h_m,
        d_lt,
        d_lr,
        d_ct,
        d_cr,
        d_lm,
        theta_t,
        theta_r,
        G_t,
        G_r,
        beta_0,
        time_percent,
        atm_method='annex1',
        ):
    '''
    Calculate the ducting/layer reflection loss, L_ba, of a propagating radio
    wave according to ITU-R P.452-16 Eq. (46-56).

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    dist - Distance between transmitter and receiver [km]
    freq - Frequency of radiation [GHz]
    omega - Fraction of the path over water [%] (see Table 3)
    temperature - Ambient temperature in relevant layer [K]
    pressure - Total air pressure (dry + wet) in relevant layer [hPa]
    N_0 - Sea-level surface refractivity at path center [N-units]
    a_e - Median effective Earth radius at path center [km]
    h_ts, h_rs - Transmitter/receiver antenna center height above mean
        sea level [m]
    h_te, h_re - Effective heights of transmitter/receiver antennas above
        terrain [m]
    h_m - Terrain roughness [m]
    d_lt, d_lr - Distance to horizon (for transmitter/receiver) [km]
        (For a LoS path, use distance to Bullington point, inferred from
        diffraction method for 50% time.)
    d_ct, d_cr - Distance over land from transmit/receive antenna to the coast
        along great circle interference path [km]
        (set to zero for terminal on ship/sea platform; only relevant if less
        than 5 km)
    d_lm - Longest continuous inland section of the great-circle path [km]
    theta_t, theta_r - For a transhorizon path, transmit and receive horizon
        elevation angles; for a LoS path, the elevation angle to the other
        terminal [mrad]
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
    - Path profile parameters (h_xx, d_xx, theta_x, N_0, a_e) can be derived
      using the [path analysis] helper functions.
    '''

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
