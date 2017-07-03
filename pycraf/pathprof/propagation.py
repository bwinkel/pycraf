#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

# from functools import partial, lru_cache
from astropy import units as apu
import numpy as np

from . import cyprop
from .. import conversions as cnv
from .. import helpers
# import ipdb


__all__ = [
    'freespace_loss', 'troposcatter_loss', 'ducting_loss',
    'diffraction_loss', 'complete_loss',
    'clutter_correction', 'clutter_imt',
    'height_profile_data', 'atten_map_fast',
    ]

# Note, we have to curry the quantities here, because Cython produces
# "built-in" functions that don't provide a signature (such that
# ranged_quantity_input fails)


@helpers.ranged_quantity_input(
    output_unit=(cnv.dB, cnv.dB, cnv.dB)
    )
def freespace_loss(pathprop):
    '''
    Calculate the free space loss, L_bfsg, of a propagating radio wave
    according to ITU-R P.452-16 Eq. (8-12).

    Parameters
    ----------
    pathprop

    Returns
    -------
    (L_bfsg, E_sp, E_sβ) - tuple
        L_bfsg - Free-space loss [dB]
        E_sp - focussing/multipath correction for p% [dB]
        E_sbeta - focussing/multipath correction for beta0% [dB]

        with these, one can form
        L_b0p = L_bfsg + E_sp [dB]
        L_b0beta = L_bfsg + E_sβ [dB]

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        [TODO].
    - This is similar to conversions.free_space_loss function but additionally
      accounts for athmospheric absorption and corrects for focusing and
      multipath effects.
    '''

    return cyprop.free_space_loss_bfsg_cython(pathprop)


@helpers.ranged_quantity_input(
    G_t=(None, None, cnv.dBi),
    G_r=(None, None, cnv.dBi),
    strip_input_units=True, output_unit=cnv.dB
    )
def troposcatter_loss(
        pathprop, G_t=0. * cnv.dBi, G_r=0. * cnv.dBi,
        ):
    '''
    Calculate the tropospheric scatter loss, L_bs, of a propagating radio wave
    according to ITU-R P.452-16 Eq. (45).

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    pathprop -
    G_t, G_r - Antenna gain (transmitter, receiver) in the direction of the
        horizon(!) along the great-circle interference path [dBi]

    Returns
    -------
    L_bs - Tropospheric scatter loss [dB]

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        [TODO].
    '''

    return cyprop.tropospheric_scatter_loss_bs_cython(pathprop, G_t, G_r)


@helpers.ranged_quantity_input(output_unit=cnv.dB)
def ducting_loss(pathprop):
    '''
    Calculate the ducting/layer reflection loss, L_ba, of a propagating radio
    wave according to ITU-R P.452-16 Eq. (46-56).

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    pathprop -

    Returns
    -------
    L_ba - Ducting/layer reflection loss [dB]

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        [TODO.
    '''

    return cyprop.ducting_loss_ba_cython(pathprop)


@helpers.ranged_quantity_input(
    output_unit=(cnv.dB, cnv.dB, cnv.dB, cnv.dB, cnv.dB)
    )
def diffraction_loss(pathprop):
    '''
    Calculate the Diffraction loss of a propagating radio
    wave according to ITU-R P.452-16 Eq. (14-44).

    Parameters
    ----------
    pathprop -

    Returns
    -------
    (L_d_50, L_dp, L_bd_50, L_bd, L_min_b0p)
        L_d_50 - Median diffraction loss [dB]
        L_dp - Diffraction loss not exceeded for p% time, [dB]
        L_bd_50 - Median basic transmission loss associated with
            diffraction [dB]; L_bd_50 = L_bfsg + L_d50
        L_bd - Basic transmission loss associated with diffraction not
            exceeded for p% time [dB]; L_bd = L_b0p + L_dp
        L_min_b0p - Notional minimum basic transmission loss associated with
            LoS propagation and over-sea sub-path diffraction
        Note: L_d_50 and L_dp are just intermediary values; the complete
            diffraction loss is L_bd_50 or L_bd, respectively (taking into
            account a free-space loss component for the diffraction path)

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        [TODO]
    '''

    return cyprop.diffraction_loss_complete_cython(pathprop)


@helpers.ranged_quantity_input(
    G_t=(None, None, cnv.dBi),
    G_r=(None, None, cnv.dBi),
    strip_input_units=True,
    output_unit=(cnv.dB, cnv.dB, cnv.dB, cnv.dB, cnv.dB, cnv.dB, cnv.dB)
    )
def complete_loss(
        pathprop, G_t=0. * cnv.dBi, G_r=0. * cnv.dBi,
        ):
    '''
    Calculate the total loss of a propagating radio
    wave according to ITU-R P.452-16 Eq. (58-64).

    Parameters
    ----------
    pathprop -
    G_t, G_r - Antenna gain (transmitter, receiver) in the direction of the
        horizon(!) along the great-circle interference path [dBi]

    Returns
    -------
    (L_bfsg, L_bd, L_bs, L_ba, L_b, L_b_corr, L)
        L_bfsg - Free-space loss [dB]
        L_bd - Basic transmission loss associated with diffraction not
            exceeded for p% time [dB]; L_bd = L_b0p + L_dp
        L_bs - Tropospheric scatter loss [dB]
        L_ba - Ducting/layer reflection loss [dB]
        L_b - Complete path propagation loss [dB]
        L_b_corr - As L_b but with clutter correction [dB]
        L - As L_b_corr but with gain correction [dB]

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        [TODO]
    '''

    return cyprop.path_attenuation_complete_cython(pathprop, G_t, G_r)


@helpers.ranged_quantity_input(
    h_g=(None, None, apu.m),
    freq=(None, None, apu.GHz),
    strip_input_units=True,
    output_unit=cnv.dB
    )
def clutter_correction(
        h_g, zone, freq
        ):
    '''
    Calculate the Clutter loss of a propagating radio
    wave according to ITU-R P.452-16 Eq. (57).

    Parameters
    ----------
    h_g - height above ground [m]
    zone - Clutter category (see CLUTTER enum)
    freq - frequency [GHz]

    Returns
    -------
    A_h - Clutter correction to path attenuation [dB]

    Notes
    -----
    - Path profile parameters (PathProps object) can be derived using the
        [TODO]
    '''

    return cyprop.clutter_correction_cython(h_g, zone, freq)


def _clutter_imt(
        freq,
        dist,
        location_percent,
        num_end_points=1,
        ):

    assert num_end_points in [1, 2]

    def Qinv(x):
        # Note, this is *not* identical to cyprop._I_helper
        # only good between 1.e-6 and 0.5
        # See R-Rec P.1546

        x = np.atleast_1d(x).copy()
        mask = x > 0.5
        x[mask] = 1 - x[mask]

        T = np.sqrt(-2 * np.log(x))
        Z = (
            (
                ((0.010328 * T + 0.802853) * T) + 2.515516698
                ) /
            (
                ((0.001308 * T + 0.189269) * T + 1.432788) * T + 1.
                )
            )

        Q = T - Z
        Q[mask] *= -1
        return Q

    # def Qinv(x):
    #     # larger x range than the approximation given in P.1546?
    #     # definitely much slower

    #     from scipy.stats import norm as qnorm

    #     x = np.atleast_1d(x).copy()

    #     mask = x > 0.5
    #     x[mask] = 1 - x[mask]

    #     Q = -qnorm.ppf(x, 0)
    #     Q[mask] *= -1

    #     return Q

    L_l = 23.5 + 9.6 * np.log10(freq)
    L_s = 32.98 + 23.9 * np.log10(dist) + 3.0 * np.log10(freq)

    L_clutter = -5 * np.log10(
        np.power(10, -0.2 * L_l) + np.power(10, -0.2 * L_s)
        ) - 6 * Qinv(location_percent / 100.)

    if num_end_points == 2:
        L_clutter *= 2

    return L_clutter


@helpers.ranged_quantity_input(
    freq=(2, 67, apu.GHz),
    dist=(0.25, None, apu.km),
    location_percent=(0, 100, apu.percent),
    strip_input_units=True, output_unit=cnv.dB
    )
def clutter_imt(
        freq,
        dist,
        location_percent,
        num_end_points=1,
        ):
    '''
    Calculate the Clutter loss according to IMT.CLUTTER document (method 2).

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    freq - Frequency [GHz]
    dist - Distance between Tx/Rx antennas [km]
        minimal distance must be 0.25 km (single endpoint clutter) or 1 km
        (if both endpoints are to be corrected for clutter)
    location_percent - Percentage of locations(?) [%]
    num_end_points - number of endpoints affected by clutter (one or two)

    Returns
    -------
    L_clutter - Clutter loss [dB]

    Notes
    -----
    - The algorithm is independent of effective antenna height (w.r.t.
        clutter height).
    '''

    return _clutter_imt(
        freq,
        dist,
        location_percent,
        num_end_points=num_end_points,
        )


# TODO: do we want to convert output dictionary arrays to quantities?
@helpers.ranged_quantity_input(
    lon_t=(0, 360, apu.deg),
    lat_t=(-90, 90, apu.deg),
    map_size_lon=(0.002, 90, apu.deg),
    map_size_lat=(0.002, 90, apu.deg),
    map_resolution=(0.0001, 0.1, apu.deg),
    d_tm=(None, None, apu.m),
    d_lm=(None, None, apu.m),
    d_ct=(None, None, apu.m),
    d_cr=(None, None, apu.m),
    strip_input_units=True, allow_none=True, output_unit=None
    )
def height_profile_data(
        lon_t, lat_t,
        map_size_lon, map_size_lat,
        map_resolution=3. * apu.arcsec,
        do_cos_delta=True,
        zone_t=cyprop.CLUTTER.UNKNOWN, zone_r=cyprop.CLUTTER.UNKNOWN,
        d_tm=None, d_lm=None,
        d_ct=None, d_cr=None,
        ):

    '''
    Calculate height profiles and auxillary maps needed for atten_map_fast.

    This can be used to cache height-profile data. Since it is independent
    of frequency, time_percent, Tx and Rx heights, etc., one can re-use
    it to save computing time when doing batch jobs.

    Note: Path attenuation is completely symmetric, i.e., it doesn't matter if
    the transmitter or the receiver is situated in the map center.

    Parameters
    ----------
    lon_t, lat_t - Transmitter coordinates [deg]
    map_size_lon, map_size_lat - Map size [deg]
    map_resolution - Pixel resolution of map [deg]
    do_cos_delta - If True, divide map_size_lon by cos(latitude) for square map
    zone_t, zone_r - Transmitter/receiver clutter zone codes.
    d_tm - longest continuous land (inland + coastal) section of the
        great-circle path [km]
    d_lm - longest continuous inland section of the great-circle path [km]
    d_ct, d_cr - Distance over land from transmit/receive antenna to the coast
        along great circle interference path [km]
        (set to zero for terminal on ship/sea platform; only relevant if less
        than 5 km)

    Returns
    -------
    Dictionary with height profiles and auxillary maps
    '''

    return cyprop.height_profile_data_cython(
        lon_t, lat_t,
        map_size_lon, map_size_lat,
        map_resolution=map_resolution,
        do_cos_delta=1 if do_cos_delta else 0,
        zone_t=zone_t, zone_r=zone_r,
        d_tm=d_tm, d_lm=d_lm,
        d_ct=d_ct, d_cr=d_cr,
        )


@helpers.ranged_quantity_input(
    freq=(0.1, 100, apu.GHz),
    temperature=(None, None, apu.K),
    pressure=(None, None, apu.hPa),
    h_tg=(None, None, apu.m),
    h_rg=(None, None, apu.m),
    time_percent=(0, 50, apu.percent),
    omega_percent=(0, 100, apu.percent),
    strip_input_units=True,
    output_unit=(cnv.dB, apu.deg, apu.deg)
    )
def atten_map_fast(
        freq,
        temperature,
        pressure,
        h_tg, h_rg,
        time_percent,
        hprof_data,  # dict_like
        omega=0 * apu.percent,
        polarization=0,
        version=16,
        ):
    '''
    Calculate attenuation map using a fast method.

    Parameters
    ----------
    freq - Frequency of radiation [GHz]
    temperature - Temperature (K)
    pressure - Pressure (hPa)
    h_tg, h_rg - Transmitter/receiver heights over ground [m]
    time_percent - Time percentage [%] (maximal 50%)
    hprof_data - Dictionary with height profiles and auxillary maps as
        calculated with height_profile_data.
    omega - Fraction of the path over water [%] (see Table 3)
    polarization - Polarization (0 - horizontal, 1 - vertical; default: 0)
    version - P.452 version to use (14 or 16)

    Returns
    -------
    (atten_map, eps_pt_map, eps_pr_map) - tuple
        atten_map - 3D array with attenuation maps
            first dimension has length 6 and refers to:
                0: L_bfsg - Free-space loss [dB]
                1: L_bd - Basic transmission loss associated with diffraction
                          not exceeded for p% time [dB]; L_bd = L_b0p + L_dp
                2: L_bs - Tropospheric scatter loss [dB]
                3: L_ba - Ducting/layer reflection loss [dB]
                4: L_b - Complete path propagation loss [dB]
                5: L_b_corr - As L_b but with clutter correction [dB]
            (i.e., the output of path_attenuation_complete_cython without
            gain-corrected values)
        eps_pt_map - 2D array with elevation angle of paths w.r.t. Tx [deg]
        eps_pr_map - 2D array with elevation angle of paths w.r.t. Rx [deg]
    '''

    return cyprop.atten_map_fast_cython(
        freq,
        temperature,
        pressure,
        h_tg, h_rg,
        time_percent,
        hprof_data,  # dict_like
        omega=omega,
        polarization=polarization,
        version=version,
        )


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
