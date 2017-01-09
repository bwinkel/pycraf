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
from .. import conversions as cnv
from .. import atm
from .. import helpers
import ipdb


__all__ = [
    'free_space_loss_bfsg',
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
    omega - fraction of the path over water [%] (see Table 3)
    temperature - ambient temperature in relevant layer [K]
    pressure - total air pressure (dry + wet) in relevant layer [hPa]
    atm_method - which annex to use for atm model P.676, ['annex1'/'annex2']
    do_corrections - whether to apply focusing/multipath corrections
        Default: False; if set to True, you need to provide the following:
    d_lt, d_lr - distance to horizon (for transmitter/receiver) [km]
        (For a LoS path, use distance to Bullington point, inferred from
        diffraction method for 50% time.)
    time_percent - time percentage [%]
        (for average month, this is just p, for worst-month, this is β_0)

    Returns
    -------
    Free-space loss, FSPL [dB]
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
    Calculate the free space loss, L_bfsg, of a propagating radio wave
    according to ITU-R P.452-16 Eq. (8-12).

    Note: All quantities must be astropy Quantities
          (astropy.units.quantity.Quantity).

    Parameters
    ----------
    dist - Distance between transmitter and receiver [m]
    freq - Frequency of radiation [Hz]
    omega - fraction of the path over water [%] (see Table 3)
    temperature - ambient temperature in relevant layer [K]
    pressure - total air pressure (dry + wet) in relevant layer [hPa]
    atm_method - which annex to use for atm model P.676, ['annex1'/'annex2']
    do_corrections - whether to apply focusing/multipath corrections
        Default: False; if set to True, you need to provide the following:
    d_lt, d_lr - distance to horizon (for transmitter/receiver) [km]
        (For a LoS path, use distance to Bullington point, inferred from
        diffraction method for 50% time.)
    time_percent - time percentage [%]
        (for average month, this is just p, for worst-month, this is β_0)

    Returns
    -------
    Free-space loss, FSPL [dB]
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
    L_c = 0.051 * np.exp(0.055 * (G_t + G_r))

    # theta is in mrad (therefore we don't need to divide dist by 1000)
    theta = 1e3 * dist / a_e + theta_t + theta_r

    L_bs = (
        190. + L_f + 20 * np.log10(dist * 1.e-3) + 0.573 * theta -
        0.15 * N_0 + L_c + A_g - 10.1 * np.log10(time_percent / 50.) ** 0.7
        )

    return L_bs


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
