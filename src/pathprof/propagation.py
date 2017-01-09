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
    dist=(1.e-30, None, apu.m),
    freq=(1.e-30, None, apu.Hz),
    omega=(0, 100, apu.percent),
    temperature=(0, None, apu.K),
    pressure=(0, None, apu.hPa),
    d_lt=(0, None, apu.m),
    d_lr=(0, None, apu.m),
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
        d_lt=0. * apu.m,  # Note: this gets stripped by decorator
        d_lr=0. * apu.m,  # but is still necessary for unit checking
        time_percent=0. * apu.percent,
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

    # bin omega to improve specific_attenuation caching
    omega_b = np.int32(omega + 0.5)

    rho_water = 7.5 + 2.5 * omega_b / 100.
    pressure_water = rho_water * temperature / 216.7
    pressure_dry = pressure - pressure_water

    if atm_method == 'annex1':
        atten_dry_dB, atten_wet_dB = _vectorized_specific_attenuation_annex1(
            freq / 1.e9, pressure_dry, pressure_water, temperature
            )
    else:
        atten_dry_dB, atten_wet_dB = _vectorized_specific_attenuation_annex2(
            freq / 1.e9, pressure, rho_water, temperature
            )

    A_g = (atten_dry_dB + atten_wet_dB) * dist / 1000.

    # better use Eq. (8) for full consistency
    # L_bfsg = cnv.conversions._free_space_loss(freq, dist)  # negative dB
    # L_bfsg = - 10. * np.log10(L_bfsg)  # positive dB
    L_bfsg = 92.5 + 20 * np.log10(freq * 1.e-9) + 20 * np.log10(dist * 1.e-3)
    L_bfsg += A_g

    if do_corrections:
        L_bfsg += 2.6 * (
            1. - np.exp(-0.1 * (d_lt + d_lr))
            ) * np.log10(time_percent / 50.)

    return L_bfsg


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
