#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

# from functools import partial, lru_cache
import os
from astropy import units as apu
import numpy as np
from .. import conversions as cnv
from .. import atm
from .. import helpers


__all__ = [
    'free_space_loss_bfsg',
    ]


@helpers.ranged_quantity_input(
    dist=(1.e-30, None, apu.m),
    freq=(1.e-30, None, apu.Hz),
    omega=(0, 100, apu.percent),
    temperature=(0, 100, apu.K),
    pressure=(0, 100, apu.hPa),
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
        d_lt=0.,
        d_lr=0.,
        time_percent=0.,
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

    Returns
    -------
    Free-space loss, FSPL [dB]


    Notes
    -----
    - This is similar to conversions.free_space_loss function but also
      accounts for athmospheric absorption and allows to correct
      for focusing and multipath effects.
    '''

    assert atm_method in ['annex1', 'annex2'], (
        'atm_method must be on of "annex1" or "annex2"'
        )

    rho_water = 7.5 + 2.5 * omega
    pressure_water = rho_water * temperature / 216.7
    pressure_dry = pressure - pressure_water

    if atm_method == 'annex1':
        atten_dry, atten_wet = atm.specific_attenuation_annex1(
            freq, pressure_dry, pressure_water, temperature
            )
    else:
        atten_dry, atten_wet = atm.specific_attenuation_annex2(
            freq, pressure, rho_water, temperature
            )

    L_bfsg = cnv._free_space_loss(freq, dist)
    L_bfsg /= (atten_dry + atten_wet) * dist / 1000.
    L_bfsg = 10. * np.log10(L_bfsg)

    if do_corrections:
        L_bfsg += 2.6 * (
            1. - np.exp(-0.1 * (d_lt + d_lr))
            ) * np.log10(time_percent / 50.)

    return 10. * np.log10(L_bfsg)


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
