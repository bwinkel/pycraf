#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

# from functools import partial, lru_cache
from astropy import units as apu
import numpy as np
from .. import conversions as cnv
from .. import helpers


__all__ = [
    'anual_time_percentage_from_worst_month'
    ]


# @apu.quantity_input(
#     p_w=apu.percent, phi=apu.deg, omega=apu.percent,
#     )
@helpers.ranged_quantity_input(
    p_w=(0, 100, apu.percent),
    phi=(-90, 90, apu.deg),
    omega=(0, 100, apu.percent),
    strip_input_units=True, output_unit=apu.percent,
    )
def anual_time_percentage_from_worst_month(
        p_w, phi, omega
        ):
    '''
    Calculate annual equivalent time percentage, p, from worst-month time
    percentage, p_w, according to ITU-R P.452-16 Eq. (1).

    Parameters
    ----------
    p_w - worst-month time percentage in %
    phi - path center latitude in degrees
    omega - fraction of the path over water in % (see Table 3)

    Returns
    -------
    p - annual equivalent time percentage in %

    Notes
    -----
    Use this function, if you want to do path propagation calculations
    for the worst month case. The resulting time percentage, p, can then
    be plugged into other functions. If you want just annual averages,
    just use your time percentage value as is.
    '''

    # _p_w = p_w.to(apu.percent).value
    # _phi = phi.to(apu.deg).value
    # _omega = omega.to(cnv.dimless).value

    # assert np.all((_p_w >= 0.) & (_p_w <= 100.)), (
    #     'p_w must be in range 0 to 100 %'
    #     )
    # assert np.all((_phi >= -90.) & (_phi <= 90.)), (
    #     'phi must be in range -90 to 90 deg'
    #     )
    # assert np.all((_omega >= 0.) & (_omega <= 1.)), (
    #     'omega must be in range 0 to 100 %'
    #     )

    # _tmp = np.abs(np.cos(2 * np.radians(_phi))) ** 0.7
    # _G_l = np.sqrt(
    #     np.where(np.abs(_phi) <= 45, 1.1 + _tmp, 1.1 - _tmp)
    #     )

    # _a = np.log10(_p_w) + np.log10(_G_l) - 0.186 * _omega - 0.444
    # _b = 0.078 * _omega + 0.816
    # _p = 10 ** (_a / _b)

    # _p = np.max([_p, _p_w / 12.], axis=0)

    # return apu.Quantity(_p, apu.percent)

    omega /= 100  # convert from percent to fraction

    tmp = np.abs(np.cos(2 * np.radians(phi))) ** 0.7
    G_l = np.sqrt(
        np.where(np.abs(phi) <= 45, 1.1 + tmp, 1.1 - tmp)
        )

    a = np.log10(p_w) + np.log10(G_l) - 0.186 * omega - 0.444
    b = 0.078 * omega + 0.816
    p = 10 ** (a / b)

    p = np.max([p, p_w / 12.], axis=0)
    return p


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
