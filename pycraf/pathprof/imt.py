#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

# from functools import partial, lru_cache
from astropy import units as apu
import numpy as np

from . import cyimt
from .. import conversions as cnv
from .. import utils
# import ipdb


__all__ = [
    'imt_rural_macro_losses', 'imt_urban_macro_losses',
    'imt_urban_micro_losses', 'clutter_imt',
    ]

# Note, we have to curry the quantities here, because Cython produces
# "built-in" functions that don't provide a signature (such that
# ranged_quantity_input fails)


def _Qinv(x):
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


def _clutter_imt(
        freq,
        dist,
        location_percent,
        num_end_points=1,
        ):

    assert num_end_points in [1, 2]

    L_l = 23.5 + 9.6 * np.log10(freq)
    L_s = 32.98 + 23.9 * np.log10(dist) + 3.0 * np.log10(freq)

    L_clutter = -5 * np.log10(
        np.power(10, -0.2 * L_l) + np.power(10, -0.2 * L_s)
        ) - 6 * _Qinv(location_percent / 100.)

    if num_end_points == 2:
        L_clutter *= 2

    return L_clutter


@utils.ranged_quantity_input(
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

    Parameters
    ----------
    freq : `~astropy.units.Quantity`
        Frequency of radiation [GHz]
    dist : `~astropy.units.Quantity`
        Distance between Tx/Rx antennas [km]

        Minimal distance must be 0.25 km (single endpoint clutter) or 1 km
        (if both endpoints are to be corrected for clutter)
    location_percent : `~astropy.units.Quantity`
        Percentage of locations for which the clutter loss `L_clutter`
        (calculated with this function) will not be exceeded [%]
    num_end_points : int, optional
        number of endpoints affected by clutter, allowed values: 1, 2

    Returns
    -------
    L_clutter : `~astropy.units.Quantity`
        Clutter loss [dB]

    Notes
    -----
    - The algorithm is independent of effective antenna height (w.r.t.
      clutter height), i.e., it doesn't distinguish between terminals which
      are close to the ground and those closer to the top of the building.
      However, the model is only appropriate if the terminal is "in the
      clutter", below the rooftops.
    - The result of this function is to be understood as a cumulative
      value. For example, if `location_percent = 2%`, it means that for
      2% of all possible locations, the clutter loss will not exceed the
      returned `L_clutter` value, for the remaining 98% of locations it
      will therefore be lower than `L_clutter`. The smaller `location_percent`,
      the smaller the returned `L_clutter`, i.e., low clutter attenuations
      are more unlikely.
    - This model was proposed by ITU study group SG 3 to replace
      `~pycraf.pathprof.clutter_correction` for IMT 5G studies (especially,
      at higher frequencies, where multipath effects play a role in
      urban and suburban areas).
    '''

    return _clutter_imt(
        freq,
        dist,
        location_percent,
        num_end_points=num_end_points,
        )


@utils.ranged_quantity_input(
    freq=(0.5, 30, apu.GHz),
    dist_2d=(0, 100000, apu.m),
    h_bs=(10, 150, apu.m),
    h_ue=(1, 10, apu.m),
    W=(5, 50, apu.m),
    h=(5, 50, apu.m),
    strip_input_units=True, allow_none=False, output_unit=(cnv.dB, cnv.dB)
    )
def imt_rural_macro_losses(
        freq,
        dist_2d,
        h_bs=35 * apu.m, h_ue=1.5 * apu.m,
        W=20 * apu.m, h=5 * apu.m,
        ):
    '''
    Calculate los/non-los propagation losses Rural-Macro IMT scenario.

    The computation is in accordance with `3GPP TR 38.901 Table 7.4.1-1
    <https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173>`_

    Parameters
    ----------
    freq : `~astropy.units.Quantity`
        Frequency of radiation [GHz]
    dist_2d : `~astropy.units.Quantity`
        Distance on the ground between BS and UE device [m]
        Note: Well-defined only for distances between 10 m and 10 km.
    h_bs : `~astropy.units.Quantity`, optional
        Basestation height [m] (Default: 35 m)
    h_ue : `~astropy.units.Quantity`, optional
        User equipment height [m] (Default: 1.5 m)
    W : `~astropy.units.Quantity`, optional
        Average street width [m] (Default: 20 m)
    h : `~astropy.units.Quantity`, optional
        Average building height [m] (Default: 5 m)

    Returns
    -------
    PL_los : `~astropy.units.Quantity`
        Path loss for line-of-sight cases [dB]
    PL_nlos : `~astropy.units.Quantity`
        Path loss for non-line-of-sight cases [dB]

    Notes
    -----
    - In statistical simulations, the LoS and Non-LoS cases occur with
      certain probabilities. For sampling of path losses the functions
      TODO should be used, which accounts for the likelihoods according
      to `3GPP TR 38.901 Table 7.4.2-1 <https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173>`_

    '''

    return cyimt.rural_macro_losses_cython(
        freq, dist_2d, h_bs, h_ue, W, h,
        )


@utils.ranged_quantity_input(
    freq=(0.5, 30, apu.GHz),
    dist_2d=(0, 100000, apu.m),
    h_bs=(10, 150, apu.m),  # according to 3GPP TR 38.901 only 25 m allowed!?
    h_ue=(1, 22.5, apu.m),
    h_e=(0, 20, apu.m),
    strip_input_units=True, allow_none=False, output_unit=(cnv.dB, cnv.dB)
    )
def imt_urban_macro_losses(
        freq,
        dist_2d,
        h_bs=25 * apu.m, h_ue=1.5 * apu.m,
        h_e=1 * apu.m,
        ):
    '''
    Calculate los/non-los propagation losses Rural-Macro IMT scenario.

    The computation is in accordance with `3GPP TR 38.901 Table 7.4.1-1
    <https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173>`_

    Parameters
    ----------
    freq : `~astropy.units.Quantity`
        Frequency of radiation [GHz]
    dist_2d : `~astropy.units.Quantity`
        Distance on the ground between BS and UE device [m]
        Note: Well-defined only for distances between 10 m and 5 km.
    h_bs : `~astropy.units.Quantity`, optional
        Basestation height [m] (Default: 35 m)
    h_ue : `~astropy.units.Quantity`, optional
        User equipment height [m] (Default: 1.5 m)
    h_e : `~astropy.units.Quantity`, optional
        Effective environment height [m] (Default: 1 m)
        Important: for `h_ue > 13 m`, this is subject to random sampling;
        see `3GPP TR 38.901 Table 7.4.1-1 Note 1
        <https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173>`_

    Returns
    -------
    PL_los : `~astropy.units.Quantity`
        Path loss for line-of-sight cases [dB]
    PL_nlos : `~astropy.units.Quantity`
        Path loss for non-line-of-sight cases [dB]

    Notes
    -----
    - In statistical simulations, the LoS and Non-LoS cases occur with
      certain probabilities. For sampling of path losses the functions
      TODO should be used, which accounts for the likelihoods according
      to `3GPP TR 38.901 Table 7.4.2-1 <https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173>`_

    '''

    return cyimt.urban_macro_losses_cython(
        freq, dist_2d, h_bs, h_ue, h_ue,
        )


@utils.ranged_quantity_input(
    freq=(0.5, 100, apu.GHz),
    dist_2d=(0, 100000, apu.m),
    h_bs=(10, 150, apu.m),  # according to 3GPP TR 38.901 only 10 m allowed!?
    h_ue=(1, 22.5, apu.m),
    strip_input_units=True, allow_none=False, output_unit=(cnv.dB, cnv.dB)
    )
def imt_urban_micro_losses(
        freq,
        dist_2d,
        h_bs=10 * apu.m, h_ue=1.5 * apu.m,
        ):
    '''
    Calculate los/non-los propagation losses Rural-Macro IMT scenario.

    The computation is in accordance with `3GPP TR 38.901 Table 7.4.1-1
    <https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173>`_

    Parameters
    ----------
    freq : `~astropy.units.Quantity`
        Frequency of radiation [GHz]
    dist_2d : `~astropy.units.Quantity`
        Distance on the ground between BS and UE device [m]
        Note: Well-defined only for distances between 10 m and 5 km.
    h_bs : `~astropy.units.Quantity`, optional
        Basestation height [m] (Default: 35 m)
    h_ue : `~astropy.units.Quantity`, optional
        User equipment height [m] (Default: 1.5 m)

    Returns
    -------
    PL_los : `~astropy.units.Quantity`
        Path loss for line-of-sight cases [dB]
    PL_nlos : `~astropy.units.Quantity`
        Path loss for non-line-of-sight cases [dB]

    Notes
    -----
    - In statistical simulations, the LoS and Non-LoS cases occur with
      certain probabilities. For sampling of path losses the functions
      TODO should be used, which accounts for the likelihoods according
      to `3GPP TR 38.901 Table 7.4.2-1 <https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173>`_

    '''

    return cyimt.urban_micro_losses_cython(
        freq, dist_2d, h_bs, h_ue,
        )


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
