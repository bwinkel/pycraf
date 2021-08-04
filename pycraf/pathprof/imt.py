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
    'imt_urban_micro_losses', 'P_pusch', 'UE_MIN_MAX_DISTANCES',
    'clutter_imt',
    ]


UE_MIN_MAX_DISTANCES = {
    'rural_macro': (10, 10000),
    'urban_macro': (10, 5000),
    'urban_micro': (10, 5000),
    }

# UE_MIN_MAX_DISTANCES.__doc__ = '''
#     Minimum and maximum distances that UEs have from base stations in each of
#     the IMT propagation models. Below and beyond these, the path losses will
#     be NaN.
#     '''

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
    strip_input_units=True, allow_none=False,
    output_unit=(cnv.dB, cnv.dB, cnv.dimless)
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
    los_prob : `~astropy.units.Quantity`
        Probability for a path to be line-of-sight. [dimless]
        (see Notes and Examples)

    Notes
    -----
    - In statistical simulations, the LoS and Non-LoS cases occur with
      certain probabilities. For sampling of path losses the return
      parameter `los_prob` can be used, which accounts for the likelihoods
      according to `3GPP TR 38.901 Table 7.4.2-1
      <https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173>`_


    Examples
    --------

    A typical usage, which also accounts for the line-of-sight probabilities,
    would be::

        >>> import numpy as np
        >>> from pycraf import conversions as cnv
        >>> from pycraf import pathprof
        >>> from astropy import units as u
        >>> from astropy.utils.misc import NumpyRNGContext

        >>> freq = 1 * u.GHz
        >>> h_bs, h_ue = 35 * u.m, 1.5 * u.m
        >>> distances = [5, 20, 1000, 20000] * u.m  # 2D distances
        >>> # Note: too small or large distances will lead to NaN values

        >>> PL_los, PL_nlos, los_prob = pathprof.imt_rural_macro_losses(
        ...     freq, distances, h_bs=h_bs, h_ue=h_ue
        ...     )
        >>> PL_los  # doctest: +FLOAT_CMP
        <Decibel [        nan, 64.38071083, 94.57828524,         nan] dB>
        >>> PL_nlos  # doctest: +FLOAT_CMP
        <Decibel [         nan,  65.10847815, 119.54294519,          nan] dB>
        >>> los_prob  # doctest: +FLOAT_CMP
        <Quantity [1.00000000e+00, 9.90049834e-01, 3.71576691e-01, 2.08186856e-09]>

        >>> # randomly assign LOS or Non-LOS type to UE (according to above prob.)
        >>> with NumpyRNGContext(0):
        ...     los_type = np.random.uniform(0, 1, distances.size) < los_prob
        >>> # note: los_type == True : LOS
        >>> #       los_type == False: Non-LOS
        >>> los_type
        array([ True,  True, False, False])
        >>> PL = np.where(
        ...     los_type, PL_los.to_value(cnv.dB), PL_nlos.to_value(cnv.dB)
        ...     ) * cnv.dB
        >>> PL  # doctest: +FLOAT_CMP
        <Decibel [         nan,  64.38071083, 119.54294519,          nan] dB>

    '''

    pl_los, pl_nlos = cyimt.rural_macro_losses_cython(
        freq, dist_2d, h_bs, h_ue, W, h,
        )

    # probability to have a LOS path;
    # see 3GPP TR 38.901, Table 7.4.2
    los_prob = np.exp(-(dist_2d - 10) / 1000)
    los_prob[dist_2d < 10] = 1.
    los_prob = np.broadcast_to(los_prob, pl_los.shape)

    return pl_los, pl_nlos, los_prob


@utils.ranged_quantity_input(
    freq=(0.5, 30, apu.GHz),
    dist_2d=(0, 100000, apu.m),
    h_bs=(10, 150, apu.m),  # according to 3GPP TR 38.901 only 25 m allowed!?
    h_ue=(1, 13, apu.m),  # for larger h_ue, need to implement "C" function
    h_e=(0, 20, apu.m),
    strip_input_units=True, allow_none=False,
    output_unit=(cnv.dB, cnv.dB, cnv.dimless)
    )
def imt_urban_macro_losses(
        freq,
        dist_2d,
        h_bs=25 * apu.m, h_ue=1.5 * apu.m,
        # h_e=1 * apu.m,
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
        Note: in the `pycraf` implementation this is restricted to
        `h_ue < 13 m`. 3GPP TR 38.901 also has a model for heights up
        to 22.5 m, but this involves a `h_e` different from 1 m and is
        probabilistic, which would make the interface much more complicated.

    Returns
    -------
    PL_los : `~astropy.units.Quantity`
        Path loss for line-of-sight cases [dB]
    PL_nlos : `~astropy.units.Quantity`
        Path loss for non-line-of-sight cases [dB]
    los_prob : `~astropy.units.Quantity`
        Probability for a path to be line-of-sight. [dimless]
        (see Notes and Examples)

    Notes
    -----
    - In statistical simulations, the LoS and Non-LoS cases occur with
      certain probabilities. For sampling of path losses the return
      parameter `los_prob` can be used, which accounts for the likelihoods
      according to `3GPP TR 38.901 Table 7.4.2-1
      <https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173>`_


    Examples
    --------

    A typical usage, which also accounts for the line-of-sight probabilities,
    would be::

        >>> import numpy as np
        >>> from pycraf import conversions as cnv
        >>> from pycraf import pathprof
        >>> from astropy import units as u
        >>> from astropy.utils.misc import NumpyRNGContext

        >>> freq = 1 * u.GHz
        >>> h_bs, h_ue = 25 * u.m, 1.5 * u.m
        >>> distances = [5, 20, 1000, 20000] * u.m  # 2D distances
        >>> # Note: too small or large distances will lead to NaN values

        >>> PL_los, PL_nlos, los_prob = pathprof.imt_urban_macro_losses(
        ...     freq, distances, h_bs=h_bs, h_ue=h_ue
        ...     )
        >>> PL_los  # doctest: +FLOAT_CMP
        <Decibel [         nan,  60.76626079, 108.24721393,          nan] dB>
        >>> PL_nlos  # doctest: +FLOAT_CMP
        <Decibel [         nan,  71.74479418, 130.78468516,          nan] dB>
        >>> los_prob  # doctest: +FLOAT_CMP
        <Quantity [1.00000000e+00, 9.72799557e-01, 1.80001255e-02, 9.00000000e-04]>

        >>> # randomly assign LOS or Non-LOS type to UE (according to above prob.)
        >>> with NumpyRNGContext(0):
        ...     los_type = np.random.uniform(0, 1, distances.size) < los_prob
        >>> # note: los_type == True : LOS
        >>> #       los_type == False: Non-LOS
        >>> los_type
        array([ True,  True, False, False])
        >>> PL = np.where(
        ...     los_type, PL_los.to_value(cnv.dB), PL_nlos.to_value(cnv.dB)
        ...     ) * cnv.dB
        >>> PL  # doctest: +FLOAT_CMP
        <Decibel [         nan,  60.76626079, 130.78468516,          nan] dB>

    '''
    # for h_ue > 13 m, need to implement "C" function; then "h_e" is needed!
    # h_e : `~astropy.units.Quantity`, optional
    #     Effective environment height [m] (Default: 1 m)
    #     Important: for `h_ue > 13 m`, this is subject to random sampling;
    #     see `3GPP TR 38.901 Table 7.4.1-1 Note 1
    #     <https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173>`_

    h_e = 1.
    pl_los, pl_nlos = cyimt.urban_macro_losses_cython(
        freq, dist_2d, h_bs, h_ue, h_e,
        )

    # probability to have a LOS path;
    # see 3GPP TR 38.901, Table 7.4.2
    _x = 18 / dist_2d
    los_prob = (
        # for h_e > 13 m, a correction factor would be necessary!!!
        _x + np.exp(-18 / 63 / _x) * (1 - _x)
        )
    los_prob[dist_2d < 18] = 1.
    los_prob = np.broadcast_to(los_prob, pl_los.shape)

    return pl_los, pl_nlos, los_prob


@utils.ranged_quantity_input(
    freq=(0.5, 100, apu.GHz),
    dist_2d=(0, 100000, apu.m),
    h_bs=(10, 150, apu.m),  # according to 3GPP TR 38.901 only 10 m allowed!?
    h_ue=(1, 22.5, apu.m),
    strip_input_units=True, allow_none=False,
    output_unit=(cnv.dB, cnv.dB, cnv.dimless)
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
    los_prob : `~astropy.units.Quantity`
        Probability for a path to be line-of-sight. [dimless]
        (see Notes and Examples)

    Notes
    -----
    - In statistical simulations, the LoS and Non-LoS cases occur with
      certain probabilities. For sampling of path losses the return
      parameter `los_prob` can be used, which accounts for the likelihoods
      according to `3GPP TR 38.901 Table 7.4.2-1
      <https://portal.3gpp.org/desktopmodules/Specifications/SpecificationDetails.aspx?specificationId=3173>`_


    Examples
    --------

    A typical usage, which also accounts for the line-of-sight probabilities,
    would be::

        >>> import numpy as np
        >>> from pycraf import conversions as cnv
        >>> from pycraf import pathprof
        >>> from astropy import units as u
        >>> from astropy.utils.misc import NumpyRNGContext

        >>> freq = 1 * u.GHz
        >>> h_bs, h_ue = 10 * u.m, 1.5 * u.m
        >>> distances = [5, 20, 1000, 20000] * u.m  # 2D distances
        >>> # Note: too small or large distances will lead to NaN values

        >>> PL_los, PL_nlos, los_prob = pathprof.imt_urban_micro_losses(
        ...     freq, distances, h_bs=h_bs, h_ue=h_ue
        ...     )
        >>> PL_los  # doctest: +FLOAT_CMP
        <Decibel [         nan,  60.47880565, 118.53377126,          nan] dB>
        >>> PL_nlos  # doctest: +FLOAT_CMP
        <Decibel [         nan,  69.59913521, 128.3005538 ,          nan] dB>
        >>> los_prob  # doctest: +FLOAT_CMP
        <Quantity [1.00000000e+00, 9.57375342e-01, 1.80000000e-02, 9.00000000e-04]>

        >>> # randomly assign LOS or Non-LOS type to UE (according to above prob.)
        >>> with NumpyRNGContext(0):
        ...     los_type = np.random.uniform(0, 1, distances.size) < los_prob
        >>> # note: los_type == True : LOS
        >>> #       los_type == False: Non-LOS
        >>> los_type
        array([ True,  True, False, False])
        >>> PL = np.where(
        ...     los_type, PL_los.to_value(cnv.dB), PL_nlos.to_value(cnv.dB)
        ...     ) * cnv.dB
        >>> PL  # doctest: +FLOAT_CMP
        <Decibel [         nan,  60.47880565, 128.3005538 ,          nan] dB>

    '''

    pl_los, pl_nlos = cyimt.urban_micro_losses_cython(
        freq, dist_2d, h_bs, h_ue,
        )

    # probability to have a LOS path;
    # see 3GPP TR 38.901, Table 7.4.2
    _x = 18 / dist_2d
    los_prob = (
        _x + np.exp(-0.5 / _x) * (1 - _x)
        )
    los_prob[dist_2d < 18] = 1.
    los_prob = np.broadcast_to(los_prob, pl_los.shape)

    return pl_los, pl_nlos, los_prob


@utils.ranged_quantity_input(
    P_cmax=(-20, 50, cnv.dBm),  # range pretty large, but what is realistic?
    P_0_pusch=(-200, 100, cnv.dBm),  # dito
    alpha=(0, 1, cnv.dimless),
    PL=(-100, 300, cnv.dB),
    strip_input_units=True, allow_none=False,
    output_unit=(cnv.dBm)
    )
def P_pusch(P_cmax, M_pusch, P_0_pusch, alpha, PL):
    '''
    Calculate power output level after UE power control.

    See `ITU-R Rec. M.2101-0 <https://www.itu.int/rec/R-REC-M.2101/en>`_
    Section 4.1.

    Parameters
    ----------
    P_cmax : `~astropy.units.Quantity`
        Maximum transmit power [dBm]
    M_pusch : `numpy.ndarray`, int
        Number of allocated resource blocks (RBs)

        Is this the bandwidth per carrier divided by the RB bandwidth (
        typically 180 kHz) and number of UE devices associated to each
        carrier?
    P_0_pusch : `numpy.ndarray`
        Initial receive target UE power level per RB [dBm]
    alpha : `~astropy.units.Quantity`
        Balancing factor for UEs with bad channel and UEs with good channel
    PL : `~astropy.units.Quantity`
        Path loss between UE and its associated BS [dB]
        One should use one of the functions
        `~pathprof.imt_rural_macro_losses`,
        `~pathprof.imt_urban_macro_losses`, or
        `~pathprof.imt_urban_micro_losses` to determine PL for the required
        scenario.

        Note: Antenna gains should be included, so this is rather the
        coupling loss than the path loss, unlike what is stated in
        `ITU-R Rec. M.2101-0 <https://www.itu.int/rec/R-REC-M.2101/en>`_.
        ``

    Returns
    -------
    P_pusch : `~astropy.units.Quantity`
        Transmit power of the terminal [dBm]



    Examples
    --------

    A typical usage would be::

        >>> import numpy as np
        >>> from pycraf import conversions as cnv
        >>> from pycraf import pathprof
        >>> from astropy import units as u
        >>> from astropy.utils.misc import NumpyRNGContext

        >>> freq = 6.65 * u.GHz
        >>> h_bs, h_ue = 10 * u.m, 1.5 * u.m
        >>> distances = [20, 100, 500, 1000] * u.m  # 2D distances

        >>> PL_los, PL_nlos, los_prob = pathprof.imt_urban_micro_losses(
        ...     freq, distances, h_bs=h_bs, h_ue=h_ue
        ...     )

        >>> # randomly assign LOS or Non-LOS type to UE (according to above prob.)
        >>> with NumpyRNGContext(0):
        ...     los_type = np.random.uniform(0, 1, distances.size) < los_prob
        >>> # note: los_type == True : LOS
        >>> #       los_type == False: Non-LOS
        >>> PL = np.where(
        ...     los_type, PL_los.to_value(cnv.dB), PL_nlos.to_value(cnv.dB)
        ...     ) * cnv.dB
        >>> PL  # doctest: +FLOAT_CMP
        <Decibel [ 76.93523856, 110.58128371, 135.20195715, 145.82665484] dB>

        >>> # Assume some antenna gains:
        >>> G_bs, G_ue = 20 * cnv.dBi, 5 * cnv.dBi
        >>> CL = (
        ...     PL.to_value(cnv.dB) -
        ...     G_bs.to_value(cnv.dBi) -
        ...     G_ue.to_value(cnv.dBi)
        ...     ) * cnv.dB


        >>> bw_carrier = 100 * u.MHz
        >>> bw_rb = 180 * u.kHz  # resource block bandwidth
        >>> num_ue = 3  # 3 UEs per BS sector and carrier
        >>> M_pusch = int(bw_carrier / bw_rb / num_ue)
        >>> M_pusch
        185

        >>> P_cmax = 23 * cnv.dBm
        >>> P_0_pusch = -95.5 * cnv.dBm
        >>> alpha = 0.8 * cnv.dimless

        >>> P_pusch = pathprof.P_pusch(
        ...     P_cmax, M_pusch, P_0_pusch, alpha, CL
        ...     )
        >>> P_pusch.to(cnv.dBm)
        <Decibel [-31.28009187,  -4.36325575,  15.333283  ,  23.          ] dB(mW)>

    '''

    P_pusch = 10 * np.log10(M_pusch) + P_0_pusch + alpha * PL
    P_pusch[P_pusch > P_cmax] = P_cmax

    return P_pusch


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
