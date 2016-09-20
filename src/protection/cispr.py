#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

from functools import partial  # , update_wrapper
import numpy as np
from astropy import units as apu
from astropy.units import Quantity, UnitsError
from .. import conversions as cnv


__all__ = ['cispr11_limits', 'cispr22_limits']


@apu.quantity_input(freq=apu.Hz, detector_dist=apu.m)
def _cispr_limits(
        params, freq, detector_type='RMS', detector_dist=30. * apu.m
        ):
    '''
    Return cispr limits (Elim in uV/m per MHz) for a given frequency
    for a distance of detector_dist.

    Parameters
    ----------
    params - Definition of limits, dict (frequency intervals, etc., see below)
    freq - Frequency or frequency vector [Hz]
    detector_type - either 'QP' (quasi-peak), or 'RMS' (default)
    detector_dist - distance between source and detector [m], default: 30 m

    Returns
    -------
    Efield limit, measure-bandwidth (of detector)

    According to Hasenpusch (BNetzA, priv. comm.) the CISPR standard
    detector is a quasi-peak detector with a bandwidth of 120 kHz. To convert
    to an RMS detector (which is more suitable to the RA.769) one has to
    subtract 5.5 dB.

    The params dictionary must have the following content:
    'lims' - list of tuples (low freq, high freq, limit in dB_uV_m)
    'dist' - distance detector <-> source for which the limits are valid
    'meas_bw' - measurement bandwidth
    '''

    assert params is not None
    assert detector_type in ['QP', 'RMS']

    freq = np.atleast_1d(freq)
    Elim = np.ones(freq.shape, dtype=np.float64) * cnv.dB_uV_m

    for lof, hif, lim in params['lims']:

        mask = (freq >= lof) & (freq < hif)
        Elim[mask] = lim.to(cnv.dB_uV_m)

    Elim.value[...] += 20 * np.log10(
        (params['dist'] / detector_dist).to(cnv.dimless)
        )  # 20, because area of sphere is proportional to radius ** 2
    if detector_type == 'RMS':
        Elim.value[...] -= 5.5

    return (
        Elim.to(apu.microvolt / apu.meter),
        params['meas_bw']
        )


CISPR11_PARAMS = {
    'lims': [
        (0 * apu.MHz, 230 * apu.MHz, 30 * cnv.dB_uV_m),
        (230 * apu.MHz, 1e20 * apu.MHz, 37 * cnv.dB_uV_m)
        ],
    'dist': 30 * apu.meter,
    'meas_bw': 120 * apu.kHz
    }
cispr11_limits = partial(_cispr_limits, CISPR11_PARAMS)
# update_wrapper(cispr11_limits, _cispr_limits)
cispr11_limits.__doc__ = '''
    Return cispr-11 limits (Elim in uV/m per MHz) for a given frequency
    for a distance of detector_dist.

    Parameters
    ----------
    freq - Frequency or frequency vector [Hz]
    detector_type - either 'QP' (quasi-peak), or 'RMS' (default)
    detector_dist - distance between source and detector [m], default: 30 m

    Returns
    -------
    Efield limit, measure-bandwidth (of detector)

    According to Hasenpusch (BNetzA, priv. comm.) the CISPR standard
    detector is a quasi-peak detector with a bandwidth of 120 kHz. To convert
    to an RMS detector (which is more suitable to the RA.769) one has to
    subtract 5.5 dB.
    '''

CISPR22_PARAMS = {
    'lims': [
        (0 * apu.MHz, 230 * apu.MHz, 40 * cnv.dB_uV_m),
        (230 * apu.MHz, 1000 * apu.MHz, 47 * cnv.dB_uV_m),
        (1000 * apu.MHz, 3000 * apu.MHz, 56 * cnv.dB_uV_m),
        (3000 * apu.MHz, 1e20 * apu.MHz, 60 * cnv.dB_uV_m)
        ],
    'dist': 30 * apu.meter,
    'meas_bw': 120 * apu.kHz  # is this correct?
    }
cispr22_limits = partial(_cispr_limits, CISPR22_PARAMS)
cispr22_limits.__doc__ = '''
    Return cispr-22 limits (Elim in uV/m per MHz) for a given frequency
    for a distance of detector_dist.

    Parameters
    ----------
    freq - Frequency or frequency vector [Hz]
    detector_type - either 'QP' (quasi-peak), or 'RMS' (default)
    detector_dist - distance between source and detector [m], default: 30 m

    Returns
    -------
    Efield limit, measure-bandwidth (of detector)

    According to Hasenpusch (BNetzA, priv. comm.) the CISPR standard
    detector is a quasi-peak detector with a bandwidth of 120 kHz. To convert
    to an RMS detector (which is more suitable to the RA.769) one has to
    subtract 5.5 dB.
    '''
