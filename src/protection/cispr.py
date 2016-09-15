#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

from functools import partial
import numpy as np
from astropy import units as apu
from astropy.units import Quantity, UnitsError
from ..conversions import *


__all__ = ['cispr11_limits', 'cispr22_limits']


def _cispr_limits(
        freq, detector_type='RMS', detector_dist=30. * apu.m, params=None
        ):
    '''
    Return cispr limits (Elim in uV/m per MHz) for a given frequency
    for a distance of detector_dist.

    Parameters
    ----------
    freq - Frequencies (astropy.Quanity)
    detector - either 'QP' (quasi-peak), or 'RMS' (default)

    Returns
    -------
    Efield limit, measure-bandwidth (of detector)

    According to Hasenpusch (BNetzA, priv. comm.) the CISPR standard
    detector is a quasi-peak detector with a bandwidth of 120 kHz. To convert
    to an RMS detector (which is more suitable to the RA.769) one has to
    subtract 5.5 dB.
    '''

    assert isinstance(freq, Quantity), 'freq must be astropy Quantity object'
    assert params is not None

    assert detector_type in ['QP', 'RMS']

    try:
        freq.to(apu.Hz)
    except UnitsError:
        raise UnitsError('Input frequency must have units Hz')

    freq = np.atleast_1d(freq)
    Elim = np.ones(freq.shape, dtype=np.float64) * dB_uV_m

    for lof, hif, lim in params['lims']:

        mask = (freq >= lof) & (freq < hif)
        Elim[mask] = lim.to(dB_uV_m)

    Elim.value[...] += 20 * np.log10(
        (params['dist'] / detector_dist).to(dimless)
        )
    if detector_type == 'RMS':
        Elim.value[...] -= 5.5

    return (
        np.sqrt(Elim.to(apu.microvolt ** 2 / apu.meter ** 2)),
        params['meas_bw']
        )

CISPR11_PARAMS = {
    'lims': [
        (0 * apu.MHz, 230 * apu.MHz, 30 * dB_uV_m),
        (230 * apu.MHz, 1e20 * apu.MHz, 37 * dB_uV_m)
        ],
    'dist': 30 * apu.meter,
    'meas_bw': 120 * apu.kHz
    }
cispr11_limits = partial(_cispr_limits, params=CISPR11_PARAMS)

CISPR22_PARAMS = {
    'lims': [
        (0 * apu.MHz, 230 * apu.MHz, 40 * dB_uV_m),
        (230 * apu.MHz, 1000 * apu.MHz, 47 * dB_uV_m),
        (1000 * apu.MHz, 3000 * apu.MHz, 56 * dB_uV_m),
        (3000 * apu.MHz, 1e20 * apu.MHz, 60 * dB_uV_m)
        ],
    'dist': 30 * apu.meter,
    'meas_bw': 120 * apu.kHz  # is this correct?
    }
cispr22_limits = partial(_cispr_limits, params=CISPR22_PARAMS)
