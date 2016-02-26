#!/usr/bin/python
# -*- coding: utf-8 -*-
# Licensed under GPL v2 - see LICENSE

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

from functools import partial
import numpy as np
from astropy import units as apu
from astropy.units import Quantity, UnitsError
from ..conversions import *


__all__ = ['cispr11_limits', 'cispr22_limits']


def _cispr_limits(freq, lims=None, dist=None):
    '''
    Return cispr limits (Elim in uV/m per MHz) for a given frequency
    for a distance of 1 m.
    '''

    assert isinstance(freq, Quantity), 'freq must be an astropy Quantity object'
    assert lims is not None
    assert dist is not None

    try:
        freq.to(apu.Hz)
    except UnitsError:
        raise UnitsError('Input frequency must have units Hz')

    freq = np.atleast_1d(freq)
    Elim = np.ones(freq.shape, dtype=np.float64) * dB_uV_m

    for lof, hif, lim in lims:

        mask = (freq >= lof) & (freq < hif)
        Elim[mask] = lim

    Elim.value[...] += 20 * np.log10(dist / apu.meter)
    return np.sqrt(Elim.to(apu.microvolt ** 2 / apu.meter ** 2)) / apu.MHz

cispr11_limits = partial(
    _cispr_limits,
    lims=[
        (0 * apu.MHz, 230 * apu.MHz, 30 * dB_uV_m),
        (230 * apu.MHz, 1e20 * apu.MHz, 37 * dB_uV_m)
        ],
    dist=30 * apu.meter
    )
cispr22_limits = partial(
    _cispr_limits,
    lims=[
        (0 * apu.MHz, 230 * apu.MHz, 40 * dB_uV_m),
        (230 * apu.MHz, 1000 * apu.MHz, 47 * dB_uV_m),
        (1000 * apu.MHz, 3000 * apu.MHz, 56 * dB_uV_m),
        (3000 * apu.MHz, 1e20 * apu.MHz, 60 * dB_uV_m)
        ],
    dist=3 * apu.meter
    )

