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
from .. import utils


__all__ = ['cispr11_limits', 'cispr22_limits']


def _cispr_limits(
        params, freq, detector_type='RMS', detector_dist=30.
        ):
    '''
    The params dictionary must have the following content:
    'lims' - list of tuples (low freq, high freq, limit in dB_uV_m)
    'dist' - distance detector <-> source for which the limits are valid
    'meas_bw' - measurement bandwidth
    '''

    assert params is not None
    assert detector_type in ['QP', 'RMS']

    freq = np.atleast_1d(freq)
    Elim = np.ones(freq.shape, dtype=np.float64)

    for lof, hif, lim in params['lims']:

        mask = (freq >= lof) & (freq < hif)
        Elim[mask] = lim

    # 20, because area of sphere is proportional to radius ** 2
    Elim += 20 * np.log10(params['dist'] / detector_dist)

    # According to Hasenpusch (BNetzA, priv. comm.) the CISPR standard
    # detector is a quasi-peak detector with a bandwidth of 120 kHz. To
    # convert to an RMS detector (which is more suitable to the RA.769) one
    # has to subtract 5.5 dB.
    if detector_type == 'RMS':
        Elim -= 5.5

    return Elim, params['meas_bw']


@utils.ranged_quantity_input(
    freq=(0, None, apu.MHz),
    detector_dist=(0.001, None, apu.m),
    strip_input_units=True, output_unit=(cnv.dB_uV_m, apu.kHz)
    )
def cispr11_limits(freq, detector_type='RMS', detector_dist=30. * apu.m):
    '''
    Industrial-device limits for given frequency and distance from device
    according to `CISPR-11 (EN 55011)
    <http://rfemcdevelopment.eu/index.php/en/emc-emi-standards/en-55011-2009>`_.

    Parameters
    ----------
    freq : `~astropy.units.Quantity`
        Frequency or frequency vector [Hz]
    detector_type : str, optional
        Detector type: 'QP' (quasi-peak), 'RMS' (default: 'RMS)
    detector_dist : `~astropy.units.Quantity`, optional
        Distance between source and detector [m], (default: 30 m)

    Returns
    -------
    efield_limit : `~astropy.units.Quantity`
        Electric-field limit permitted by CISPR-11 at given distance.
        [dB(uV^2 / m^2)]
    meas_bw : `~astropy.units.Quantity`
        Measure bandwidth (of detector) [kHz]
    '''

    cispr11_params = {
        'lims': [
            (0, 230, 30),
            (230, 1e20, 37)
            ],
        'dist': 30,
        'meas_bw': 120,
        }

    return _cispr_limits(
        cispr11_params, freq,
        detector_type=detector_type, detector_dist=detector_dist
        )


@utils.ranged_quantity_input(
    freq=(0, None, apu.MHz),
    detector_dist=(0.001, None, apu.m),
    strip_input_units=True, output_unit=(cnv.dB_uV_m, apu.kHz)
    )
def cispr22_limits(freq, detector_type='RMS', detector_dist=30. * apu.m):
    '''
    Industrial-device limits for given frequency and distance from device
    according to `CISPR-22 (EN 55022)
    <http://www.rfemcdevelopment.eu/en/en-55022-2010>`_.

    Parameters
    ----------
    freq : `~astropy.units.Quantity`
        Frequency or frequency vector [MHz]
    detector_type : str, optional
        Detector type: 'QP' (quasi-peak), 'RMS' (default: 'RMS)
    detector_dist : `~astropy.units.Quantity`, optional
        Distance between source and detector [m], (default: 30 m)

    Returns
    -------
    efield_limit : `~astropy.units.Quantity`
        Electric-field limit permitted by CISPR-22 at given distance.
        [dB(uV^2 / m^2)]
    meas_bw : `~astropy.units.Quantity`
        Measure bandwidth (of detector) [kHz]
    '''

    cispr22_params = {
        'lims': [
            (0, 230, 40),
            (230, 1000, 47),
            (1000, 3000, 56),
            (3000, 1e20, 60)
            ],
        'dist': 30,
        'meas_bw': 120,  # is this correct?
        }

    return _cispr_limits(
        cispr22_params, freq,
        detector_type=detector_type, detector_dist=detector_dist
        )
