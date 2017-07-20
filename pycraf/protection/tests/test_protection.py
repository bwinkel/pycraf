#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import assert_quantity_allclose
from astropy import units as apu
from astropy.units import Quantity
from ... import conversions as cnv
from ... import protection as prot
from ...utils import check_astro_quantities
# from astropy.utils.misc import NumpyRNGContext


TOL_KWARGS = {'atol': 0., 'rtol': 1.e-6}


# @pytest.mark.skip(reason='failing on AppVeyor and Travis for unknown reason')
def test_cispr_limits():

    args_list = [
        (0, None, apu.Hz),
        ]
    kwargs_list = [
        ('detector_dist', 0.001, None, apu.m),
        ]
    check_astro_quantities(prot.cispr11_limits, args_list, kwargs_list)
    check_astro_quantities(prot.cispr22_limits, args_list, kwargs_list)

    res = prot.cispr11_limits(1 * apu.Hz)
    assert isinstance(res, tuple)
    assert res[0].unit == cnv.dB_uV_m
    res = prot.cispr22_limits(1 * apu.Hz)
    assert isinstance(res, tuple)
    assert res[0].unit == cnv.dB_uV_m

    freqs = Quantity(np.linspace(10, 900, 7), apu.MHz)

    assert_quantity_allclose(
        prot.cispr11_limits(
            freqs,
            detector_type='RMS',
            detector_dist=30. * apu.m
            )[0],
        np.array([
            24.5, 24.5, 31.5, 31.5, 31.5, 31.5, 31.5
            ]) * cnv.dB_uV_m,
        )

    assert_quantity_allclose(
        prot.cispr11_limits(
            freqs,
            detector_type='RMS',
            detector_dist=300. * apu.m
            )[0],
        np.array([
            4.5, 4.5, 11.5, 11.5, 11.5, 11.5, 11.5
            ]) * cnv.dB_uV_m,
        )

    assert_quantity_allclose(
        prot.cispr11_limits(
            freqs,
            detector_type='QP',
            detector_dist=30. * apu.m
            )[0],
        np.array([
            30., 30., 37., 37., 37., 37., 37.
            ]) * cnv.dB_uV_m,
        )

    freqs = Quantity(np.linspace(10, 3100, 7), apu.MHz)

    assert_quantity_allclose(
        prot.cispr22_limits(
            freqs,
            detector_type='RMS',
            detector_dist=30. * apu.m
            )[0],
        np.array([
            34.5, 41.5, 50.5, 50.5, 50.5, 50.5, 54.5
            ]) * cnv.dB_uV_m,
        )

    assert_quantity_allclose(
        prot.cispr22_limits(
            freqs,
            detector_type='RMS',
            detector_dist=300. * apu.m
            )[0],
        np.array([
            14.5, 21.5, 30.5, 30.5, 30.5, 30.5, 34.5
            ]) * cnv.dB_uV_m,
        )

    assert_quantity_allclose(
        prot.cispr22_limits(
            freqs,
            detector_type='QP',
            detector_dist=30. * apu.m
            )[0],
        np.array([
            40., 47., 56., 56., 56., 56., 60.
            ]) * cnv.dB_uV_m,
        )


CONT_LIMS_VALUES = [
    5006.0, 5.1822630178e-21, 7.4925620118e-18, 1594.88178292, 28.200091414,
    ]
CONT_LIMS_DB_VALUES = [
    5006.0, -202.854805488, -171.25369654, -227.972715025, 29.0050103227,
    ]
SPEC_LIMS_VALUES = [
    0.7071067812, 3.03742736e-21, 2.186381732e-15, 4225.3020266, 12.522083029,
    ]
SPEC_LIMS_DB_VALUES = [
    0.707106781187, -205.174940994, -146.6027401, -223.7414224, 21.9535315816,
    ]

COL_NAMES = [
    'frequency',
    'bandwidth',
    'T_A',
    'T_rx',
    'T_rms',
    'P_rms_nu',
    'Plim',
    'Slim',
    'Slim_nu',
    'Efield',
    'Efield_norm',
    ]

COL_UNITS = [
    apu.MHz,
    apu.MHz,
    apu.K,
    apu.K,
    apu.mK,
    apu.W / apu.Hz,
    apu.W,
    apu.W / apu.m ** 2,
    apu.Jy,
    apu.uV / apu.m,
    apu.uV / apu.m,
    ]

COL_UNITS_DB = [
    apu.MHz,
    apu.MHz,
    apu.K,
    apu.K,
    apu.mK,
    cnv.dB_W_Hz,
    cnv.dB_W,
    cnv.dB_W_m2,
    cnv.dB_W_m2_Hz,
    cnv.dB_uV_m,
    cnv.dB_uV_m,
    ]


def test_cispr_limits_qtable_column():

    # this tests for a "bug" encountered in the wind-turbine notebook
    # if one uses the frequency column of the ra769 table,
    # an early version of the cispr function refused to work

    cont_lims = prot.ra769_limits(mode='continuum')[4:9]

    freqs = cont_lims['frequency']

    detector_dist = 30 * apu.m
    # we query the QP values and assume that they equal 'RMS';
    # as discussed above
    detector_type = 'QP'

    # case 1
    cispr11_lim, cispr11_bw = prot.cispr11_limits(
        freqs, detector_type=detector_type, detector_dist=detector_dist
        )

    assert_quantity_allclose(
        cispr11_lim,
        np.array([
            37., 37., 37., 37., 37.
            ]) * cnv.dB_uV_m,
        )


def test_ra769_limits():

    with pytest.raises(AssertionError):
        prot.ra769_limits(mode='FOO')

    with pytest.raises(AssertionError):
        prot.ra769_limits(scale='FOO')

    cont_lims = prot.ra769_limits(mode='continuum', scale='linear')
    cont_lims_dB = prot.ra769_limits(mode='continuum', scale='dB')
    spec_lims = prot.ra769_limits(mode='spectroscopy', scale='linear')
    spec_lims_dB = prot.ra769_limits(mode='spectroscopy', scale='dB')

    # test some "random" elements in all tables (values and units)
    # prefer derived quantities
    for idx, (row, col) in enumerate([
            (0, 'T_rms'),
            (5, 'Plim'),
            (10, 'Slim'),
            (15, 'Slim_nu'),
            (20, 'Efield'),
            ]):
        assert_allclose(cont_lims[row][col], CONT_LIMS_VALUES[idx], **TOL_KWARGS)
        assert_allclose(cont_lims_dB[row][col], CONT_LIMS_DB_VALUES[idx])

    for idx, (row, col) in enumerate([
            (0, 'T_rms'),
            (3, 'Plim'),
            (6, 'Slim'),
            (9, 'Slim_nu'),
            (12, 'Efield'),
            ]):
        assert_allclose(spec_lims[row][col], SPEC_LIMS_VALUES[idx], **TOL_KWARGS)
        assert_allclose(spec_lims_dB[row][col], SPEC_LIMS_DB_VALUES[idx])

    for colname, colunit in zip(COL_NAMES, COL_UNITS):
        assert cont_lims.columns[colname].unit == colunit
        assert spec_lims.columns[colname].unit == colunit

    for colname, colunit in zip(COL_NAMES, COL_UNITS_DB):
        assert cont_lims_dB.columns[colname].unit == colunit
        assert spec_lims_dB.columns[colname].unit == colunit
