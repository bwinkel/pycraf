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
# from astropy.utils.misc import NumpyRNGContext


TOL_KWARGS = {'atol': 0., 'rtol': 1.e-6}


@pytest.mark.skip(reason='failing on AppVeyor and Travis for unknown reason')
def test_cispr_limits():

    _clims = prot.cispr._cispr_limits
    params11 = prot.cispr.CISPR11_PARAMS
    params22 = prot.cispr.CISPR22_PARAMS

    # first test, if assert Quantity works
    with pytest.raises(TypeError):
        _clims(params11, 1)

    with pytest.raises(TypeError):
        _clims(params11, 1 * apu.Hz, detector_dist=30)

    with pytest.raises(apu.UnitsError):
        _clims(params11, 1 * apu.m)

    with pytest.raises(apu.UnitsError):
        _clims(params11, 1 * apu.Hz, detector_dist=30 * apu.Hz)

    with pytest.raises(AssertionError):
        _clims(params11, 1 * apu.Hz, detector_type='FOO')

    res = _clims(params11, 1 * apu.Hz)
    assert isinstance(res, tuple)
    assert res[0].unit == apu.uV / apu.m

    freqs = Quantity(np.linspace(10, 900, 7), apu.MHz)

    assert_quantity_allclose(
        _clims(
            params11, freqs,
            detector_type='RMS',
            detector_dist=30. * apu.m
            )[0],
        Quantity([
            16.78804018, 16.78804018, 37.58374043, 37.58374043,
            37.58374043, 37.58374043, 37.58374043
            ], apu.uV / apu.m),
        )

    assert_quantity_allclose(
        _clims(
            params22, freqs,
            detector_type='RMS',
            detector_dist=30. * apu.m
            )[0],
        Quantity([
            53.08844442, 53.08844442, 118.85022274, 118.85022274,
            118.85022274, 118.85022274, 118.85022274
            ], apu.uV / apu.m),
        )

    assert_quantity_allclose(
        _clims(
            params11, freqs,
            detector_type='RMS',
            detector_dist=3000. * apu.m
            )[0],
        Quantity([
            0.1678804, 0.1678804, 0.3758374, 0.3758374, 0.3758374,
            0.3758374, 0.3758374
            ], apu.uV / apu.m),
        )

    assert_quantity_allclose(
        _clims(
            params22, freqs,
            detector_type='RMS',
            detector_dist=3000. * apu.m
            )[0],
        Quantity([
            0.53088444, 0.53088444, 1.18850223, 1.18850223, 1.18850223,
            1.18850223, 1.18850223
            ], apu.uV / apu.m),
        )

    assert_quantity_allclose(
        _clims(
            params11, freqs,
            detector_type='QP',
            detector_dist=30. * apu.m
            )[0],
        Quantity([
            31.6227766, 31.6227766, 70.79457844, 70.79457844,
            70.79457844, 70.79457844, 70.79457844
            ], apu.uV / apu.m),
        )

    assert_quantity_allclose(
        _clims(
            params22, freqs,
            detector_type='QP',
            detector_dist=30. * apu.m
            )[0],
        Quantity([
            100., 100., 223.87211386, 223.87211386,
            223.87211386, 223.87211386, 223.87211386
            ], apu.uV / apu.m),
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


def test_ras_protection_limits():

    with pytest.raises(AssertionError):
        prot.protection_limits(mode='FOO')

    with pytest.raises(AssertionError):
        prot.protection_limits(scale='FOO')

    cont_lims = prot.protection_limits(mode='continuum', scale='linear')
    cont_lims_dB = prot.protection_limits(mode='continuum', scale='dB')
    spec_lims = prot.protection_limits(mode='spectroscopy', scale='linear')
    spec_lims_dB = prot.protection_limits(mode='spectroscopy', scale='dB')

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
