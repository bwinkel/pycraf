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
    22.36067977, 9.6051867e-23, 6.91394471e-17, 133615.75496,
    2.226776016,
    ]
SPEC_LIMS_DB_VALUES = [
    22.36067977, -220.17494187, -161.60274098, -208.7414233,
    6.953530701,
    ]

VLBI_LIMS_VALUES = [
    27321.043322845562, 1059608.0665059818, 159488145.94716346
    ]
VLBI_LIMS_DB_VALUES = [
    -215.63502720045301, -199.74854744197285, -177.97271590607113
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
    [apu.MHz],
    [apu.kHz, apu.MHz],
    [apu.K],
    [apu.K],
    [apu.mK],
    [apu.W / apu.Hz],
    [apu.W],
    [apu.W / apu.m ** 2],
    [apu.Jy],
    [apu.uV / apu.m],
    [apu.uV / apu.m],
    ]

COL_UNITS_DB = [
    [apu.MHz],
    [apu.kHz, apu.MHz],
    [apu.K],
    [apu.K],
    [apu.mK],
    [cnv.dB_W_Hz],
    [cnv.dB_W],
    [cnv.dB_W_m2],
    [cnv.dB_W_m2_Hz],
    [cnv.dB_uV_m],
    [cnv.dB_uV_m],
    ]

VLBI_COL_IDXS = [0, 2, 3, 8]


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
    vlbi_lims = prot.ra769_limits(mode='vlbi', scale='linear')
    vlbi_lims_dB = prot.ra769_limits(mode='vlbi', scale='dB')

    # test some "random" elements in all tables (values and units)
    # prefer derived quantities
    for idx, (row, col) in enumerate([
            (0, 'T_rms'),
            (5, 'Plim'),
            (10, 'Slim'),
            (15, 'Slim_nu'),
            (20, 'Efield'),
            ]):
        colidx = COL_NAMES.index(col)
        assert_quantity_allclose(
            cont_lims[row][col],
            CONT_LIMS_VALUES[idx] * COL_UNITS[colidx][-1],
            rtol=1.e-6
            )
        assert_quantity_allclose(
            cont_lims_dB[row][col],
            CONT_LIMS_DB_VALUES[idx] * COL_UNITS_DB[colidx][-1]
            )

    for idx, (row, col) in enumerate([
            (0, 'T_rms'),
            (3, 'Plim'),
            (6, 'Slim'),
            (9, 'Slim_nu'),
            (12, 'Efield'),
            ]):
        colidx = COL_NAMES.index(col)
        assert_quantity_allclose(
            spec_lims[row][col],
            SPEC_LIMS_VALUES[idx] * COL_UNITS[colidx][0],
            rtol=1.e-6
            )
        assert_quantity_allclose(
            spec_lims_dB[row][col],
            SPEC_LIMS_DB_VALUES[idx] * COL_UNITS_DB[colidx][0]
            )

    for idx, (row, col) in enumerate([
            (5, 'Slim_nu'),
            (10, 'Slim_nu'),
            (15, 'Slim_nu'),
            ]):
        colidx = COL_NAMES.index(col)
        assert_quantity_allclose(
            vlbi_lims[row][col],
            VLBI_LIMS_VALUES[idx] * COL_UNITS[colidx][0],
            rtol=1.e-6
            )
        assert_quantity_allclose(
            vlbi_lims_dB[row][col],
            VLBI_LIMS_DB_VALUES[idx] * COL_UNITS_DB[colidx][0]
            )

    for colname, colunits in zip(COL_NAMES, COL_UNITS):
        assert cont_lims.columns[colname].unit in colunits
        assert spec_lims.columns[colname].unit in colunits

    for colname, colunits in zip(COL_NAMES, COL_UNITS_DB):
        assert cont_lims_dB.columns[colname].unit in colunits
        assert spec_lims_dB.columns[colname].unit in colunits

    for colidx in VLBI_COL_IDXS:
        colname = COL_NAMES[colidx]
        colunits, colunits_db = COL_UNITS[colidx], COL_UNITS_DB[colidx]
        assert vlbi_lims.columns[colname].unit in colunits
        assert vlbi_lims_dB.columns[colname].unit in colunits_db
