#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function, unicode_literals

import pytest
import numpy as np
from numpy.testing import assert_equal
from astropy.tests.helper import assert_quantity_allclose
from astropy import units as apu

from ... import conversions as cnv
from .. import low_freq_propagation as sm2028

TOL = {'rtol': 1.e-3}
C_M_S = 299792458.0
LAM_R = C_M_S / 1.e6 / (2 * np.pi)
E_LIM = 14.23 * apu.dB(apu.uV / apu.m)


def _erp(m_val):
    # eq. 3 yields kW -> +30 for dB(W)
    return (10 * np.log10((20 / LAM_R**4) * m_val**2 / 1000) + 30) * cnv.dB_W


def test_findE40(): 
    # SM.2028 Table 1: land (GT6) @ 1/10 MHz; case-insensitive lookup
    E40, sigma, eps = sm2028.findE40(np.array([1.0, 10.0]) * apu.MHz, 'land')
    assert_quantity_allclose(E40, [120.0, 90.0] * cnv.dB_uV_m, **TOL)
    assert_quantity_allclose(sigma, 3.e-3 * apu.S / apu.m, **TOL)
    assert_quantity_allclose(eps, 22.0 * cnv.dimless, **TOL)
    assert_quantity_allclose(
        sm2028.findE40(1.0 * apu.MHz, 'LAND')[0], E40[0], **TOL)
    assert_quantity_allclose(
        sm2028.findE40(1.0 * apu.MHz, 'sea_water_average_salinity')[0],
        153.0 * cnv.dB_uV_m, **TOL)

    with pytest.raises(ValueError, match='not recognized'):
        sm2028.findE40(1.0 * apu.MHz, 'ocean')
    with pytest.raises(Exception):
        sm2028.findE40(50.0 * apu.MHz, 'land')      # out of range


def test_low_prop():    
    # SM.2028 Table 3 reference
    m, m_type, d_tr, ERP, dist, regime = sm2028.low_prop(
        18.0 * apu.dB(apu.uA / apu.m), 3.0 * apu.m, 1.0 * apu.MHz, 'land', E_LIM)
    assert_quantity_allclose(m, 1.344e-3 * apu.A * apu.m**2, rtol=1.e-3)
    assert_quantity_allclose(ERP, -111.58 * cnv.dB_W, **TOL)
    assert_quantity_allclose(d_tr, 3349.65 * apu.m, rtol=1.e-4)
    assert_quantity_allclose(dist, 25.e-3 * apu.km, rtol=1.e-3)
    assert_equal((m_type, regime), ('coaxial', 'near-field'))

    with pytest.raises(ValueError, match='not recognized'):
        sm2028.low_prop(18.0 * apu.dB(apu.uA / apu.m), 3.0 * apu.m,
                        1.0 * apu.MHz, 'land', E_LIM, victim_location='airborne')


@pytest.mark.parametrize('Hm, d, freq, E, loc, regime, mtype, dist', [
    (-20.0, 10.0, 13.385, -54.86, 'ground_wave', 'GW 40dB/dec', 'coplanar', 4.832e3),
    (45.39,  3.0,  1.0,    14.23, 'ground_wave', 'GW 20dB/dec', 'coaxial',  113.5),
    (-20.0, 10.0, 13.385, -54.86, 'free_space',  'FS 20dB/dec', 'coplanar', None),
])
def test_low_prop_regimes(Hm, d, freq, E, loc, regime, mtype, dist):
    m_, m_type, d_tr, _, r, reg = sm2028.low_prop(
        Hm * apu.dB(apu.uA / apu.m), d * apu.m, freq * apu.MHz, 'land',
        E * apu.dB(apu.uV / apu.m), victim_location=loc)
    assert_equal((m_type, reg), (mtype, regime))
    if loc == 'free_space':
        assert np.isnan(d_tr.value)
    else:
        assert_quantity_allclose(r, dist * apu.m, rtol=1.e-2)


def test_low_prop_bwr():
    args = (18.0 * apu.dB(apu.uA / apu.m), 3.0 * apu.m, 1.0 * apu.MHz, 'land', E_LIM)
    d0 = sm2028.low_prop(*args)[4]
    d3 = [sm2028.low_prop(*args, BWR=b)[4] for b in (3.0, 3.0 * cnv.dB, 3.0 * apu.dB)]
    d6 = sm2028.low_prop(*args, BWR=6.0)[4]

    assert d0 > d3[0] > d6                      # threshold up -> distance down
    assert_quantity_allclose(d3[0], d3[1], **TOL)
    assert_quantity_allclose(d3[0], d3[2], **TOL)


@pytest.mark.parametrize('m, E, regime, dist', [
    (1.344890e-3, 14.23, 'near-field',       25.0),
    (0.861,       50.0,  'close near-field', 58.42),
    (3.15e-2,     14.23, 'GW 20dB/dec',      113.5),
])


def test_protection_distance(m, E, regime, dist):
    m_q = m * apu.A * apu.m**2
    r, reg = sm2028.protection_distance(
        m_q, _erp(m), 120.0 * cnv.dB_uV_m,
        E * apu.dB(apu.uV / apu.m), LAM_R * apu.m, 3349.65 * apu.m)
    assert_equal(reg, regime)
    assert_quantity_allclose(r, dist * apu.m, rtol=1.e-2)


def test_protection_distance_bwr():
    m = 1.344890e-3 * apu.A * apu.m**2
    head = (m, _erp(m.value), 120.0 * cnv.dB_uV_m)
    tail = (LAM_R * apu.m, 3349.65 * apu.m)

    d_manual = sm2028.protection_distance(
        *head, (14.23 + 3.0) * apu.dB(apu.uV / apu.m), *tail)[0]
    d_bwr = sm2028.protection_distance(
        *head, E_LIM, *tail, BWR=3.0)[0]
    assert_quantity_allclose(d_manual, d_bwr, **TOL)

def test_low_prop_vectorised():
    # guards against the N x N broadcasting bug in regime selection:
    # array inputs must yield (N,)-shaped outputs, matching scalar calls.
    N = 8
    Hm = np.linspace(0.0, 40.0, N) * apu.dB(apu.uA / apu.m)
    d = np.full(N, 3.0) * apu.m
    freq = np.full(N, 1.0) * apu.MHz
    E = np.full(N, 14.23) * apu.dB(apu.uV / apu.m)

    m, m_type, d_tr, ERP, dist, regime = sm2028.low_prop(Hm, d, freq, 'land', E)
    assert dist.shape == (N,)
    assert regime.shape == (N,)

    # each element matches the equivalent scalar call
    for i in range(N):
        *_, dist_i, reg_i = sm2028.low_prop(
            Hm[i], d[i], freq[i], 'land', E[i])
        assert_quantity_allclose(dist[i], dist_i, **TOL)
        assert_equal(regime[i], reg_i)