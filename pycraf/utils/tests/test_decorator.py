#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
import astropy.units as apu
from astropy.tests.helper import assert_quantity_allclose
from ...utils import ranged_quantity_input


def test_ranged_quantity_input_simple():

    @ranged_quantity_input(a=(0, 1, apu.m))
    def func(a):
        return a ** 2

    assert_quantity_allclose(func(0.5 * apu.m), 0.25 * apu.m ** 2)

    with pytest.raises(ValueError):
        func(2 * apu.m)

    with pytest.raises(apu.UnitsError):
        func(2 * apu.s)

    with pytest.raises(TypeError):
        func(2)

    @ranged_quantity_input(a=(0, None, apu.m))
    def func(a):
        return a ** 2

    assert_quantity_allclose(func(2 * apu.m), 4 * apu.m ** 2)

    with pytest.raises(apu.UnitsError):
        func(2 * apu.s)

    @ranged_quantity_input(a=(0, 1, apu.m), b=(0, None, apu.s))
    def func(a, b):
        return a ** 2, 1 / b

    res = func(0.5 * apu.m, 2 * apu.s)
    assert_quantity_allclose(res[0], 0.25 * apu.m ** 2)
    assert_quantity_allclose(res[1], 0.5 / apu.s)

    # No Exception is raised, if one has a typo in the argument names!
    @ranged_quantity_input(a=(0, 1, apu.m), c=(0, None, apu.s))
    def func(a, b):
        return a ** 2, 1 / b

    res = func(0.5 * apu.m, -2 * apu.s)
    assert_quantity_allclose(res[0], 0.25 * apu.m ** 2)
    assert_quantity_allclose(res[1], -0.5 / apu.s)


def test_ranged_quantity_input_defaultargs():

    @ranged_quantity_input(a=(0, 1, apu.m))
    def func(a=0.5 * apu.m):
        return a ** 2

    assert_quantity_allclose(func(), 0.25 * apu.m ** 2)
    assert_quantity_allclose(func(0.1 * apu.m), 0.01 * apu.m ** 2)

    with pytest.raises(ValueError):
        func(2 * apu.m)

    with pytest.raises(apu.UnitsError):
        func(2 * apu.s)


def test_ranged_quantity_input_noneargs():

    @ranged_quantity_input(a=(0, 1, apu.m))
    def func(a=None):
        return a ** 2

    with pytest.raises(TypeError):
        func()

    with pytest.raises(ValueError):
        func(2 * apu.m)

    @ranged_quantity_input(a=(0, 1, apu.m), allow_none=True)
    def func(a=None):
        if a is None:
            a = 0.5
        return a ** 2

    assert_quantity_allclose(func(), 0.25)
    assert_quantity_allclose(func(0.1 * apu.m), 0.01 * apu.m ** 2)

    with pytest.raises(apu.UnitsError):
        func(2 * apu.s)


def test_ranged_quantity_input_stripinput():

    @ranged_quantity_input(a=(0, 1, apu.m), strip_input_units=True)
    def func(a):
        return a ** 2

    with pytest.raises(ValueError):
        func(2 * apu.m)

    assert_quantity_allclose(func(0.5 * apu.m), 0.25)

    @ranged_quantity_input(
        a=(0, 1, apu.m), b=(0, None, apu.s),
        strip_input_units=True
        )
    def func(a, b):
        return a ** 2, 1 / b

    res = func(0.5 * apu.m, 2 * apu.s)
    assert_quantity_allclose(res[0], 0.25)
    assert_quantity_allclose(res[1], 0.5)


def test_ranged_quantity_input_applyoutput():

    @ranged_quantity_input(
        a=(0, 1, apu.m),
        strip_input_units=True,
        output_unit=apu.m ** 2
        )
    def func(a):
        return a ** 2

    assert_quantity_allclose(func(0.5 * apu.m), 0.25 * apu.m ** 2)

    @ranged_quantity_input(
        a=(0, 1, apu.m), b=(0, None, apu.s),
        strip_input_units=True,
        output_unit=(apu.m ** 2, 1 / apu.s),
        )
    def func(a, b):
        return a ** 2, 1 / b

    res = func(0.5 * apu.m, 2 * apu.s)
    assert_quantity_allclose(res[0], 0.25 * apu.m ** 2)
    assert_quantity_allclose(res[1], 0.5 / apu.s)
