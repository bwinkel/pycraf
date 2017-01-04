#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

import pytest
from astropy import units as apu
import numpy as np
from .. import conversions as cnv


__all__ = [
    'check_astro_quantities',
    ]


def tiny_fraction(num, digit=6):
    '''
    Return a tiny fraction of a number (in the specified digit)
    '''

    num = np.abs(num)
    num = np.max([num, 1.e-30])
    return 10 ** (np.log10(num) - digit)


def check_astro_quantities(func, args_list, invalid_unit=apu.byte):

    def make_args(alist):

        args = []
        for lowval, hival, unit in args_list:

            if lowval is None and hival is not None:
                lowval = hival
            elif lowval is not None and hival is None:
                hival = lowval
            elif lowval is None and hival is None:
                lowval = hival = 0.

            args.append(np.mean((lowval, hival)) * unit)

        return args

    for case in range(len(args_list)):

        args = make_args(args_list)

        # test for value ranges

        inv_lowval = args_list[case][0]
        if inv_lowval is not None:
            inv_lowval -= tiny_fraction(inv_lowval)
            args[case] = inv_lowval * args[case].unit

            with pytest.raises(AssertionError):
                func(*args)

            args[case] = args_list[case][0] * args[case].unit

        inv_hival = args_list[case][1]
        if inv_hival is not None:
            inv_hival += tiny_fraction(inv_hival)
            args[case] = inv_hival * args[case].unit

            with pytest.raises(AssertionError):
                func(*args)

            args[case] = args_list[case][1] * args[case].unit

    for case in range(len(args_list)):

        args = make_args(args_list)

        # test for wrong unit
        args[case] = args[case].value * invalid_unit

        with pytest.raises(apu.UnitsError):
            func(*args)

        # test for missing unit
        args[case] = args[case].value

        with pytest.raises(TypeError):
            func(*args)
