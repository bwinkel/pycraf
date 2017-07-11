#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

from astropy import units as apu
import numpy as np
from .. import utils
from .cygeodesics import inverse_cython, direct_cython


__all__ = ['inverse', 'direct']


@utils.ranged_quantity_input(
    lon1=(-np.pi, np.pi, apu.rad),
    lat1=(-np.pi / 2, np.pi / 2, apu.rad),
    lon2=(-np.pi, np.pi, apu.rad),
    lat2=(-np.pi / 2, np.pi / 2, apu.rad),
    strip_input_units=True,
    output_unit=(apu.m, apu.rad, apu.rad)
    )
def inverse(
        lon1, lat1,
        lon2, lat2,
        eps=1.e-12,  # corresponds to approximately 0.06mm
        maxiter=50,
        ):

    return inverse_cython(lon1, lat1, lon2, lat2, eps=eps, maxiter=maxiter)


@utils.ranged_quantity_input(
    lon1=(-np.pi, np.pi, apu.rad),
    lat1=(-np.pi / 2, np.pi / 2, apu.rad),
    bearing1=(-np.pi, np.pi, apu.rad),
    dist=(0.1, None, apu.m),
    strip_input_units=True,
    output_unit=(apu.rad, apu.rad, apu.rad)
    )
def direct(
        lon1, lat1,
        bearing1, dist,
        eps=1.e-12,  # corresponds to approximately 0.06mm
        maxiter=50,
        ):

    return direct_cython(
        lon1, lat1, bearing1, dist, eps=eps, maxiter=maxiter, wrap=True
        )


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
