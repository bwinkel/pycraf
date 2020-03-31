#!python
# -*- coding: utf-8 -*-
# cython: language_level=3
# cython: cdivision=True, boundscheck=False, wraparound=False
# cython: embedsignature=True

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

cimport cython
from cython.parallel import prange, parallel
cimport numpy as np
from numpy cimport PyArray_MultiIter_DATA as Py_Iter_DATA
from libc.math cimport (
    exp, sqrt, fabs, M_PI, sin, cos, tan, asin, acos, atan2, fmod, log10
    )
import numpy as np

np.import_array()

# __all__ = ['inverse', 'direct']


cdef double DEG2RAD = M_PI / 180.
cdef double RAD2DEG = 180. / M_PI
cdef double M_2PI = 2 * M_PI

# see https://en.wikipedia.org/wiki/Vincenty's_formulae

cdef inline double _ras_pattern(
        double phi, double d_wlen,
        double gmax, double g1,
        double phi_m, double phi_r,
        ) nogil:

    if (120. <= phi) and (phi <= 180.):
        return -12.
    elif (80. <= phi) and (phi < 120.):
        return -7.
    elif (34.1 <= phi) and (phi < 80.):
        return -12.
    elif (10. <= phi) and (phi < 34.1):
        return 34. - 30. * log10(phi)
    elif (phi_r <= phi) and (phi < 10.):
        return 29. - 25. * log10(phi)
    elif (phi_m <= phi) and (phi < phi_r):
        return g1
    elif (0 <= phi) and (phi < phi_m):
        return gmax - 2.5e-3 * (d_wlen * phi) ** 2
    else:
        # huh?
        return 0.


def ras_pattern_cython(
        phi, d_wlen, gmax, g1, phi_m, phi_r,
        gain=None,
        ):
    '''
    Parallelized RAS pattern (non-bessel part only).
    '''

    cdef:

        # the memory view leads to an error:
        # ValueError: buffer source array is read-only
        # but new cython version should support it!?
        # double [::] _lon1_rad, _lat1_rad, _lon2_rad, _lat2_rad
        # double [::] _out_ang_dist
        np.ndarray[double] _phi, _d_wlen, _gmax, _g1, _phi_m, _phi_r
        np.ndarray[double] _gain

        int i, size

    it = np.nditer(
        [
            phi, d_wlen, gmax, g1, phi_m, phi_r,
            gain
            ],
        flags=['external_loop', 'buffered', 'delay_bufalloc'],
        op_flags=[
            ['readonly'], ['readonly'], ['readonly'], ['readonly'],
            ['readonly'], ['readonly'],
            ['readwrite', 'allocate'],
            ],
        op_dtypes=[
            'float64', 'float64', 'float64', 'float64',
            'float64', 'float64',
            'float64',
            ]
        )

    # it would be better to use the context manager but
    # "with it:" requires numpy >= 1.14

    it.reset()

    for itup in it:
        _phi = itup[0]
        _d_wlen = itup[1]
        _gmax = itup[2]
        _g1 = itup[3]
        _phi_m = itup[4]
        _phi_r = itup[5]
        _gain = itup[6]

        size = _phi.shape[0]

        for i in prange(size, nogil=True):

            _gain[i] = _ras_pattern(
                _phi[i], _d_wlen[i], _gmax[i], _g1[i], _phi_m[i], _phi_r[i]
                )

    return it.operands[6]
