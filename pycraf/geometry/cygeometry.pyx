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
    exp, sqrt, fabs, M_PI, sin, cos, tan, asin, acos, atan2, fmod
    )
import numpy as np

np.import_array()

# __all__ = ['inverse', 'direct']


cdef double DEG2RAD = M_PI / 180.
cdef double RAD2DEG = 180. / M_PI
cdef double M_2PI = 2 * M_PI

# see https://en.wikipedia.org/wiki/Vincenty's_formulae

cdef inline double _true_angular_distance(
        double lon1_rad, double lat1_rad,
        double lon2_rad, double lat2_rad,
        ) nogil:

    cdef:
        double sin_diff_lon = sin(lon2_rad - lon1_rad)
        double cos_diff_lon = cos(lon2_rad - lon1_rad)
        double sin_lat1 = sin(lat1_rad)
        double sin_lat2 = sin(lat2_rad)
        double cos_lat1 = cos(lat1_rad)
        double cos_lat2 = cos(lat2_rad)
        double num1 = cos_lat2 * sin_diff_lon
        double num2 = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_diff_lon
        double denominator = (
            sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_diff_lon
            )

    return atan2(
        sqrt(num1 ** 2 + num2 ** 2), denominator
        )


cdef inline double _great_circle_bearing(
        double lon1_rad, double lat1_rad,
        double lon2_rad, double lat2_rad,
        ) nogil:

    cdef:
        double sin_diff_lon = sin(lon2_rad - lon1_rad)
        double cos_diff_lon = cos(lon2_rad - lon1_rad)
        double sin_lat1 = sin(lat1_rad)
        double sin_lat2 = sin(lat2_rad)
        double cos_lat1 = cos(lat1_rad)
        double cos_lat2 = cos(lat2_rad)
        double a = cos_lat2 * sin_diff_lon
        double b = cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_diff_lon

    return atan2(a, b)


def true_angular_distance_cython(
        lon1_deg, lat1_deg,
        lon2_deg, lat2_deg,
        out_ang_dist_deg=None,
        ):
    '''
    Parallelized true angular distance.
    '''

    cdef:

        # the memory view leads to an error:
        # ValueError: buffer source array is read-only
        # but new cython version should support it!?
        # double [::] _lon1_rad, _lat1_rad, _lon2_rad, _lat2_rad
        # double [::] _out_ang_dist
        np.ndarray[double] _lon1_deg, _lat1_deg, _lon2_deg, _lat2_deg
        np.ndarray[double] _out_ang_dist_deg

        int i, size

    it = np.nditer(
        [
            lon1_deg, lat1_deg, lon2_deg, lat2_deg,
            out_ang_dist_deg
            ],
        flags=['external_loop', 'buffered', 'delay_bufalloc'],
        op_flags=[
            ['readonly'], ['readonly'], ['readonly'], ['readonly'],
            ['readwrite', 'allocate'],
            ],
        op_dtypes=[
            'float64', 'float64', 'float64', 'float64',
            'float64',
            ]
        )

    # it would be better to use the context manager but
    # "with it:" requires numpy >= 1.14

    it.reset()

    for itup in it:
        _lon1_deg = itup[0]
        _lat1_deg = itup[1]
        _lon2_deg = itup[2]
        _lat2_deg = itup[3]
        _out_ang_dist_deg = itup[4]

        size = _lon1_deg.shape[0]

        for i in prange(size, nogil=True):

            _out_ang_dist_deg[i] = _true_angular_distance(
                _lon1_deg[i] * DEG2RAD,
                _lat1_deg[i] * DEG2RAD,
                _lon2_deg[i] * DEG2RAD,
                _lat2_deg[i] * DEG2RAD,
                ) * RAD2DEG

    return it.operands[4]


def great_circle_bearing_cython(
        lon1_deg, lat1_deg,
        lon2_deg, lat2_deg,
        out_bearing_deg=None,
        ):
    '''
    Parallelized true angular distance.
    '''

    cdef:

        # the memory view leads to an error:
        # ValueError: buffer source array is read-only
        # but new cython version should support it!?
        # double [::] _lon1_rad, _lat1_rad, _lon2_rad, _lat2_rad
        # double [::] _out_ang_dist
        np.ndarray[double] _lon1_deg, _lat1_deg, _lon2_deg, _lat2_deg
        np.ndarray[double] _out_bearing_deg

        int i, size

    it = np.nditer(
        [
            lon1_deg, lat1_deg, lon2_deg, lat2_deg,
            out_bearing_deg
            ],
        flags=['external_loop', 'buffered', 'delay_bufalloc'],
        op_flags=[
            ['readonly'], ['readonly'], ['readonly'], ['readonly'],
            ['readwrite', 'allocate'],
            ],
        op_dtypes=[
            'float64', 'float64', 'float64', 'float64',
            'float64',
            ]
        )

    # it would be better to use the context manager but
    # "with it:" requires numpy >= 1.14

    it.reset()

    for itup in it:
        _lon1_deg = itup[0]
        _lat1_deg = itup[1]
        _lon2_deg = itup[2]
        _lat2_deg = itup[3]
        _out_bearing_deg = itup[4]

        size = _lon1_deg.shape[0]

        for i in prange(size, nogil=True):

            _out_bearing_deg[i] = _great_circle_bearing(
                _lon1_deg[i] * DEG2RAD,
                _lat1_deg[i] * DEG2RAD,
                _lon2_deg[i] * DEG2RAD,
                _lat2_deg[i] * DEG2RAD,
                ) * RAD2DEG

    return it.operands[4]
