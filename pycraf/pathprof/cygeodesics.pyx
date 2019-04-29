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


cdef double WGS_a = 6378137.0
cdef double WGS_b = 6356752.314245
cdef double WGS_eps = 1 - (WGS_b / WGS_a) ** 2
cdef double WGS_ieps = 1 - (WGS_a / WGS_b) ** 2
cdef double WGS_f = 1 / 298.257223563
cdef double DEG2RAD = M_PI / 180.
cdef double RAD2DEG = 180. / M_PI
cdef double M_2PI = 2 * M_PI

# see https://en.wikipedia.org/wiki/Vincenty's_formulae

cdef (double, double, double) _inverse(
        double lon1_rad, double lat1_rad,
        double lon2_rad, double lat2_rad,
        double eps,
        int maxiter,
        ) nogil:

    cdef:
        # note: "a" is short for alpha, "s" for sigma in this function
        double U1, U2, L, lam, last_lam, a, s
        double sin_U1, cos_U1, sin_U2, cos_U2, tan_U1, tan_U2
        double sin_lam, cos_lam
        double sin_s, cos_s, sin_a, cos2_a, cos_2sm, cos2_2sm
        double C, u2, A, B, ds

        double dist, bearing1_rad, bearing2_rad

        int _iter

    tan_U1 = (1 - WGS_f) * tan(lat1_rad)
    tan_U2 = (1 - WGS_f) * tan(lat2_rad)
    L = lon2_rad - lon1_rad

    cos_U1 = 1. / sqrt(1 + tan_U1 ** 2)
    cos_U2 = 1. / sqrt(1 + tan_U2 ** 2)
    sin_U1 = tan_U1 * cos_U1
    sin_U2 = tan_U2 * cos_U2

    lam = last_lam = L
    _iter = 0

    while True:

        sin_lam = sin(lam)
        cos_lam = cos(lam)

        sin_s = sqrt(
            (cos_U2 * sin_lam) ** 2 +
            (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lam) ** 2
            )
        cos_s = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lam
        s = atan2(sin_s, cos_s)
        sin_a = cos_U1 * cos_U2 * sin_lam / sin_s
        cos2_a = 1 - sin_a ** 2
        cos_2sm = cos_s - 2 * sin_U1 * sin_U2 / cos2_a
        cos2_2sm = cos_2sm ** 2

        C = WGS_f / 16 * cos2_a * (4 + WGS_f * (4 - 3 * cos2_a))
        lam = L + (1 - C) * WGS_f * sin_a * (
            s + C * sin_s * (
                cos_2sm + C * cos_s * (-1 + 2 * cos2_2sm)
                )
            )

        if fabs(lam - last_lam) < eps or _iter > maxiter:
            break

        _iter += 1
        last_lam = lam

    u2 = cos2_a * (WGS_a ** 2 - WGS_b ** 2) / WGS_b ** 2
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))

    ds = B * sin_s * (
        cos_2sm + 0.25 * B * (
            cos_s * (-1 + 2 * cos2_2sm) -
            1. / 6. * B * cos_2sm * (-3 + 4 * sin_s ** 2) * (-3 + 4 * cos2_2sm)
            )
        )

    dist = WGS_b * A * (s - ds)
    bearing1_rad = atan2(
        cos_U2 * sin_lam,
        cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lam
        )
    bearing2_rad = atan2(
        cos_U1 * sin_lam,
        -sin_U1 * cos_U2 + cos_U1 * sin_U2 * cos_lam
        )

    return (dist, bearing1_rad, bearing2_rad)


def inverse_cython(
        lon1_rad, lat1_rad,
        lon2_rad, lat2_rad,
        double eps=1.e-12,  # corresponds to approximately 0.06mm
        int maxiter=50,
        out_dist=None,
        out_bearing1=None,
        out_bearing2=None
        ):
    '''
    As `inverse_cython` but parallelized. Needs testing...
    '''

    cdef:

        # the memory view leads to an error:
        # ValueError: buffer source array is read-only
        # but new cython version should support it!?
        # double [::] _lon1_rad, _lat1_rad, _lon2_rad, _lat2_rad
        # double [::] _out_dist, _out_bearing1, _out_bearing2
        np.ndarray[double] _lon1_rad, _lat1_rad, _lon2_rad, _lat2_rad
        np.ndarray[double] _out_dist, _out_bearing1, _out_bearing2

        int i, size

    it = np.nditer(
        [
            lon1_rad, lat1_rad, lon2_rad, lat2_rad,
            out_dist, out_bearing1, out_bearing2
            ],
        flags=['external_loop', 'buffered', 'delay_bufalloc'],
        op_flags=[
            ['readonly'], ['readonly'], ['readonly'], ['readonly'],
            ['readwrite', 'allocate'], ['readwrite', 'allocate'],
            ['readwrite', 'allocate'],
            ],
        op_dtypes=[
            'float64', 'float64', 'float64', 'float64',
            'float64', 'float64', 'float64'
            ]
        )

    # it would be better to use the context manager but
    # "with it:" requires numpy >= 1.14

    it.reset()

    for itup in it:
        _lon1_rad = itup[0]
        _lat1_rad = itup[1]
        _lon2_rad = itup[2]
        _lat2_rad = itup[3]
        _out_dist = itup[4]
        _out_bearing1 = itup[5]
        _out_bearing2 = itup[6]

        size = _lon1_rad.shape[0]

        for i in prange(size, nogil=True):

            (
                _out_dist[i],
                _out_bearing1[i],
                _out_bearing2[i],
                ) = _inverse(
                    _lon1_rad[i],
                    _lat1_rad[i],
                    _lon2_rad[i],
                    _lat2_rad[i],
                    eps,
                    maxiter,
                    )

    return it.operands[4:7]


cdef (double, double, double) _direct(
        double lon1_rad, double lat1_rad,
        double bearing1_rad,
        double dist,
        double eps,
        int maxiter,
        int cwrap
        ) nogil:

    cdef:
        # note: "a" is short for alpha, "s" for sigma in this function
        double U1, L, lam, a, s, s1, last_s
        double sin_U1, cos_U1, tan_U1, sin_a1, cos_a1
        double sin_lam, cos_lam
        double sin_s, cos_s, sin_a, cos2_a, cos_2sm, cos2_2sm
        double C, u2, A, B, ds

        double lon2_rad, lat2_rad, bearing2_rad

        int _iter

    tan_U1 = (1 - WGS_f) * tan(lat1_rad)

    cos_U1 = 1. / sqrt(1 + tan_U1 ** 2)
    sin_U1 = tan_U1 * cos_U1
    sin_a1 = sin(bearing1_rad)
    cos_a1 = cos(bearing1_rad)

    s1 = atan2(tan_U1, cos_a1)

    sin_a = cos_U1 * sin_a1
    sin2_a = sin_a ** 2
    cos2_a = 1 - sin2_a

    u2 = cos2_a * (WGS_a ** 2 - WGS_b ** 2) / WGS_b ** 2
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))

    s = last_s = dist / WGS_b / A
    _iter = 0

    while True:

        cos_2sm = cos(2 * s1 + s)
        cos2_2sm = cos_2sm ** 2

        sin_s = sin(s)
        cos_s = cos(s)

        ds = B * sin_s * (
            cos_2sm + 0.25 * B * (
                cos_s * (-1 + 2 * cos2_2sm) -
                1. / 6. * B * cos_2sm * (-3 + 4 * sin_s ** 2) *
                (-3 + 4 * cos2_2sm)
                )
            )

        s = dist / WGS_b / A + ds

        if fabs(s - last_s) < eps or _iter > maxiter:
            break

        _iter += 1
        last_s = s

    lat2_rad = atan2(
        sin_U1 * cos_s + cos_U1 * sin_s * cos_a1,
        (1 - WGS_f) * sqrt(
            sin2_a + (sin_U1 * sin_s - cos_U1 * cos_s * cos_a1) ** 2
            )
        )

    lam = atan2(
        sin_s * sin_a1,
        cos_U1 * cos_s - sin_U1 * sin_s * cos_a1
        )

    C = WGS_f / 16 * cos2_a * (4 + WGS_f * (4 - 3 * cos2_a))
    L = lam - (1 - C) * WGS_f * sin_a * (
        s + C * sin_s * (
            cos_2sm + C * cos_s * (-1 + 2 * cos2_2sm)
            )
        )

    lon2_rad = L + lon1_rad
    bearing2_rad = atan2(
        sin_a,
        -sin_U1 * sin_s + cos_U1 * cos_s * cos_a1
        )

    if cwrap:
        while lon2_rad < 0:
            lon2_rad += M_2PI

        lon2_rad = fmod(lon2_rad + M_PI, M_2PI) - M_PI

    return lon2_rad, lat2_rad, bearing2_rad


def direct_cython(
        lon1_rad, lat1_rad,
        bearing1_rad, dist_m,
        double eps=1.e-12,  # corresponds to approximately 0.06mm
        int maxiter=50,
        wrap=True,
        out_lon2=None,
        out_lat2=None,
        out_bearing2=None
        ):
    '''
    As `direct_cython` but parallelized. Needs testing...
    '''

    cdef:

        # the memory view leads to an error:
        # ValueError: buffer source array is read-only
        # but new cython version should support it!?
        # double [::] _lon1_rad, _lat1_rad, _lon2_rad, _lat2_rad
        # double [::] _out_dist, _out_bearing1, _out_bearing2
        np.ndarray[double] _lon1_rad, _lat1_rad, _bearing1_rad, _dist_m
        np.ndarray[double] _out_lon2, _out_lat2, _out_bearing2

        int cwrap = 1 if wrap else 0
        int i, size

    it = np.nditer(
        [
            lon1_rad, lat1_rad, bearing1_rad, dist_m,
            out_lon2, out_lat2, out_bearing2
            ],
        flags=['external_loop', 'buffered', 'delay_bufalloc'],
        op_flags=[
            ['readonly'], ['readonly'], ['readonly'], ['readonly'],
            ['readwrite', 'allocate'], ['readwrite', 'allocate'],
            ['readwrite', 'allocate'],
            ],
        op_dtypes=[
            'float64', 'float64', 'float64', 'float64',
            'float64', 'float64', 'float64'
            ]
        )

    # it would be better to use the context manager but
    # "with it:" requires numpy >= 1.14

    it.reset()

    for itup in it:
        _lon1_rad = itup[0]
        _lat1_rad = itup[1]
        _bearing1_rad = itup[2]
        _dist_m = itup[3]
        _out_lon2 = itup[4]
        _out_lat2 = itup[5]
        _out_bearing2 = itup[6]

        size = _lon1_rad.shape[0]

        for i in prange(size, nogil=True):

            (
                _out_lon2[i],
                _out_lat2[i],
                _out_bearing2[i],
                ) = _direct(
                    _lon1_rad[i],
                    _lat1_rad[i],
                    _bearing1_rad[i],
                    _dist_m[i],
                    eps,
                    maxiter,
                    cwrap,
                    )

    return it.operands[4:7]


cdef double ellipse_radius(double phi_rad, double a, double b) nogil:

    return a * b / sqrt(
        (a * sin(phi_rad)) ** 2 + (b * cos(phi_rad)) ** 2
        )


cdef double _area_wgs84(
        double lon1_rad, double lon2_rad,
        double lat1_rad, double lat2_rad
        ) nogil:
    '''
    Adapted from https://math.stackexchange.com/questions/1379341/how-to-find-the-surface-area-of-revolution-of-an-ellipsoid-from-ellipse-rotating
    '''

    cdef:

        double xi1, xi2, i1, i2, area

    xi1 = asin(
        WGS_ieps / WGS_b * sin(lat1_rad) *
        ellipse_radius(lat1_rad, WGS_a, WGS_b)
        )
    xi2 = asin(
        WGS_ieps / WGS_b * sin(lat2_rad) *
        ellipse_radius(lat2_rad, WGS_a, WGS_b)
        )

    i1 = 0.5 * xi1 + 0.25 * sin(2 * xi1)
    i2 = 0.5 * xi2 + 0.25 * sin(2 * xi2)

    area = (lon2_rad - lon1_rad) * WGS_a * WGS_b / WGS_ieps * (i2 - i1)

    return area


def area_wgs84_cython(lon1_rad, lon2_rad, lat1_rad, lat2_rad, out_area=None):

    cdef:

        np.ndarray[double] _lon1_rad, _lat1_rad, _lon2_rad, _lat2_rad
        np.ndarray[double] _out_area

        int i, size

    it = np.nditer(
        [
            lon1_rad, lat1_rad, lon2_rad, lat2_rad,
            out_area,
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
        _lon1_rad = itup[0]
        _lon2_rad = itup[1]
        _lat1_rad = itup[2]
        _lat2_rad = itup[3]
        _out_area = itup[4]

        size = _lon1_rad.shape[0]

        for i in prange(size, nogil=True):

            _out_area[i] = _area_wgs84(
                _lon1_rad[i], _lat1_rad[i], _lon2_rad[i], _lat2_rad[i],
                )

    return it.operands[4]


cdef inline int find_in_ordered(
        cython.floating[:] x,
        cython.floating x0,
        ) nogil:
    '''
    Find index of x0 in ordered vector x.

    Using a divide-and-conquer style algorithm.
    '''

    cdef:
        int l, h, i
        int length = x.shape[0]

    l = 0
    h = length - 1

    while x[l] <= x0 and x[h] >= x0:

        i = l + <int> ((x0 - x[l]) / (x[h] - x[l]) * (h - l))

        if x[i] < x0:
            l = i + 1
        elif x[i] > x0:
            h = i - 1
        else:
            return i

    return i


cdef inline double gauss1d(double offset, double s) nogil:

    return exp(-0.5 * offset * offset / s / s)


def regrid1d_with_x(
        cython.floating[:] x not None,
        cython.floating[:] y not None,
        cython.floating[:] x_new not None,
        cython.floating[:] y_new not None,  # output
        cython.floating width,
        bint regular=False,
        bint ordered=True,
        ):
    '''
    Regrid an array of values, measured at support x to a new support x_new.

    Example code::

        >>> from pycraf.pathprof.cygeodesics import regrid1d_with_x
        >>> import numpy as np
        >>> import matplotlib.pyplot as plt

        >>> x = np.linspace(1., 0., 100)
        >>> x_new = np.linspace(0., 1., 1000)
        >>> y_ = np.random.normal(0., 1., 100)
        >>> y_[50] = 10

        >>> y = np.empty_like(x)
        >>> y_new = np.empty_like(x_new)

        >>> regrid1d_with_x(x[::-1], y_[::-1], x, y, 0.005)
        >>> regrid1d_with_x(x[::-1], y_[::-1], x_new, y_new, 0.005)

        >>> plt.close()
        >>> plt.plot(x, y_, 'k-')
        >>> plt.plot(x, y, 'r-')
        >>> plt.plot(x_new, y_new, 'b-')
        >>> plt.show()
    '''

    cdef:

        double this_x, kv, ssum, norm

        int i, j, s, e
        int length = x.size
        int length_new = x_new.size

        double dx = fabs(x[0] - x[length - 1]) / length

    assert x.size == y.size, 'x and y must have equal size'

    # for i in prange(length_new, nogil=True):
    for i in range(length_new):

        this_x = x_new[i]

        # find optimal s, e (if in ordered mode)
        if regular:

            s = int((this_x - 5. * width) / dx - 0.5)
            e = int((this_x + 5. * width) / dx + 1.5)
            if s < 0:
                s = 0
            if e >= length:
                e = length

        elif ordered:
            s = find_in_ordered(x, this_x)
            e = s + 1

            while True:
                s -= 1
                if s < 1:
                    s = 0
                    break
                if fabs(x[s] - this_x) > 5. * width:
                    break

            while True:
                e += 1
                if e >= length:
                    e = length
                    break
                if fabs(x[e - 1] - this_x) > 5. * width:
                    break

        else:
            s = 0
            e = length

        # print(i, s, e)

        norm = 0.
        ssum = 0.
        for j in range(s, e):
            kv = gauss1d(x[j] - this_x, width)
            # inplace operation leads to an error:
            # Cannot read reduction variable in loop body
            # ssum += kv * y[j]
            # norm += kv
            ssum = ssum + kv * y[j]
            norm = norm + kv

        # print(i, s, e, norm, ssum)
        if fabs(norm) < 1.e-12:
            y_new[i] = 0.
        else:
            y_new[i] = ssum / norm


def regrid2d_with_x(
        cython.floating[:] x not None,
        cython.floating[:, :] y not None,
        cython.floating[:] x_new not None,
        cython.floating[:, :] y_new not None,  # output
        cython.floating width,
        bint regular=False,
        bint ordered=True,
        ):
    '''
    Like regrid1d_with_x but for batches of 1D arrays; openmp powered::

        from pycraf.pathprof.cygeodesics import regrid1d_with_x, regrid2d_with_x
        import numpy as np

        x = np.linspace(0., 1., 100)
        x_new = np.linspace(0., 1., 1000)
        y_ = np.random.normal(0., 1., (50, 100))  # 50x length-100 arrays
        y = np.empty((50, 100), dtype=np.float64)
        y_new = np.empty((50, 1000), dtype=np.float64)

        %timeit regrid1d_with_x(x, y_[0], x_new, y_new[0], 0.005, regular=True)
        %timeit regrid2d_with_x(x, y_, x_new, y_new, 0.005, regular=True)

    '''

    cdef:

        double this_x, kv, ssum, norm

        int n, i, j, s, e
        int maxn = y.shape[0]
        int length = x.size
        int length_new = x_new.size

        double dx = fabs(x[0] - x[length - 1]) / length

    assert x.size == y.shape[1], 'x and y[0] must have equal size'

    for i in range(length_new):

        this_x = x_new[i]

        # find optimal s, e (if in ordered mode)
        if regular:

            s = int((this_x - 5. * width) / dx - 0.5)
            e = int((this_x + 5. * width) / dx + 1.5)
            if s < 0:
                s = 0
            if e >= length:
                e = length

        elif ordered:
            s = find_in_ordered(x, this_x)
            e = s + 1

            while True:
                s -= 1
                if s < 1:
                    s = 0
                    break
                if fabs(x[s] - this_x) > 5. * width:
                    break

            while True:
                e += 1
                if e >= length:
                    e = length
                    break
                if fabs(x[e - 1] - this_x) > 5. * width:
                    break

        else:
            s = 0
            e = length

        # print(i, s, e)

        for n in prange(maxn, nogil=True):
            norm = 0.
            ssum = 0.
            for j in range(s, e):
                kv = gauss1d(x[j] - this_x, width)
                # inplace operation leads to an error:
                # Cannot read reduction variable in loop body
                # ssum += kv * y[j]
                # norm += kv
                ssum = ssum + kv * y[n, j]
                norm = norm + kv

            # print(i, s, e, norm, ssum)
            if fabs(norm) < 1.e-12:
                y_new[n, i] = 0.
            else:
                y_new[n, i] = ssum / norm
