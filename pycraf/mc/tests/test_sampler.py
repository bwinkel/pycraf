#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import assert_quantity_allclose, remote_data
from astropy.utils.misc import NumpyRNGContext
from astropy import units as apu
from ...utils import check_astro_quantities
from ... import mc


TOL_KWARGS = {'atol': 1.e-4, 'rtol': 1.e-4}


class TestHistogramSamper():

    def setup(self):

        with NumpyRNGContext(1):
            self.x = np.random.normal(0, 1, 1000)
            self.y = np.random.normal(2, 2, 1000)

        self.xbins = np.linspace(-4, 4, 17)
        self.ybins = np.linspace(-6, 10, 17)
        self.xmids = (self.xbins[1:] + self.xbins[:-1]) / 2
        self.ymids = (self.ybins[1:] + self.ybins[:-1]) / 2

        self.hist, _ = np.histogram(self.x, bins=self.xbins)
        self.hist2d, *_ = np.histogram2d(
            self.x, self.y, bins=[self.xbins, self.ybins],
            )

    def test_sample1d(self):

        my_sampler = mc.HistogramSampler(self.hist)
        with NumpyRNGContext(1):
            indices = my_sampler.sample(10)
        assert_equal(
            indices,
            np.array([7, 9, 1, 7, 6, 5, 6, 7, 7, 8]),
            )

        assert_allclose(
            self.xmids[indices],
            np.array([-0.25, 0.75, -3.25, -0.25, -0.75,
                      -1.25, -0.75, -0.25, -0.25, 0.25]),
            )

    def test_call(self):

        my_sampler = mc.HistogramSampler(self.hist)
        with NumpyRNGContext(1):
            indices = my_sampler(10)
        assert_equal(
            indices,
            np.array([7, 9, 1, 7, 6, 5, 6, 7, 7, 8]),
            )

        assert_allclose(
            self.xmids[indices],
            np.array([-0.25, 0.75, -3.25, -0.25, -0.75,
                      -1.25, -0.75, -0.25, -0.25, 0.25]),
            )

    def test_sample2d(self):

        my_sampler = mc.HistogramSampler(self.hist2d)
        with NumpyRNGContext(1):
            indices = my_sampler.sample(10)
        assert_equal(
            indices,
            np.array([
                [7, 9, 1, 7, 6, 5, 6, 7, 7, 8],
                [8, 6, 7, 5, 4, 7, 7, 6, 8, 6]
                ]),
            )

        assert_allclose(
            self.xmids[indices[0]],
            np.array([-0.25, 0.75, -3.25, -0.25, -0.75,
                      -1.25, -0.75, -0.25, -0.25, 0.25]),
            )

        assert_allclose(
            self.ymids[indices[1]],
            np.array([2.5, 0.5, 1.5, -0.5, -1.5,
                      1.5, 1.5, 0.5, 2.5, 0.5]),
            )
