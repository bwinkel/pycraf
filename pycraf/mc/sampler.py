#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


__all__ = ['HistogramSampler']


class HistogramSampler(object):
    '''
    Sampler to get random values obeying a discrete(!) density distribution.

    With this class, one can use discrete densities (think of them as
    binned entities, aka histograms) to get random samples that follow
    the same (binned) distribution as the histogram. For simplicity
    the returned values are the N-dim indices that need to be fed
    into the histogram bin-ranges if one wants to convert them
    to physical values (see examples, below).

    Parameters
    ----------
    histvals : N-D `~numpy.ndarray`
        Discrete density distribution. (This is the histogram array, which
        one would get out of `~numpy.histogram` functions.)

    Returns
    -------
    hist_sampler : `~pycraf.mc.HistogramSampler`
        A `~pycraf.mc.HistogramSampler` instance.

    Examples
    --------

    A trivial one-dimensional use case::

        >>> import numpy as np
        >>> from pycraf import mc
        >>> from astropy.utils.misc import NumpyRNGContext

        >>> with NumpyRNGContext(1):
        ...     x = np.random.normal(0, 1, 100)

        >>> hist, bins = np.histogram(x, bins=16, range=(-4, 4))
        >>> mid_points = (bins[1:] + bins[:-1]) / 2

        >>> my_sampler = mc.HistogramSampler(hist)
        >>> with NumpyRNGContext(1):
        ...     indices = my_sampler.sample(10)
        ...     # Note that you could also do
        ...     # indices = my_sampler(10)
        >>> print(indices)
        [7 9 3 7 6 5 6 7 7 8]

        >>> print(mid_points[indices])  # doctest: +FLOAT_CMP
        [-0.25  0.75 -2.25 -0.25 -0.75 -1.25 -0.75 -0.25 -0.25  0.25]

    Works equally simple in 2D::

        >>> with NumpyRNGContext(1):
        ...     x = np.random.normal(0, 1, 1000)
        ...     y = np.random.normal(2, 2, 1000)

        >>> hist2d, xbins, ybins = np.histogram2d(
        ...     x, y, bins=(16, 16), range=((-4, 4), (-6, 10))
        ...     )
        >>> xmids = (xbins[1:] + xbins[:-1]) / 2
        >>> ymids = (ybins[1:] + ybins[:-1]) / 2

        >>> my_sampler = mc.HistogramSampler(hist2d)
        >>> with NumpyRNGContext(1):
        ...     indices = my_sampler.sample(10)
        >>> print(list(zip(*indices)))
        [(7, 8), (9, 6), (1, 7), (7, 5), (6, 4),
         (5, 7), (6, 7), (7, 6), (7, 8), (8, 6)]

        >>> print(list(zip(xmids[indices[0]], ymids[indices[1]])))  # doctest: +FLOAT_CMP
        [(-0.25, 2.5), (0.75, 0.5), (-3.25, 1.5), (-0.25, -0.5),
         (-0.75, -1.5), (-1.25, 1.5), (-0.75, 1.5), (-0.25, 0.5),
         (-0.25, 2.5), (0.25, 0.5)]

    It is also easily possible to apply weights. Just assume
    that one bin was observed exceptionally frequent::

        >>> weights = np.ones_like(x)
        >>> weights[500] = 1000

        >>> hist2d, xbins, ybins = np.histogram2d(
        ...     x, y, bins=(16, 16), range=((-4, 4), (-6, 10)),
        ...     weights=weights
        ...     )

        >>> my_sampler = mc.HistogramSampler(hist2d)
        >>> with NumpyRNGContext(1):
        ...     indices = my_sampler.sample(10)
        >>> print(list(zip(xmids[indices[0]], ymids[indices[1]])))  # doctest: +FLOAT_CMP
        [(-1.75, 4.5), (-0.25, 3.5), (-3.25, 1.5), (-1.75, 4.5),
         (-1.75, 4.5), (-1.75, 4.5), (-1.75, 4.5), (-1.75, 4.5),
         (-1.75, 4.5), (-1.25, 0.5)]

    As can be seen, the value `((1.25, -1.5))` is now exceptionally
    often sampled from the distribution.

    As discussed in the notes, for some use-cases a KDE might
    be the better tool::

        >>> from scipy.stats import gaussian_kde  # doctest: +SKIP

        >>> kernel = gaussian_kde((x, y))  # doctest: +SKIP
        >>> with NumpyRNGContext(1):  # doctest: +SKIP
        ...     values = kernel.resample(10)

        >>> print(*zip(values[0], values[1]))  # doctest: +SKIP +FLOAT_CMP
        [(-1.4708084392424643, 0.73081055816321849),
         (0.088396607804818894, 3.4075844477993105),
        ...
         (-2.0977896525658681, -0.2514770710536518),
         (0.26194085609813555, -0.93622928331194344)]

    Notes
    -----
    Even if you have continuous data to start with, using the histogram
    approach obviously works, as well (as is demonstrated in the examples).
    However, one looses the "continuous" property in the process.
    The resulting samples will always be just be able to work
    out the bin, but no continuous quantity can be reconstructed.
    For many use cases this is probably fine, but in others one
    might be better of by using Kernel Density Estimation.
    There is a function for this in `~scipy.stats`
    (`~scipy.stats.gaussian_kde`), which even allows works with
    multi-variate data and allows one to sample from the KDE PDF
    (see also the examples). Unfortunately, one cannot work with
    weighted data.
    '''

    def __init__(self, histvals):

        histvals = np.atleast_1d(histvals)

        self._hshape = histvals.shape
        self._ndim = histvals.ndim
        # cdf is flat, will need to unravel indices later
        self._cdf = np.cumsum(
            histvals.flatten().astype(np.float64, copy=False)
            )
        self._cdf /= self._cdf[-1]

    def sample(self, n):
        '''
        Sample from the (discrete) density distribution.

        Parameters
        ----------
        n : int
            Number of samples to draw.

        Returns
        -------
        Indices : tuple of `~numpy.ndarray`
            The indices of the drawn samples with respect to the
            discrete density array (aka histogram object). See
            `~pycraf.mc.HistogramSampler` for examples of use.
        '''

        rsamples = np.random.rand(n)
        rbins = np.searchsorted(self._cdf, rsamples)

        indices = np.unravel_index(rbins, self._hshape)

        if self._ndim == 1:
            return indices[0]
        else:
            return indices

    def __call__(self, n):
        '''
        Convenience method to allow using an *instance* of
        `~pycraf.mc.HistogramSampler` like a function::

            my_sampler = mc.HistogramSampler(hist)
            my_sampler(10)

        Calls `~pycraf.mc.HistogramSampler.sample` internally.
        '''

        return self.sample(n)
