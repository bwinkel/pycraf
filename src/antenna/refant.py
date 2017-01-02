#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

# from functools import partial, lru_cache
from astropy import units as apu
import numpy as np
from .. import conversions as cnv


__all__ = [
    'simple_model',
    ]


@apu.quantity_input(
    phi=apu.deg, diameter=apu.m, wavelength=apu.m, eta_a=apu.percent
    )
def simple_model(
        phi, diameter, wavelength, eta_a=100. * apu.percent, do_bessel=False
        ):
    '''
    Antenna gain as a function of angular distance after ITU-R RA.1631.

    Parameters
    ----------
    phi - angular distance in degrees
    diameter - antenna diameter
    wavelength - observing wavelength
    do_bessel - if set to True, use Bessel function approx. for inner 1 deg
    eta_a - antenna efficiency (default: 100%)

    Returns
    -------
    antenna gain in dBi

    Notes
    -----
    See ITU-R RA.1631 for explanation and applicability of this model.


    Example
    -------

    from pycraf.antenna import *
    import numpy as np
    import matplotlib.pyplot as plt
    from astropy import units as apu

    phi = np.linspace(0, 20, 1000) * apu.deg
    diam = np.array([10, 50, 100])[:, np.newaxis] * apu.m
    gain = simple_model(phi, diam, 0.21 * apu.m)
    plt.plot(phi, gain.T, '-')
    plt.show()

    phi = np.linspace(0, 2, 10000) * apu.deg
    gain = simple_model(phi, 100 * apu.m, 0.21 * apu.m, do_bessel=True)
    plt.plot(phi, gain, '-')
    plt.show()
    '''

    _phi = phi.to(apu.deg).value
    _diam = diameter.to(apu.m).value
    _wlen = wavelength.to(apu.m).value
    _eta_a = eta_a.to(apu.percent).value / 100.

    assert np.all((_phi >= 0.) & (_phi <= 180.)), (
        'phi must be in range 0 to 180 deg'
        )
    assert np.all((_diam >= 0.) & (_diam <= 2000.)), (
        'diameter must be in range 0 to 2000 m'
        )
    assert np.all((_wlen >= 0.001) & (_wlen <= 2.)), (
        'wavelength must be in range 1 mm to 2 m'
        )

    # the following are independent on phi, no need to compute after broadcast
    _d_wlen = _diam / _wlen

    # Note, we use the version that accounts for antenna efficiency
    # see ITU radio astronomy handbook, page 50
    # _gmax = 20 * np.log10(np.pi * _d_wlen)
    _gmax = 10 * np.log10(_eta_a * (np.pi * _d_wlen) ** 2)

    _g1 = -1. + 15. * np.log10(_d_wlen)
    _phi_m = 20. / _d_wlen * np.sqrt(_gmax - _g1)
    _phi_r = 15.85 * _d_wlen ** -0.6

    # note automatic broadcasting should be possible
    # _tmp = np.broadcast(_phi, _diam, _wlen)
    (
        _phi, _d_wlen, _gmax, _g1, _phi_m, _phi_r,
        ) = np.broadcast_arrays(
        _phi, _d_wlen, _gmax, _g1, _phi_m, _phi_r,
        )
    _gain = np.empty(_phi.shape, np.float64)

    # case 1:
    _mask = (0 <= _phi) & (_phi < _phi_m)
    _gain[_mask] = _gmax[_mask] - 2.5e-3 * (_d_wlen[_mask] * _phi[_mask]) ** 2

    # case 2:
    _mask = (_phi_m <= _phi) & (_phi < _phi_r)
    _gain[_mask] = _g1[_mask]

    # case 3:
    _mask = (_phi_r <= _phi) & (_phi < 10.)
    _gain[_mask] = 29. - 25. * np.log10(_phi[_mask])

    # case 4:
    _mask = (10. <= _phi) & (_phi < 34.1)
    _gain[_mask] = 34. - 30. * np.log10(_phi[_mask])

    # case 5:
    _mask = (34.1 <= _phi) & (_phi < 80.)
    _gain[_mask] = -12.

    # case 6:
    _mask = (80. <= _phi) & (_phi < 120.)
    _gain[_mask] = -7.

    # case 7:
    _mask = (120. <= _phi) & (_phi <= 180.)
    _gain[_mask] = -12.

    if do_bessel:

        from scipy.special import j1

        _phi_0 = 69.88 / _d_wlen
        _x_pi = np.radians(np.pi / 2. * _d_wlen * _phi)

        # case 1:
        _mask = (0 <= _phi) & (_phi < _phi_0)
        _tmp_x = _x_pi[_mask]
        _gain[_mask] = _gmax[_mask] + 20 * np.log10(j1(2 * _tmp_x) / _tmp_x)

        # case 2:
        _mask = (_phi_0 <= _phi) & (_phi < 1.)
        _B_sqrt = 10 ** 1.6 * np.radians(np.pi * _d_wlen[_mask] / 2.)
        _tmp_x = _x_pi[_mask]
        _gain[_mask] = 20 * np.log10(
            _B_sqrt * np.cos(2 * _tmp_x - 0.75 * np.pi + 0.0953) / _tmp_x
            )

    return apu.Quantity(_gain, cnv.dB)


if __name__ == '__main__':
    print('This not a standalone python program! Use as module.')
