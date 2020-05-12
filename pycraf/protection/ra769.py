#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

import os
import numpy as np
from astropy import units as apu
# from astropy.units import Quantity, UnitsError
# import astropy.constants as con
from astropy.table import QTable, Table
from astropy.utils.data import get_pkg_data_filename
from .. import conversions as cnv
from .. import utils


__all__ = ['ra769_calculate_entry', 'ra769_limits']


@utils.ranged_quantity_input(
    frequency=(1e-9, 1.e12, apu.Hz),
    bandwidth=(1e-9, 1.e11, apu.Hz),
    T_A=(1e-9, 100000, apu.K),
    T_rx=(1e-9, 100000, apu.K),
    integ_time=(1e-9, None, apu.s)
    )
def ra769_calculate_entry(
        frequency, bandwidth, T_A, T_rx,
        mode='continuum', scale='dB', integ_time=2000. * apu.s,
        ):
    '''
    Limits (single entry) for spectral line, continuum, and VLBI observations
    according to
    `ITU-R Rec RA.769 <https://www.itu.int/rec/R-REC-RA.769-2-200305-I/en>`_.


    Parameters
    ----------
    frequency : `~astropy.units.Quantity`
        Center frequency [Hz]
    bandwidth : `~astropy.units.Quantity`
        Assumed bandwidth [Hz]

        Note, if `mode='vlbi'` the bandwidth is irrelevant, since only the
        spectral power flux density is calculated.

    T_A : `~astropy.units.Quantity`
        Minimum antenna noise temperature [K]
    T_rx : `~astropy.units.Quantity`
        Receiver noise temperature [K]
    mode : str, optional
        Observing mode: 'continuum', 'spectroscopy', or 'vlbi'
        (default: 'continuum')
    scale : str, optional
        Default scale to use: 'linear', 'dB' (default: 'linear')
    integ_time : `~astropy.units.Quantity`, optional
        Integration time [s] (default: 2000)

        Note, if `mode='vlbi'` integration time is irrelevant, because the
        limits are based on 1% of the receiver noise plus antenna temperature.

    Returns
    -------
    T_rms : `~astropy.units.Quantity`
        System noise after integration [K]
    P_rms_nu : `~astropy.units.Quantity`
        System noise power spectral density [W/Hz, dB(W/Hz)]
    Plim : `~astropy.units.Quantity`
        Power limit [W, dB(W)]
    Plim_nu : `~astropy.units.Quantity`
        Spectral power limit [W/Hz, dB(W/Hz)]
    Slim : `~astropy.units.Quantity`
        Power flux density (pfd) limit [W/m^2, dB(W/m^2)]
    Slim_nu : `~astropy.units.Quantity`
        Spectral power flux density limit [Jy, dB(W/m^2/Hz)]
    Efield : `~astropy.units.Quantity`
        Electrical field strength limit [uV/m, db(uV^2/m^2)]
    Efield_norm : `~astropy.units.Quantity`
        As `Efield` but normalized to 1 MHz bandwidth [uV/m, db(uV^2/m^2)]

    Note, if `mode='vlbi'` only `Slim_nu` is returned.

    Notes
    -----
    Because all columns are Astropy Quantities (see
    `~astropy.units.Quantity`), one can easily convert between linear and
    log-scale at a later stage. the `scale` parameter just defines the
    scale to use initially.
    '''

    modes = ['continuum', 'spectroscopy', 'vlbi']
    scales = ['linear', 'dB']

    assert mode in modes, (
        'mode must be either "continuum", "spectroscopy", or "vlbi"'
        )
    assert scale in scales, 'scale must be either "linear" or "dB"'

    if mode != 'vlbi':

        T_rms = (
            (T_A + T_rx) /
            np.sqrt(integ_time * bandwidth)
            ).to(apu.mK)

        _P_rms_nu = cnv.KB * T_rms

        _Plim = 10 * apu.percent * _P_rms_nu * bandwidth
        _Slim = 4. * np.pi / cnv.C ** 2 * frequency ** 2 * _Plim

        P_rms_nu = _P_rms_nu.to(
            apu.Watt / apu.Hz if scale == 'linear' else cnv.dB_W_Hz
            )
        Plim = _Plim.to(apu.Watt if scale == 'linear' else cnv.dB_W)
        Plim_nu = (_Plim / bandwidth).to(
            apu.W / apu.Hz if scale == 'linear' else cnv.dB_W_Hz
            )
        Slim = _Slim.to(
            (apu.Watt / apu.m ** 2) if scale == 'linear' else cnv.dB_W_m2
            )
        Slim_nu = (_Slim / bandwidth).to(
            apu.Jy if scale == 'linear' else cnv.dB_W_m2_Hz
            )

        # field strength from full bandwidth:
        Efield = (np.sqrt(cnv.R0 * _Slim)).to(
            (apu.microvolt / apu.m) if scale == 'linear' else cnv.dB_uV_m
            )
        # Efield2 = (Efield ** 2).to(dB_uV_m)

        # now normalize for 1 MHz BW for comparison with spectroscopy
        # TODO: is this correct???
        Efield_norm = np.sqrt(
            cnv.R0 * _Slim * apu.MHz / bandwidth.to(apu.MHz)
            ).to(
            (apu.microvolt / apu.m) if scale == 'linear' else cnv.dB_uV_m
            )
        # Efield2_norm = (Efield_norm ** 2).to(dB_uV_m)

        return (
            T_rms, P_rms_nu, Plim, Plim_nu, Slim, Slim_nu, Efield, Efield_norm
            )

    else:

        # The tolerable interference level is determined by the requirement
        # that the power level of the interfering signal should be no more
        # than 1% of the receiver noise power to prevent serious errors in
        # the measurement of the amplitude of the cosmic signals.

        _Plim_nu = 1 * apu.percent * cnv.KB * (T_rx + T_A)
        _Slim_nu = 4. * np.pi / cnv.C ** 2 * frequency ** 2 * _Plim_nu

        Slim_nu = _Slim_nu.to(
            apu.Jy if scale == 'linear' else cnv.dB_W_m2_Hz
            )

        return Slim_nu


@utils.ranged_quantity_input(integ_time=(1e-9, None, apu.s))
def ra769_limits(mode='continuum', scale='dB', integ_time=2000. * apu.s):
    '''
    Limits for spectral line, continuum, and VLBI observations according to
    `ITU-R Rec RA.769 <https://www.itu.int/rec/R-REC-RA.769-2-200305-I/en>`_.


    Parameters
    ----------
    mode : str, optional
        Observing mode: 'continuum', 'spectroscopy', or 'vlbi'
        (default: 'continuum')
    scale : str, optional
        Default scale to use: 'linear', 'dB' (default: 'linear')
    integ_time : `~astropy.units.Quantity`, optional
        Integration time [s] (default: 2000)

        Note, if `mode='vlbi'` integration time is irrelevant, because the
        limits are based on 1% of the receiver noise plus antenna temperature.

    Returns
    -------
    ra769_limits : `~astropy.table.Table`
        A table with the following columns:

        - "frequency" [MHz]
        - "bandwidth" [MHz]
        - "T_A" [K]
        - "T_rx" [K]
        - "T_rms" [K]
        - "P_rms_nu" [W/Hz, dB(W/Hz)]
        - "Plim" [W, dB(W)]
        - "Plim_nu" [W/Hz, dB(W/Hz)]
        - "Slim" [W/m^2, dB(W/m^2)]
        - "Slim_nu" [Jy, dB(W/m^2/Hz)]
        - "Efield" [uV/m, db(uV^2/m^2)]
        - "Efield_norm", normalized to 1 MHz bandwidth [uV/m, db(uV^2/m^2)]

        If `mode='vlbi'` the returned table has the following entries, only

        - "frequency" [MHz]
        - "bandwidth" [MHz]
        - "T_A" [K]
        - "T_rx" [K]
        - "Slim_nu" [Jy, dB(W/m^2/Hz)]


    Notes
    -----
    Because all columns are Astropy Quantities (see
    `~astropy.units.Quantity`), one can easily convert between linear and
    log-scale at a later stage. the `scale` parameter just defines the
    scale to use initially.
    '''

    modes = ['continuum', 'spectroscopy', 'vlbi']
    scales = ['linear', 'dB']

    assert mode in modes, (
        'mode must be either "continuum", "spectroscopy", or "vlbi"'
        )
    assert scale in scales, 'scale must be either "linear" or "dB"'

    csv_mode = mode if mode != 'vlbi' else 'continuum'
    midx = modes.index(csv_mode)
    csv_name = 'ra_769_table{}_limits_{}.csv'.format(midx + 1, csv_mode)
    csv_path = get_pkg_data_filename('../itudata/ra.769-2/' + csv_name)

    csv_tab = np.genfromtxt(
        csv_path, delimiter=',', skip_header=1, names=True, dtype=np.float64
        )

    qtab = QTable(meta={'name': 'RA.769 {} limits'.format(mode)})
    freq = csv_tab['freq0'] * apu.MHz
    bandwidth = csv_tab['Delta_f'] * (
        apu.kHz if mode == 'spectroscopy' else apu.MHz
        )

    qtab['frequency'] = freq
    if mode != 'vlbi':
        qtab['bandwidth'] = bandwidth
    qtab['T_A'] = csv_tab['T_A'] * apu.K
    qtab['T_rx'] = csv_tab['T_rx'] * apu.K

    if mode != 'vlbi':

        (
            qtab['T_rms'],
            qtab['P_rms_nu'],
            qtab['Plim'],
            qtab['Plim_nu'],
            qtab['Slim'],
            qtab['Slim_nu'],
            qtab['Efield'],
            qtab['Efield_norm'],
            ) = ra769_calculate_entry(
            freq, bandwidth, qtab['T_A'], qtab['T_rx'],
            mode=mode, scale=scale, integ_time=integ_time,
            )

    else:

        qtab['Slim_nu'] = ra769_calculate_entry(
            freq, 1 * apu.Hz, qtab['T_A'], qtab['T_rx'],
            mode=mode, scale=scale, integ_time=integ_time,
            )

    # table formatting doesn't seem to work for QTable, so we temporarily
    # convert to Table; interestingly, if we cast back to QTable afterwards
    # the format specifier is still valid (gets converted to QuantityInfo...)
    tab = Table(qtab)

    for col in ['frequency', 'bandwidth', 'T_A', 'T_rx']:
        try:
            tab[col].format = '%.0f'
        except KeyError:
            pass

    for col in ['T_rms']:
        try:
            tab[col].format = '%.3f'
        except KeyError:
            pass

    for col in [
            'P_rms_nu',
            'Plim', 'Plim_nu', 'Slim', 'Slim_nu',
            'Efield', 'Efield_norm'
            ]:
        try:
            tab[col].format = '%.1e' if scale == 'linear' else '%.1f'
        except KeyError:
            pass

    return QTable(tab)
