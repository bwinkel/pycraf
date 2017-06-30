#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

import os
import numpy as np
from astropy import units as apu
# from astropy.units import Quantity, UnitsError
import astropy.constants as con
from astropy.table import QTable, Table
from .. import conversions as cnv


__all__ = ['protection_limits']


def protection_limits(mode='continuum', scale='dB'):

    modes = ['continuum', 'spectroscopy']
    scales = ['linear', 'dB']

    assert mode in modes, 'mode must be either "continuum" or "spectroscopy"'
    assert scale in scales, 'scale must be either "linear" or "dB"'
    midx = modes.index(mode)

    csv_name = 'ra_769_table{}_limits_{}.csv'.format(midx + 1, mode)
    this_dir, this_filename = os.path.split(__file__)
    csv_path = os.path.join(this_dir, '../itudata/ra.769-2', csv_name)

    csv_tab = np.genfromtxt(
        csv_path, delimiter=',', skip_header=1, names=True, dtype=np.float64
        )

    qtab = QTable(meta={'name': 'RA.769 {} limits'.format(mode)})
    qtab['frequency'] = csv_tab['freq0'] * apu.MHz
    qtab['bandwidth'] = csv_tab['Delta_f'] * apu.MHz
    qtab['T_A'] = csv_tab['T_A'] * apu.K
    qtab['T_rx'] = csv_tab['T_rx'] * apu.K

    qtab['T_rms'] = (
        (qtab['T_A'] + qtab['T_rx']) /
        np.sqrt(2000. * apu.s * qtab['bandwidth'])
        ).to(apu.mK)
    P_rms_nu = con.k_B * qtab['T_rms']
    qtab['P_rms_nu'] = P_rms_nu.to(
        apu.Watt / apu.Hz if scale == 'linear' else cnv.dB_W_Hz
        )
    Plim = 0.1 * P_rms_nu * qtab['bandwidth']
    qtab['Plim'] = Plim.to(apu.Watt if scale == 'linear' else cnv.dB_W)
    qtab['Plim_nu'] = (Plim / qtab['bandwidth']).to(
        apu.W / apu.Hz if scale == 'linear' else cnv.dB_W_Hz
        )
    Slim = 4. * np.pi / con.c ** 2 * qtab['frequency'] ** 2 * Plim
    qtab['Slim'] = Slim.to(
        (apu.Watt / apu.m ** 2) if scale == 'linear' else cnv.dB_W_m2
        )
    qtab['Slim_nu'] = (Slim / qtab['bandwidth']).to(
        apu.Jy if scale == 'linear' else cnv.dB_W_m2_Hz
        )

    # field strength from full bandwidth:
    qtab['Efield'] = (np.sqrt(cnv.R0 * Slim)).to(
        (apu.microvolt / apu.m) if scale == 'linear' else cnv.dB_uV_m
        )
    # qtab['Efield2'] = (qtab['Efield'] ** 2).to(dB_uV_m)

    # now normalize for 1 MHz BW for comparison with spectroscopy
    # TODO: is this correct???
    qtab['Efield_norm'] = np.sqrt(
        cnv.R0 * Slim * apu.MHz / qtab['bandwidth'].to(apu.MHz)
        ).to(
        (apu.microvolt / apu.m) if scale == 'linear' else cnv.dB_uV_m
        )
    # qtab['Efield2_norm'] = (qtab['Efield_norm'] ** 2).to(dB_uV_m)

    # table formatting doesn't seem to work for QTable, so we convert
    tab = Table(qtab)

    for col in ['frequency', 'bandwidth', 'T_A', 'T_rx']:
        tab[col].format = '%.0f'

    for col in ['T_rms']:
        tab[col].format = '%.3f'

    for col in [
            'P_rms_nu',
            'Plim', 'Plim_nu', 'Slim', 'Slim_nu',
            'Efield', 'Efield_norm'
            ]:
        tab[col].format = '%.1e' if scale == 'linear' else '%.1f'

    return tab
