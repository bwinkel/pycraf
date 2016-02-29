#!/usr/bin/python
# -*- coding: utf-8 -*-
# Licensed under GPL v2 - see LICENSE

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )

import os
import numpy as np
from astropy import units as apu
from astropy.units import Quantity, UnitsError
import astropy.constants as con
from astropy.table import QTable, Table
from ..conversions import *


__all__ = ['protection_limits']


def protection_limits(mode='continuum'):

    modes = ['continuum', 'spectroscopy']

    assert mode in modes, 'mode must be either "continuum" or "spectroscopy"'
    midx = modes.index(mode)

    csv_name = 'ra_769_table{}_limits_{}.csv'.format(midx + 1, mode)
    this_dir, this_filename = os.path.split(__file__)
    csv_path = os.path.join(this_dir, 'data', csv_name)

    tab = np.genfromtxt(
        csv_path, delimiter=',', skip_header=1, names=True, dtype=np.float64
        )

    qtab = QTable(meta={'name': 'RA.769 {} limits'.format(mode)})
    qtab['frequency'] = tab['freq0'] * apu.MHz
    qtab['bandwidth'] = tab['Delta_f'] * apu.MHz
    qtab['T_A'] = tab['T_A'] * apu.K
    qtab['T_rx'] = tab['T_rx'] * apu.K

    qtab['T_rms'] = (
        (qtab['T_A'] + qtab['T_rx']) /
        np.sqrt(2000. * apu.s * qtab['bandwidth'])
        ).to(apu.mK)
    qtab['P_rms_nu'] = (con.k_B * qtab['T_rms']).to(apu.Watt / apu.Hz)
    qtab['Plim'] = (0.1 * qtab['P_rms_nu'] * qtab['bandwidth']).to(apu.Watt)
    qtab['Slim'] = (
        4. * np.pi / con.c ** 2 * qtab['frequency'] ** 2 * qtab['Plim']
        ).to(apu.Jy * apu.Hz)
    qtab['Slim_nu'] = (qtab['Slim'] / qtab['bandwidth']).to(apu.Jy)

    # field strength from full bandwidth:
    qtab['Efield'] = (np.sqrt(R0 * qtab['Slim'])).to(apu.microvolt / apu.m)
    # qtab['Efield2'] = (qtab['Efield'] ** 2).to(dB_uV_m)

    # now normalize for 1 MHz BW for comparison with spectroscopy
    # TODO: is this correct???
    qtab['Efield_norm'] = np.sqrt(R0 * qtab['Slim'] *
        apu.MHz / qtab['bandwidth'].to(apu.MHz)
        ).to(apu.microvolt / apu.m)
    # qtab['Efield2_norm'] = (qtab['Efield_norm'] ** 2).to(dB_uV_m)

    qtab_dB = QTable(meta={'name': 'RA.769 {} limits'.format(mode)})
    qtab_dB['frequency'] = qtab['frequency']
    qtab_dB['bandwidth'] = qtab['bandwidth']
    qtab_dB['T_A'] = qtab['T_A']
    qtab_dB['T_rx'] = qtab['T_rx']
    qtab_dB['T_rms'] = qtab['T_rms']
    qtab_dB['P_rms_nu'] = qtab['P_rms_nu']
    qtab_dB['Plim'] = qtab['Plim'].to(dB_W)
    qtab_dB['Slim'] = qtab['Slim'].to(dB_W_m2)
    qtab_dB['Slim_nu'] = qtab['Slim_nu'].to(dB_W_m2_Hz)
    qtab_dB['Efield'] = (qtab['Efield'] ** 2).to(dB_uV_m)
    qtab_dB['Efield_norm'] = (qtab['Efield_norm'] ** 2).to(dB_uV_m)

    # table formatting doesn't seem to work for QTable, so we convert

    tab = Table(qtab)
    tab_dB = Table(qtab_dB)

    for col in ['frequency', 'bandwidth', 'T_A', 'T_rx']:
        tab[col].format = '%.0f'
        tab_dB[col].format = '%.0f'

    for col in ['T_rms']:
        tab[col].format = '%.3f'
        tab_dB[col].format = '%.3f'

    for col in [
            'P_rms_nu', 'Plim', 'Slim', 'Slim_nu', 'Efield', 'Efield_norm'
            ]:
        tab[col].format = '%.1e'

    for col in [
            'Plim', 'Slim', 'Slim_nu', 'Efield', 'Efield_norm'
            ]:
        tab_dB[col].format = '%.1f'

    tab_dB['P_rms_nu'].format = '%.1f'

    return tab, tab_dB


