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
from astropy.table import QTable
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
        )
    qtab['P_rms_nu'] = con.k_B * qtab['T_rms']
    qtab['Plim'] = 0.1 * qtab['P_rms_nu'] * qtab['bandwidth']
    qtab['Slim'] = (
        4. * np.pi / con.c ** 2 * qtab['frequency'] ** 2 * qtab['Plim']
        )
    qtab['Slim_nu'] = qtab['Slim'] / qtab['bandwidth']

    # field strength from full bandwidth:
    qtab['Efield'] = np.sqrt(R0 * qtab['Slim'])
    qtab['Efield2'] = (qtab['Efield'] ** 2).to(dB_uV_m)

    # now normalize for 1 MHz BW for comparison with spectroscopy
    # TODO: is this correct???
    qtab['Efield_norm'] = np.sqrt(R0 * qtab['Slim'] *
        apu.MHz / qtab['bandwidth'].to(apu.MHz)
        )
    qtab['Efield2_norm'] = (qtab['Efield_norm'] ** 2).to(dB_uV_m)

    return qtab


