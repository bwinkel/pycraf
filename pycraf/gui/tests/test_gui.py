#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import pytest
import re
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import assert_quantity_allclose, remote_data
from astropy import units as apu
from astropy.units import Quantity
# from astropy.utils.misc import NumpyRNGContext
# from astropy.utils.data import get_pkg_data_filename
from PyQt5 import QtCore, QtWidgets
from .. import gui
from ... import conversions as cnv


LABEL_TEXT = '''
<style>
    table {
        color: black;
        width: 100%;
        text-align: center;
        font-family: "Futura-Light", sans-serif;
        font-weight: 400;
        font-size: 14px;
    }
    th {
        color: blue;
        font-size: 16px;
    }
    th, td { padding: 2px; }
    thead.th {
        height: 110%;
        border-bottom: solid 0.25em black;
    }
    .lalign { text-align: left; padding-left: 12px;}
    .ralign { text-align: right; padding-right: 12px; }
    </style>

<table>
  <thead>
    <tr>
      <th colspan="2">Radio properties</th>
      <th colspan="2">Path geometry</th>
      <th colspan="2">Path losses</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="lalign">a_e (50%)</td>
      <td class="ralign"> 8455 km</td>
      <td class="lalign">alpha_tr</td>
      <td class="ralign"> 122.081 deg</td>
      <td class="lalign">L_b0p (LoS)</td>
      <td class="ralign">125.0 dB</td>
    </tr>
    <tr>
      <td class="lalign">a_e (beta0)</td>
      <td class="ralign">19113 km</td>
      <td class="lalign">alpha_rt</td>
      <td class="ralign"> -57.390 deg</td>
      <td class="lalign">L_bd (Diffraction)</td>
      <td class="ralign">172.4 dB</td>
    </tr>
    <tr>
      <td class="lalign">beta0</td>
      <td class="ralign">1.25 %</td>
      <td class="lalign">eps_pt</td>
      <td class="ralign">   0.512 deg</td>
      <td class="lalign">L_bs (Troposcatter)</td>
      <td class="ralign">207.4 dB</td>
    </tr>
    <tr>
      <td class="lalign">N0</td>
      <td class="ralign">324.7</td>
      <td class="lalign">eps_pr</td>
      <td class="ralign">   3.458 deg</td>
      <td class="lalign">L_ba (Anomalous)</td>
      <td class="ralign">219.8 dB</td>
    </tr>
    <tr>
      <td class="lalign">Delta N</td>
      <td class="ralign">38.69 1 / km</td>
      <td class="lalign">h_eff</td>
      <td class="ralign">   510.0 m</td>
      <td class="lalign">L_b (Total)</td><td class="ralign">172.4 dB</td>
    </tr>
    <tr>
      <td class="lalign"></td>
      <td class="ralign"></td>
      <td class="lalign">Path type</td>
      <td class="ralign">Trans-horizon</td>
      <td class="lalign" style="color: blue;">L_b_corr (Total + Clutter)</td>
      <td class="ralign" style="color: blue;">171.8 dB</td>
    </tr>
  </tbody>
</table>
'''


def _set_parameters(ui):

    ui.freqDoubleSpinBox.setValue(1.0)
    ui.timepercentDoubleSpinBox.setValue(2.0)
    ui.stepsizeDoubleSpinBox.setValue(10.0)
    ui.tempDoubleSpinBox.setValue(293.15)
    ui.pressDoubleSpinBox.setValue(980.0)
    ui.txLonDoubleSpinBox.setValue(6.2)
    ui.txLatDoubleSpinBox.setValue(50.8)
    ui.txHeightDoubleSpinBox.setValue(50.0)
    ui.rxLonDoubleSpinBox.setValue(6.88361)
    ui.rxLatDoubleSpinBox.setValue(50.52483)
    ui.rxHeightDoubleSpinBox.setValue(10.0)
    ui.mapSizeLonDoubleSpinBox.setValue(0.2)
    ui.mapSizeLatDoubleSpinBox.setValue(0.2)
    ui.mapResolutionDoubleSpinBox.setValue(3.0)


@remote_data(source='any')
@pytest.mark.usefixtures('srtm_handler')
def test_gui_startup_shows_pathgeometry(qtbot):
    # change download option to missing and test, if the results label
    # in geometry pane has correct values (need to wait for startup-timer
    # to fire)

    myapp = gui.PycrafGui()
    qtbot.addWidget(myapp)
    _set_parameters(myapp.ui)
    myapp.ui.srtmDownloadComboBox.setCurrentIndex(
        gui.SRTM_DOWNLOAD_MAPPING.index('missing')
        )
    with qtbot.waitSignal(
            myapp.my_geo_worker.result_ready[object, object],
            raising=False, timeout=50000,
            ):
        myapp.timer.start(10)

    ltxt = myapp.ui.ppRichTextLabel.text()
    assert re.sub("\s*", " ", ltxt) == re.sub("\s*", " ", LABEL_TEXT)


@remote_data(source='any')
@pytest.mark.usefixtures('srtm_handler')
def test_pp_worker(qtbot):
    # change download option to missing and test, if the results are correct

    myapp = gui.PycrafGui()
    qtbot.addWidget(myapp)
    _set_parameters(myapp.ui)
    myapp.ui.srtmDownloadComboBox.setCurrentIndex(
        gui.SRTM_DOWNLOAD_MAPPING.index('missing')
        )
    with qtbot.waitSignal(
            myapp.my_pp_worker.result_ready[object, object],
            raising=False, timeout=50000,
            ):
        myapp.on_pathprof_compute_pressed()

    res = myapp.pathprof_results

    assert_quantity_allclose(
        res['L_b'][::1000].to(cnv.dB).value,
        [0., 128.879956, 163.285657, 155.078537, 156.097511, 170.742422]
        )
    assert_quantity_allclose(
        res['eps_pt'][::1000].to(apu.deg).value,
        [0., 0.45954562, 0.51161058, 0.51158408, 0.51155677, 0.51152865]
        )
    assert_equal(res['path_type'][::1000], [0, 1, 1, 1, 1, 1])
    # assert myapp.pathprof_results is None


@remote_data(source='any')
@pytest.mark.usefixtures('srtm_handler')
def test_map_worker(qtbot):
    # change download option to missing and test, if the results are correct

    myapp = gui.PycrafGui()
    qtbot.addWidget(myapp)
    _set_parameters(myapp.ui)
    myapp.ui.srtmDownloadComboBox.setCurrentIndex(
        gui.SRTM_DOWNLOAD_MAPPING.index('missing')
        )
    with qtbot.waitSignal(
            myapp.my_map_worker.result_ready[object, object],
            raising=False, timeout=50000,
            ):
        myapp.on_map_compute_pressed()

    res = myapp.map_results

    assert_quantity_allclose(
        res['L_b'][::80, ::80].to(cnv.dB).value, [
            [147.83278909, 133.58127549, 111.43421362, 134.45908465],
            [111.45317964, 126.40950486, 127.38498799, 119.25246431],
            [119.00447374, 105.44468519, 126.87036123, 119.33221431],
            [113.64151601, 125.49854085, 116.10536532, 136.80592942],
            ])
    assert_quantity_allclose(
        res['eps_pt'][::80, ::80].to(apu.deg).value, [
            [0.27189932, 0.20469701, 0.65046079, 0.64865387],
            [-0.06627310, 0.03763396, 0.44081357, 0.30713528],
            [-0.37531096, -0.58974224, -0.46243938, -0.62455889],
            [-0.37766769, -0.49652843, -0.62897185, -0.46686728],
            ])
    assert_equal(res['path_type'][::80, ::80], [
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 1],
        ])
    # assert myapp.pathprof_results is None
