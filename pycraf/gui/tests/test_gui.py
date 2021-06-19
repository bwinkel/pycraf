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
@pytest.mark.do_gui_tests
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
@pytest.mark.do_gui_tests
@pytest.mark.usefixtures('srtm_handler')
def test_stats_worker(qtbot):
    # change download option to missing and test, if the results are correct

    myapp = gui.PycrafGui()
    qtbot.addWidget(myapp)
    _set_parameters(myapp.ui)
    myapp.ui.srtmDownloadComboBox.setCurrentIndex(
        gui.SRTM_DOWNLOAD_MAPPING.index('missing')
        )
    with qtbot.waitSignal(
            myapp.my_stats_worker.result_ready[object, object],
            raising=False, timeout=50000,
            ):
        myapp.timer.start(10)

    res = myapp.statistics_results

    assert_quantity_allclose(
        res['L_b'][:, ::20].to(cnv.dB).value, [
            [138.8771118, 140.8853131, 142.8934803, 144.9016341, 147.7434509],
            [156.2189124, 158.2270703, 160.2352217, 162.2433706, 164.7989202],
            [165.3765899, 167.3847525, 169.3929057, 171.4010551, 173.6622298],
            [174.5007580, 176.5089330, 178.5170906, 180.5252413, 182.7685164],
            [186.5540705, 188.5623023, 190.5704806, 192.5786379, 194.8213818],
            [195.9055898, 197.9140057, 199.9222511, 201.9304294, 204.1729092],
            [210.4921391, 212.5046189, 214.5143471, 216.5229899, 218.7653956],
            [236.8190545, 238.8738936, 240.8993047, 242.9128866, 245.1563437],
            [243.0043903, 247.2872525, 251.7308653, 256.0069129, 259.5560927],
            ])
    # assert myapp.pathprof_results is None


@remote_data(source='any')
@pytest.mark.do_gui_tests
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
@pytest.mark.do_gui_tests
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

    print(res['L_b'][::80, ::80].to(cnv.dB).value)
    print(res['eps_pt'][::80, ::80].to(apu.deg).value)
    assert_quantity_allclose(
        res['L_b'][::80, ::80].to(cnv.dB).value, [
            [147.92110506, 133.64246369, 111.43421362, 134.20808414],
            [111.45317964, 126.50588342, 127.41079286, 119.47281152],
            [118.95606625, 105.44468519, 126.89590692, 119.33537173],
            [113.64151601, 125.3059271, 116.06473428, 136.72188462],
            ])
    assert_quantity_allclose(
        res['eps_pt'][::80, ::80].to(apu.deg).value, [
            [0.27265693, 0.20522323, 0.6505462, 0.64897268],
            [-0.06723739, 0.03969113, 0.44085012, 0.30607933],
            [-0.37428988, -0.58972059, -0.46088839, -0.62473188],
            [-0.37882374, -0.49772378, -0.62911416, -0.46750982],
            ])
    assert_equal(res['path_type'][::80, ::80], [
        [1, 1, 0, 1],
        [0, 1, 1, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 1],
        ])
    # assert myapp.pathprof_results is None
