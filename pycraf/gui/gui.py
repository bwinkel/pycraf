#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys
from PyQt5 import QtCore, QtGui, QtWidgets
import numpy as np
from astropy import units as u
# from scipy.interpolate import interp1d
from matplotlib.ticker import ScalarFormatter
from .resources.main_form import Ui_MainWindow
from .plot_widget import CustomToolbar, PlotWidget
from .helpers import setup_earth_axes
from .workers import (
    GeometryWorker, StatisticsWorker, PathProfWorker, MapWorker
    )
from ..geometry import true_angular_distance
from ..pathprof import SrtmConf

__all__ = ['PycrafGui', 'start_gui']

CLUTTER_DESCRIPTIONS = [
    # name in combobox, clutter-enum int
    ('No clutter', -1),
    ('Sparse', 0),
    ('Village', 1),
    ('Decidious trees', 2),
    ('Coniferous trees', 3),
    ('Tropical forest', 4),
    ('Suburban', 5),
    ('Dense suburban', 6),
    ('Urban', 7),
    ('Dense urban', 8),
    ('High urban', 9),
    ('Industrial zone', 10),
    ]

PATH_DISPLAY_OPTIONS = [
    # name in combobox, where to find ('hprof_data'/'results'), associated key
    ('Geometry: Height', 'hprof_data', 'heights', 'm'),
    ('Geometry: Rx longitude', 'hprof_data', 'lons', 'deg'),
    ('Geometry: Rx latitude', 'hprof_data', 'lats', 'deg'),
    ('Geometry: Back-bearing', 'hprof_data', 'backbearings', 'deg'),
    ('Geometry: Tx path horizon angle', 'results', 'eps_pt', 'deg'),
    ('Geometry: Rx path horizon angle', 'results', 'eps_pr', 'deg'),
    ('Geometry: Path type', 'results', 'path_type', '0: LoS, 1: non-LoS'),
    ('Attenuation: LoS loss', 'results', 'L_b0p', 'dB'),
    ('Attenuation: Diffraction loss', 'results', 'L_bd', 'dB'),
    ('Attenuation: Troposcatter loss', 'results', 'L_bs', 'dB'),
    ('Attenuation: Ducting/Anomalous loss', 'results', 'L_ba', 'dB'),
    ('Attenuation: Total loss', 'results', 'L_b', 'dB'),
    ('Attenuation: Total loss with clutter', 'results', 'L_b_corr', 'dB'),
    ]

MAP_DISPLAY_OPTIONS = [
    # name in combobox, where to find ('hprof_data'/'results'), associated key
    ('Geometry: Height', 'hprof_data', 'height_map', 'm'),
    ('Geometry: Distance', 'hprof_data', 'dist_map', 'km'),
    ('Geometry: Bearing', 'hprof_data', 'bearing_map', 'deg'),
    ('Geometry: Back-bearing', 'hprof_data', 'back_bearing_map', 'deg'),
    ('Geometry: Tx path horizon angle', 'results', 'eps_pt', 'deg'),
    ('Geometry: Rx path horizon angle', 'results', 'eps_pr', 'deg'),
    ('Geometry: Path type', 'results', 'path_type', '0: LoS, 1: non-LoS'),
    ('Attenuation: LoS loss', 'results', 'L_b0p', 'dB'),
    ('Attenuation: Diffraction loss', 'results', 'L_bd', 'dB'),
    ('Attenuation: Troposcatter loss', 'results', 'L_bs', 'dB'),
    ('Attenuation: Ducting/Anomalous loss', 'results', 'L_ba', 'dB'),
    ('Attenuation: Total loss', 'results', 'L_b', 'dB'),
    ('Attenuation: Total loss with clutter', 'results', 'L_b_corr', 'dB'),
    ]

SRTM_DOWNLOAD_MAPPING = ['never', 'missing', 'always']
SRTM_SERVER_MAPPING = ['nasa_v1.0', 'nasa_v2.1', 'viewpano']

PP_TEXT_TEMPLATE = '''
<style>
    table {{
        color: black;
        width: 100%;
        text-align: center;
        font-family: "Futura-Light", sans-serif;
        font-weight: 400;
        font-size: 14px;
    }}
    th {{
        color: blue;
        font-size: 16px;
    }}
    th, td {{ padding: 2px; }}
    thead.th {{
        height: 110%;
        border-bottom: solid 0.25em black;
    }}
    .lalign {{ text-align: left; padding-left: 12px;}}
    .ralign {{ text-align: right; padding-right: 12px; }}
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
      <td class="ralign">{a_e_50:5.0f}</td>
      <td class="lalign">alpha_tr</td>
      <td class="ralign">{alpha_tr:8.3f}</td>
      <td class="lalign">L_b0p (LoS)</td>
      <td class="ralign">{L_b0p:5.1f}</td>
    </tr>
    <tr>
      <td class="lalign">a_e (beta0)</td>
      <td class="ralign">{a_e_b0:5.0f}</td>
      <td class="lalign">alpha_rt</td>
      <td class="ralign">{alpha_rt:8.3f}</td>
      <td class="lalign">L_bd (Diffraction)</td>
      <td class="ralign">{L_bd:5.1f}</td>
    </tr>
    <tr>
      <td class="lalign">beta0</td>
      <td class="ralign">{beta0:.2f}</td>
      <td class="lalign">eps_pt</td>
      <td class="ralign">{eps_pt:8.3f}</td>
      <td class="lalign">L_bs (Troposcatter)</td>
      <td class="ralign">{L_bs:5.1f}</td>
    </tr>
    <tr>
      <td class="lalign">N0</td>
      <td class="ralign">{N0:.1f}</td>
      <td class="lalign">eps_pr</td>
      <td class="ralign">{eps_pr:8.3f}</td>
      <td class="lalign">L_ba (Anomalous)</td>
      <td class="ralign">{L_ba:5.1f}</td>
    </tr>
    <tr>
      <td class="lalign">Delta N</td>
      <td class="ralign">{delta_N:.2f}</td>
      <td class="lalign">h_eff</td>
      <td class="ralign">{h_eff_50:8.1f}</td>
      <td class="lalign">L_b (Total)</td><td class="ralign">{L_b:5.1f}</td>
    </tr>
    <tr>
      <td class="lalign"></td>
      <td class="ralign"></td>
      <td class="lalign">Path type</td>
      <td class="ralign">{path_type_str:s}</td>
      <td class="lalign" style="color: blue;">L_b_corr (Total + Clutter)</td>
      <td class="ralign" style="color: blue;">{L_b_corr:5.1f}</td>
    </tr>
  </tbody>
</table>
'''


SrtmConf.set(download='never', server='viewpano')


class PycrafGui(QtWidgets.QMainWindow):

    geo_job_triggered = QtCore.pyqtSignal(dict, name='geo_job_triggered')
    stats_job_triggered = QtCore.pyqtSignal(dict, name='stats_job_triggered')
    pp_job_triggered = QtCore.pyqtSignal(dict, name='pp_job_triggered')
    map_job_triggered = QtCore.pyqtSignal(dict, name='map_job_triggered')
    clear_caches_triggered = QtCore.pyqtSignal(name='clear_caches_triggered')

    def __init__(self, **kwargs):

        self.do_init = True
        super().__init__()

        self.geometry_hprof_data = None
        self.geometry_results = None
        self.statistics_f_p = None
        self.statistics_results = None
        self.pathprof_hprof_data = None
        self.pathprof_results = None
        self.map_hprof_data = None
        self.map_results = None

        self.setup_gui()

        # want that at start, user is presented with a plot
        # we will also use this timer to implement a short delay between
        # gui parameter changes and calling the plotting routine; this is
        # because Qt will otherwise go through all intermediate steps
        # (e.g., if one changes multiple digits in a spinbox), which
        # is usually undesired
        self.timer = QtCore.QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.on_any_param_changed)
        self.timer.start(10)

    @QtCore.pyqtSlot()
    def setup_gui(self):

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # fill clutter combo boxes
        names = [t[0] for t in CLUTTER_DESCRIPTIONS]
        self.ui.txClutterComboBox.addItems(names)
        self.ui.rxClutterComboBox.addItems(names)

        # fill path result combo box
        names = [t[0] for t in PATH_DISPLAY_OPTIONS]
        _cb = self.ui.pathprofPlotChooserComboBox
        _cb.addItems(names)
        _cb.setCurrentIndex(_cb.findText(
            'Attenuation: Total loss with clutter'
            ))

        # fill map result combo box
        names = [t[0] for t in MAP_DISPLAY_OPTIONS]
        _cb = self.ui.mapPlotChooserComboBox
        _cb.addItems(names)
        _cb.setCurrentIndex(_cb.findText(
            'Attenuation: Total loss with clutter'
            ))

        self.ui.srtmPathPushButton.setText(
            os.path.abspath(SrtmConf.srtm_dir)
            )
        self.ui.srtmDownloadComboBox.setCurrentIndex(
            SRTM_DOWNLOAD_MAPPING.index(SrtmConf.download)
            )
        self.ui.srtmServerComboBox.setCurrentIndex(
            SRTM_SERVER_MAPPING.index(SrtmConf.server)
            )

        self.create_plotters()
        self.setup_signals()
        # self.setup_menu()
        # self.setup_actions()
        self.setup_workers()

        # setup busy indicator gif
        self.busy_movie = QtGui.QMovie(":gifs/busy_indicator.gif")
        self.ui.pathprofBusyIndicatorLabel.clear()
        self.ui.mapBusyIndicatorLabel.clear()

    @QtCore.pyqtSlot(object)
    def setup_signals(self):

        self.ui.srtmPathPushButton.pressed.connect(
            self.on_options_srtmpath_pressed
            )
        self.ui.srtmDownloadComboBox.currentIndexChanged[int].connect(
            self.on_options_srtmdownload_changed
            )
        self.ui.srtmDownloadComboBox.currentIndexChanged[int].connect(
            self.on_options_srtmserver_changed
            )

        self.ui.pathprofComputePushButton.pressed.connect(
            self.on_pathprof_compute_pressed
            )
        self.ui.pathprofPlotChooserComboBox.currentIndexChanged[int].connect(
            self.plot_pathprof
            )

        self.ui.mapComputePushButton.pressed.connect(
            self.on_map_compute_pressed
            )
        self.ui.mapPlotChooserComboBox.currentIndexChanged[int].connect(
            self.plot_map
            )

        for w in [
                self.ui.freqDoubleSpinBox,
                self.ui.timepercentDoubleSpinBox,
                self.ui.stepsizeDoubleSpinBox,
                self.ui.tempDoubleSpinBox,
                self.ui.pressDoubleSpinBox,
                self.ui.txLonDoubleSpinBox,
                self.ui.txLatDoubleSpinBox,
                self.ui.txHeightDoubleSpinBox,
                self.ui.rxLonDoubleSpinBox,
                self.ui.rxLatDoubleSpinBox,
                self.ui.rxHeightDoubleSpinBox,
                ]:
            w.valueChanged.connect(self.on_any_param_changed_initiated)

        for w in [
                self.ui.versionComboBox,
                self.ui.txClutterComboBox,
                self.ui.rxClutterComboBox,
                ]:
            w.currentIndexChanged.connect(self.on_any_param_changed_initiated)

    @QtCore.pyqtSlot(object)
    def setup_workers(self):

        self.my_geo_worker_thread = QtCore.QThread()
        self.my_geo_worker = GeometryWorker(self)

        self.my_geo_worker.moveToThread(self.my_geo_worker_thread)
        self.geo_job_triggered.connect(self.my_geo_worker.on_job_triggered)
        self.clear_caches_triggered.connect(
            self.my_geo_worker.on_clear_caches_triggered
            )
        self.my_geo_worker.result_ready[object, object].connect(
            self.on_geometry_result_ready
            )
        self.my_geo_worker.job_excepted[str].connect(self.on_job_excepted)
        self.my_geo_worker_thread.start()

        self.my_stats_worker_thread = QtCore.QThread()
        self.my_stats_worker = StatisticsWorker(self)

        self.my_stats_worker.moveToThread(self.my_stats_worker_thread)
        self.stats_job_triggered.connect(
            self.my_stats_worker.on_job_triggered
            )
        self.clear_caches_triggered.connect(
            self.my_stats_worker.on_clear_caches_triggered
            )
        self.my_stats_worker.result_ready[object, object].connect(
            self.on_statistics_result_ready
            )
        self.my_stats_worker.job_excepted[str].connect(self.on_job_excepted)
        self.my_stats_worker_thread.start()

        self.my_pp_worker_thread = QtCore.QThread()
        self.my_pp_worker = PathProfWorker(self)

        self.my_pp_worker.moveToThread(self.my_pp_worker_thread)
        self.my_pp_worker.job_started.connect(self.busy_start_pp)
        self.my_pp_worker.job_finished.connect(self.busy_stop_pp)
        self.my_pp_worker.job_excepted.connect(self.busy_stop_pp)
        self.pp_job_triggered.connect(self.my_pp_worker.on_job_triggered)
        self.clear_caches_triggered.connect(
            self.my_pp_worker.on_clear_caches_triggered
            )
        self.my_pp_worker.result_ready[object, object].connect(
            self.on_pathprof_result_ready
            )
        self.my_pp_worker.job_excepted[str].connect(self.on_job_excepted)
        self.my_pp_worker_thread.start()

        self.my_map_worker_thread = QtCore.QThread()
        self.my_map_worker = MapWorker(self)

        self.my_map_worker.moveToThread(self.my_map_worker_thread)
        self.my_map_worker.job_started.connect(self.busy_start_map)
        self.my_map_worker.job_finished.connect(self.busy_stop_map)
        self.my_map_worker.job_excepted.connect(self.busy_stop_map)
        self.map_job_triggered.connect(self.my_map_worker.on_job_triggered)
        self.clear_caches_triggered.connect(
            self.my_map_worker.on_clear_caches_triggered
            )
        self.my_map_worker.result_ready[object, object].connect(
            self.on_map_result_ready
            )
        self.my_map_worker.job_excepted[str].connect(self.on_job_excepted)
        self.my_map_worker_thread.start()

    @QtCore.pyqtSlot()
    def create_plotters(self):

        self.geometry_plot_area = PlotWidget(
            subplotx=1, subploty=1, plottername='Geometry',
            )
        self.ui.geometryVerticalLayout.addWidget(self.geometry_plot_area)

        self.statistics_plot_area = PlotWidget(
            subplotx=1, subploty=1, plottername='Statistics',
            )
        self.ui.statisticsVerticalLayout.addWidget(self.statistics_plot_area)

        self.pathprof_plot_area = PlotWidget(
            subplotx=1, subploty=2,
            sharex=True,
            plottername='Path profile',
            )
        self.ui.pathprofVerticalLayout.addWidget(self.pathprof_plot_area)

        self.map_plot_area = PlotWidget(
            subplotx=1, subploty=1, do_cbars=True, plottername='Map',
            )
        self.ui.mapVerticalLayout.addWidget(self.map_plot_area)

    @QtCore.pyqtSlot(object)
    def closeEvent(self, event):

        self.write_settings()
        super().closeEvent(event)

    @QtCore.pyqtSlot(object)
    def showEvent(self, se):
        '''
        it is necessary to perform "readSettings" after all of the GUI elements
        were processed and the first showevent occurs
        otherwise not all settings will be processed correctly
        '''

        super().showEvent(se)
        if self.do_init:
            self.read_settings()
            self.do_init = False

    @QtCore.pyqtSlot(str)
    def read_settings(self):
        '''
        Read stored settings (including layout and window geometry).
        '''

        settings = QtCore.QSettings('pycraf', 'gui-layout')

        geometry = settings.value('mainform/geometry')
        windowstate = settings.value('mainform/windowState')

        if geometry is not None:
            self.restoreGeometry(geometry)
        if windowstate is not None:
            self.restoreState(windowstate)

    @QtCore.pyqtSlot(str)
    def write_settings(self):
        '''
        Store settings (including layout and window geometry).
        '''

        settings = QtCore.QSettings('pycraf', 'gui-layout')

        settings.setValue('mainform/geometry', self.saveGeometry())
        settings.setValue('mainform/windowState', self.saveState())

    def _get_parameters(self):

        job_dict = {}
        job_dict['freq'] = self.ui.freqDoubleSpinBox.value()
        job_dict['timepercent'] = self.ui.timepercentDoubleSpinBox.value()
        job_dict['stepsize'] = self.ui.stepsizeDoubleSpinBox.value()
        job_dict['temp'] = self.ui.tempDoubleSpinBox.value()
        job_dict['press'] = self.ui.pressDoubleSpinBox.value()
        job_dict['version'] = int(self.ui.versionComboBox.currentText())
        job_dict['polarization'] = self.ui.versionComboBox.currentIndex()

        job_dict['tx_lon'] = self.ui.txLonDoubleSpinBox.value()
        job_dict['tx_lat'] = self.ui.txLatDoubleSpinBox.value()
        job_dict['tx_height'] = self.ui.txHeightDoubleSpinBox.value()
        job_dict['tx_clutter'] = self.ui.txClutterComboBox.currentIndex()
        job_dict['rx_lon'] = self.ui.rxLonDoubleSpinBox.value()
        job_dict['rx_lat'] = self.ui.rxLatDoubleSpinBox.value()
        job_dict['rx_height'] = self.ui.rxHeightDoubleSpinBox.value()
        job_dict['rx_clutter'] = self.ui.rxClutterComboBox.currentIndex()

        return job_dict

    @QtCore.pyqtSlot()
    def on_options_srtmpath_pressed(self):

        srtm_dir = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            'Choose directory with SRTM tile data (.hgt files)',
            SrtmConf.srtm_dir,
            QtWidgets.QFileDialog.ShowDirsOnly |
            QtWidgets.QFileDialog.DontResolveSymlinks,
            )

        if srtm_dir:
            SrtmConf.set(srtm_dir=srtm_dir)
            self.ui.srtmPathPushButton.setText(
                os.path.abspath(SrtmConf.srtm_dir)
                )
            self.clear_caches_triggered.emit()

    @QtCore.pyqtSlot(int)
    def on_options_srtmdownload_changed(self, idx):

        SrtmConf.set(download=SRTM_DOWNLOAD_MAPPING[idx])
        self.clear_caches_triggered.emit()
        print(SrtmConf)

    @QtCore.pyqtSlot(int)
    def on_options_srtmserver_changed(self, idx):

        SrtmConf.set(server=SRTM_SERVER_MAPPING[idx])
        self.clear_caches_triggered.emit()
        print(SrtmConf)

    @QtCore.pyqtSlot(str)
    def on_job_excepted(self, error_msg):
        print('Job error: ', error_msg)
        QtWidgets.QMessageBox.critical(self, 'Job error', error_msg)

    @QtCore.pyqtSlot()
    def on_any_param_changed_initiated(self):
        self.timer.start(250)

    @QtCore.pyqtSlot()
    def on_any_param_changed(self):

        job_dict = self._get_parameters()
        self.geo_job_triggered.emit(job_dict)
        self.stats_job_triggered.emit(job_dict)

        if self.ui.pathprofAutoUpdateCheckBox.isChecked():
            self.on_pathprof_compute_pressed()

    @QtCore.pyqtSlot()
    def busy_start_pp(self):
        self.ui.pathprofComputePushButton.setEnabled(False)
        self.ui.pathprofBusyIndicatorLabel.setMovie(self.busy_movie)
        print('movie start')
        self.busy_movie.start()

    @QtCore.pyqtSlot()
    def busy_stop_pp(self):
        self.busy_movie.stop()
        self.ui.pathprofBusyIndicatorLabel.clear()
        self.ui.pathprofComputePushButton.setEnabled(True)

    @QtCore.pyqtSlot()
    def busy_start_map(self):
        self.ui.mapComputePushButton.setEnabled(False)
        self.ui.mapBusyIndicatorLabel.setMovie(self.busy_movie)
        self.busy_movie.start()

    @QtCore.pyqtSlot()
    def busy_stop_map(self):
        self.busy_movie.stop()
        self.ui.mapBusyIndicatorLabel.clear()
        self.ui.mapComputePushButton.setEnabled(True)

    @QtCore.pyqtSlot()
    def on_pathprof_compute_pressed(self):

        job_dict = self._get_parameters()
        job_dict['do_generic'] = (
            not self.ui.pathprofIncludeHeightCheckBox.isChecked()
            )
        # self.geo_job_triggered.emit(job_dict)
        self.pp_job_triggered.emit(job_dict)

    @QtCore.pyqtSlot()
    def on_map_compute_pressed(self):

        job_dict = self._get_parameters()
        job_dict['size_lon'] = self.ui.mapSizeLonDoubleSpinBox.value()
        job_dict['size_lat'] = self.ui.mapSizeLatDoubleSpinBox.value()
        job_dict['map_reso'] = self.ui.mapResolutionDoubleSpinBox.value()

        self.map_job_triggered.emit(job_dict)

    @QtCore.pyqtSlot(object, object)
    def on_geometry_result_ready(self, hprof_data, results):

        self.geometry_hprof_data = hprof_data
        self.geometry_results = results

        self.plot_geometry()

    @QtCore.pyqtSlot(object, object)
    def on_statistics_result_ready(self, f_p_data, results):

        self.statistics_f_p = f_p_data
        self.statistics_results = results

        self.plot_statistics()

    @QtCore.pyqtSlot(object, object)
    def on_pathprof_result_ready(self, hprof_data, results):

        self.pathprof_hprof_data = hprof_data
        self.pathprof_results = results

        display_index = self.ui.pathprofPlotChooserComboBox.currentIndex()
        self.plot_pathprof(display_index)

    @QtCore.pyqtSlot(object, object)
    def on_map_result_ready(self, hprof_data, results):

        self.map_hprof_data = hprof_data
        self.map_results = results

        display_index = self.ui.mapPlotChooserComboBox.currentIndex()
        self.plot_map(display_index)

    @QtCore.pyqtSlot()
    def plot_geometry(self):

        if self.geometry_hprof_data is None or self.geometry_results is None:
            print('nothing to plot yet')
            return

        hprof = self.geometry_hprof_data
        results = self.geometry_results

        lons, lats, distance, distances, heights, *_ = hprof

        lon_rx, lat_rx = results['lon_r'], results['lat_r']
        delta = true_angular_distance(lon_rx, lat_rx, lons, lats)

        # strip units for calculations below (leading underscore -> no unit)
        _distances = distances.to(u.km).value
        _heights = heights.to(u.m).value
        _delta = delta.to(u.rad).value
        _dist = results['distance'].to(u.km).value
        _h_ts = results['h_ts'].to(u.m).value
        _h_rs = results['h_rs'].to(u.m).value
        _h_tg = results['h_tg'].to(u.m).value
        _h_rg = results['h_rg'].to(u.m).value
        _a_e_m = results['a_e_50'].to(u.m).value
        _a_e_km = results['a_e_50'].to(u.km).value
        _S_tim = results['S_tim_50'].to(1).value
        _S_rim = results['S_rim_50'].to(1).value
        _S_tr = results['S_tr_50'].to(1).value
        _d_bp = results['d_bp_50'].to(u.km).value
        _h_bp = results['h_bp_50'].to(u.m).value
        _h_eff = results['h_eff_50'].to(u.m).value
        _h_ke = _h_bp - abs(_h_eff)
        print('d_bp, h_bp, h_ke', _d_bp, _h_bp, _h_ke)

        theta_scale = (
            (_delta[-1] - _delta[0]) / (_distances[-1] - _distances[0])
            )
        theta_lim = _distances[0], _distances[-1]
        h_lim = (
            min([_heights.min(), _h_ke]) - 5,
            max([_heights.max(), _h_ts, _h_rs, _h_bp]) + 5
            )

        plot_area = self.geometry_plot_area
        fig = plot_area.figure
        axes = plot_area.axes
        try:
            for ax in axes:
                ax.clear()
            print('len(axes)', len(axes))
        except TypeError:
            axes.clear()
            fig.clear()

        plot_area._axes = ax, aux_ax = setup_earth_axes(
            fig, 111, theta_lim, h_lim, _a_e_m, theta_scale
            )

        # plot height profile
        aux_ax.plot(_distances, _heights, 'k-')
        # plot bullington point
        # aux_ax.plot(_d_bp, _h_bp, 'o', color='orange')
        # foot point of the knife edge
        # aux_ax.plot(_d_bp, _h_ke, 'o', color='orange')
        aux_ax.plot([_d_bp, _d_bp], [_h_ke, _h_bp], '-', color='orange', lw=2)
        # plot Tx/Rx "heights"
        aux_ax.plot(
            [0, 0], [_heights[0], _heights[0] + _h_tg],
            'b-', lw=3
            )
        aux_ax.plot(
            [_dist, _dist], [_heights[-1], _heights[-1] + _h_rg],
            'r-', lw=3
            )

        # plot the paths to clarify the Bullington geometry
        # need to interpolate paths (mpl.plot does straight lines in
        # viewport not in physical space)
        # Note: the paths appear curvy in the plot, but his is because
        # the two axes are scaled so differently; we like to keep it like
        # this - although the curvature is exaggerated, the heights in the
        # plots are correct (i.e., the heights of Earth's surface as a
        # function of the distance from Tx)

        def _calc_path(d0, h0, S, steps, direction=1):

            assert direction in [-1, 1]

            eps0 = d0 / _a_e_km
            r0 = _a_e_km + h0 / 1000
            x0 = r0 * np.sin(eps0)
            y0 = r0 * np.cos(eps0)

            xs = x0 + direction * steps
            ys = y0 + steps * (S - _dist / 2 / _a_e_km + eps0)
            ds = np.arctan2(xs, ys) * _a_e_km
            hs = np.sqrt(xs ** 2 + ys ** 2) - _a_e_km

            return ds, hs

        steps = np.linspace(0, 2 * _dist, 51)
        if results['path_type'] == 1:
            _ds, _hs = _calc_path(0, _h_ts, _S_tim, steps)
            aux_ax.plot(_ds, _hs * 1000, 'b--')
            _ds, _hs = _calc_path(_dist, _h_rs, _S_rim, steps, direction=-1)
            aux_ax.plot(_ds, _hs * 1000, 'r--')

        _ds, _hs = _calc_path(0, _h_ts, _S_tr, steps)
        aux_ax.plot(_ds, _hs * 1000, 'g--')

        ax.grid(color='0.5')
        ax.set_aspect('auto')

        plot_area.clear_history()
        plot_area.canvas.draw()

        print(results)
        results['path_type_str'] = ['LoS', 'Trans-horizon'][
            int(results['path_type'])
            ]

        self.ui.ppRichTextLabel.setText(PP_TEXT_TEMPLATE.format(**results))

    @QtCore.pyqtSlot()
    def plot_statistics(self):

        if self.statistics_f_p is None or self.statistics_results is None:
            print('nothing to plot yet')
            return

        freqs, timepercents = self.statistics_f_p
        L_b_corr = self.statistics_results['L_b_corr']

        plot_area = self.statistics_plot_area
        ax = plot_area.axes

        ax.clear()

        for sax in [ax.xaxis, ax.yaxis]:
            sax.set_major_formatter(ScalarFormatter(useOffset=False))

        ax.set_xlabel('Time percent [%]')
        ax.grid()
        ax.set_xlim((timepercents[0], timepercents[-1]))

        t = timepercents.squeeze()
        lidx = np.argmin(np.abs(t - 2e-3))
        for idx, f in enumerate(freqs.squeeze()):
            p = ax.semilogx(
                t, L_b_corr.value[idx], '-', label='{:5.1f}'.format(f)
                )
            ax.text(
                2e-3, L_b_corr.value[idx][lidx], '{:.1f} GHz'.format(f),
                ha='left', va='top', color=p[0].get_color(),
                )

        ax.set_ylabel('L_b_corr [dB]')
        # ax.legend(
        #     *ax.get_legend_handles_labels(),
        #     title='Frequency [GHz]', ncol=2,
        #     )

        plot_area.clear_history()
        plot_area.canvas.draw()

    @QtCore.pyqtSlot(int)
    def plot_pathprof(self, display_index):

        if self.pathprof_hprof_data is None or self.pathprof_results is None:
            print('nothing to plot yet')
            return

        name, designation, dkey, unit = PATH_DISPLAY_OPTIONS[display_index]
        print('plotting', name, designation, dkey, unit)

        dists = self.pathprof_hprof_data['distances'][5:]
        y1 = self.pathprof_hprof_data['heights'][5:]
        y2 = getattr(self, 'pathprof_{:s}'.format(designation))[dkey][5:]

        plot_area = self.pathprof_plot_area
        axes = plot_area.axes
        ax1, ax2 = axes

        for ax in [ax1, ax2]:
            ax.clear()

            for sax in [ax.xaxis, ax.yaxis]:
                sax.set_major_formatter(ScalarFormatter(useOffset=False))

            ax.set_xlabel('Distance [km]')
            ax.grid()
            ax.set_xlim((dists[0], dists[-1]))

        ax1.plot(dists, y1, 'k-')
        ax2.plot(dists, y2, 'k-')

        ax1.set_ylabel('Heights [m]')
        # ax2.set_ylabel('Path attenuation [dB]')
        ax2.set_ylabel('{:s} [{:s}]'.format(name.split(': ')[1], unit))
        # ax2.set_ylim((80, 270))

        plot_area.clear_history()
        plot_area.canvas.draw()

    @QtCore.pyqtSlot(int)
    def plot_map(self, display_index):

        if self.map_hprof_data is None or self.map_results is None:
            print('nothing to plot yet')
            return

        name, designation, dkey, unit = MAP_DISPLAY_OPTIONS[display_index]
        print('plotting', name, designation, dkey, unit)

        lons = self.map_hprof_data['xcoords']
        lats = self.map_hprof_data['ycoords']
        # z = self.map_results['L_b_corr']
        z = getattr(self, 'map_{:s}'.format(designation))[dkey][5:]
        try:
            z = z.value
        except AttributeError:
            pass

        plot_area = self.map_plot_area
        fig = plot_area.figure
        ax = plot_area.axes
        cax = plot_area.caxes
        ax.clear()
        cax.clear()

        for sax in [ax.xaxis, ax.yaxis]:
            sax.set_major_formatter(ScalarFormatter(useOffset=False))

        ax.set_xlabel('Longitude [deg]')
        ax.set_ylabel('Latitude [deg]')
        ax.set_xlim((lons[0], lons[-1]))
        ax.set_ylim((lats[0], lats[-1]))
        ax.xaxis.tick_top()
        ax.xaxis.set_label_position('top')

        cim = ax.imshow(
            z,
            origin='lower', interpolation='nearest', cmap='inferno_r',
            # vmin=-5, vmax=175,
            extent=(lons[0], lons[-1], lats[0], lats[-1]),
            )
        cbar = fig.colorbar(cim, cax=cax)
        ax.set_aspect(abs(lons[-1] - lons[0]) / abs(lats[-1] - lats[0]))
        cbar.set_label(r'{:s} [{:s}]'.format(name.split(': ')[1], unit))

        plot_area.clear_history()
        plot_area.canvas.draw()

    def close(self):
        '''
        This is a must to prevent the error:

        Qt has caught an exception thrown from an event handler. Throwing
        exceptions from an event handler is not supported in Qt. You must not
        let any exception whatsoever propagate through Qt code. If that is
        not possible, in Qt 5 you must at least reimplement
        QCoreApplication::notify() and catch all exceptions there.

        Furthermore, one has to:

            app.aboutToQuit.connect(my_mw.close)

        (see below)
        '''
        self.my_geo_worker_thread.quit()
        self.my_stats_worker_thread.quit()
        self.my_pp_worker_thread.quit()
        self.my_map_worker_thread.quit()
        self.my_geo_worker_thread.wait()
        self.my_stats_worker_thread.wait()
        self.my_pp_worker_thread.wait()
        self.my_map_worker_thread.wait()

        super().close()


def start_gui():

    app = QtWidgets.QApplication(sys.argv)
    my_mw = PycrafGui()
    my_mw.show()

    # this is necessary for everything other than file-menu -> quit
    app.aboutToQuit.connect(my_mw.close)

    sys.exit(app.exec_())
