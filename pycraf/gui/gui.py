#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys
from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.ticker import ScalarFormatter
from .resources.main_form import Ui_MainWindow
from .plot_widget import CustomToolbar, PlotWidget
from .workers import PathProfWorker, MapWorker

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
    ('Geometry: Height', 'hprof_data', 'heights'),
    ('Geometry: Rx longitude', 'hprof_data', 'lons'),
    ('Geometry: Rx latitude', 'hprof_data', 'lats'),
    ('Geometry: Back-bearing', 'hprof_data', 'backbearings'),
    ('Geometry: Tx path horizon angle', 'results', 'eps_pt'),
    ('Geometry: Rx path horizon angle', 'results', 'eps_pr'),
    ('Geometry: Path type', 'results', 'path_type'),
    ('Attenuation: LoS', 'results', 'L_bfsg'),
    ('Attenuation: Diffraction', 'results', 'L_bd'),
    ('Attenuation: Troposcatter', 'results', 'L_bs'),
    ('Attenuation: Ducting/Anomalous', 'results', 'L_ba'),
    ('Attenuation: Total', 'results', 'L_b'),
    ('Attenuation: Total with clutter', 'results', 'L_b_corr'),
    ]

MAP_DISPLAY_OPTIONS = [
    # name in combobox, where to find ('hprof_data'/'results'), associated key
    ('Geometry: Height', 'hprof_data', 'heights'),
    ('Geometry: Back-bearing', 'hprof_data', 'backbearings'),
    ('Geometry: Tx path horizon angle', 'results', 'eps_pt'),
    ('Geometry: Rx path horizon angle', 'results', 'eps_pr'),
    ('Geometry: Path type', 'results', 'path_type'),
    ('Attenuation: LoS', 'results', 'L_bfsg'),
    ('Attenuation: Diffraction', 'results', 'L_bd'),
    ('Attenuation: Troposcatter', 'results', 'L_bs'),
    ('Attenuation: Ducting/Anomalous', 'results', 'L_ba'),
    ('Attenuation: Total', 'results', 'L_b'),
    ('Attenuation: Total with clutter', 'results', 'L_b_corr'),
    ]


class PycrafGui(QtWidgets.QMainWindow):

    pp_job_triggered = QtCore.pyqtSignal(dict, name='pp_job_triggered')
    map_job_triggered = QtCore.pyqtSignal(dict, name='map_job_triggered')

    def __init__(self, **kwargs):

        self.do_init = True
        super().__init__()

        self.pathprof_hprof_data = None
        self.pathprof_results = None
        self.map_hprof_data = None
        self.map_results = None

        self.setup_gui()

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
        _cb.setCurrentIndex(_cb.findText('Attenuation: Total with clutter'))

        # fill map result combo box
        names = [t[0] for t in MAP_DISPLAY_OPTIONS]
        _cb = self.ui.mapPlotChooserComboBox
        _cb.addItems(names)
        _cb.setCurrentIndex(_cb.findText('Attenuation: Total with clutter'))

        self.create_plotters()
        self.setup_signals()
        # self.setup_menu()
        # self.setup_actions()
        self.setup_workers()

    @QtCore.pyqtSlot(object)
    def setup_signals(self):

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

    @QtCore.pyqtSlot(object)
    def setup_workers(self):

        self.my_pp_worker_thread = QtCore.QThread()
        self.my_pp_worker = PathProfWorker(self)

        self.my_pp_worker.moveToThread(self.my_pp_worker_thread)
        # self.my_pp_worker.job_started.connect(self.busy_start)
        # self.my_pp_worker.job_finished.connect(self.busy_stop)
        # self.my_pp_worker.job_excepted.connect(self.busy_except)
        self.pp_job_triggered.connect(self.my_pp_worker.on_job_triggered)
        self.my_pp_worker.result_ready.connect(self.on_pathprof_result_ready)
        self.my_pp_worker_thread.start()

        self.my_map_worker_thread = QtCore.QThread()
        self.my_map_worker = MapWorker(self)

        self.my_map_worker.moveToThread(self.my_map_worker_thread)
        # self.my_map_worker.job_started.connect(self.busy_start)
        # self.my_map_worker.job_finished.connect(self.busy_stop)
        # self.my_map_worker.job_excepted.connect(self.busy_except)
        self.map_job_triggered.connect(self.my_map_worker.on_job_triggered)
        self.my_map_worker.result_ready.connect(self.on_map_result_ready)
        self.my_map_worker_thread.start()

    @QtCore.pyqtSlot()
    def create_plotters(self):

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
    def on_pathprof_compute_pressed(self):

        job_dict = self._get_parameters()
        self.pp_job_triggered.emit(job_dict)

    @QtCore.pyqtSlot()
    def on_map_compute_pressed(self):

        job_dict = self._get_parameters()
        job_dict['size_lon'] = self.ui.mapSizeLonDoubleSpinBox.value()
        job_dict['size_lat'] = self.ui.mapSizeLatDoubleSpinBox.value()
        job_dict['map_reso'] = self.ui.mapResolutionDoubleSpinBox.value()

        self.map_job_triggered.emit(job_dict)

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

    @QtCore.pyqtSlot(int)
    def plot_pathprof(self, display_index):

        if self.pathprof_hprof_data is None or self.pathprof_results is None:
            print('nothing to plot yet')
            return

        name, designation, dkey = PATH_DISPLAY_OPTIONS[display_index]
        print('plotting', name, designation, dkey)

        dists = self.pathprof_hprof_data['distances'][5:]
        y1 = self.pathprof_hprof_data['heights'][5:]
        y2 = self.pathprof_results['L_b_corr'][5:]

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
        ax2.set_ylabel('Path attenuation [dB]')
        # ax2.set_ylim((80, 270))

        plot_area.clear_history()
        plot_area.canvas.draw()

    @QtCore.pyqtSlot(int)
    def plot_map(self, display_index):

        if self.map_hprof_data is None or self.map_results is None:
            print('nothing to plot yet')
            return

        name, designation, dkey = MAP_DISPLAY_OPTIONS[display_index]
        print('plotting', name, designation, dkey)

        lons = self.map_hprof_data['xcoords']
        lats = self.map_hprof_data['ycoords']
        z = self.map_results['L_b_corr']

        plot_area = self.map_plot_area
        fig = plot_area.figure
        ax = plot_area.axes
        cax = plot_area.caxes
        ax.clear()

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
            vmin=-5, vmax=175,
            extent=(lons[0], lons[-1], lats[0], lats[-1]),
            )
        cbar = fig.colorbar(cim, cax=cax)
        ax.set_aspect(abs(lons[-1] - lons[0]) / abs(lats[-1] - lats[0]))
        cbar.set_label(r'Path propagation loss')

        plot_area.clear_history()
        plot_area.canvas.draw()


def start_gui():

    app = QtWidgets.QApplication(sys.argv)
    myapp = PycrafGui()
    myapp.show()
    sys.exit(app.exec_())
