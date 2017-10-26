#!/usr/bin/python
# -*- coding: utf-8 -*-


from PyQt5 import QtCore
from functools import lru_cache
from astropy import units as u
from pycraf import pathprof
from pycraf import conversions as cnv


__all__ = ['PathProfWorker', 'MapWorker']


pathprof.SrtmConf.set(download='missing')


# Note: pathprof.height_path_data is super fast usually, so this
# is probably overkill
@lru_cache(maxsize=10, typed=False)
def cached_height_path_data(*args, **kwargs):
    return pathprof.height_path_data(*args, **kwargs)


@lru_cache(maxsize=10, typed=False)
def cached_height_map_data(*args, **kwargs):
    return pathprof.height_map_data(*args, **kwargs)


class PathProfWorker(QtCore.QObject):

    result_ready = QtCore.pyqtSignal(object, object, name='result_ready')
    job_started = QtCore.pyqtSignal(name='job_started')
    job_finished = QtCore.pyqtSignal(name='job_finished')
    job_excepted = QtCore.pyqtSignal(str, name='job_excepted')

    def __init__(self, parent):

        super().__init__()

        self.parent = parent
        self.job_waiting = False

        self.event_loop()

    @QtCore.pyqtSlot(dict)
    def on_job_triggered(self, job_dict):

        self.job_dict = job_dict.copy()
        self.job_waiting = True

    @QtCore.pyqtSlot()
    def event_loop(self):

        if self.job_waiting:
            self.do_job()

        # setup one-time counter
        self.timer = QtCore.QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.event_loop)
        self.timer.start(50)

    @QtCore.pyqtSlot()
    def do_job(self):

        self.job_waiting = False
        jdict = self.job_dict
        print('doing job', jdict)

        hprof_data = cached_height_path_data(
            jdict['tx_lon'] * u.deg, jdict['tx_lat'] * u.deg,
            jdict['rx_lon'] * u.deg, jdict['rx_lat'] * u.deg,
            jdict['stepsize'] * u.m,
            zone_t=jdict['tx_clutter'], zone_r=jdict['rx_clutter'],
            )

        results = pathprof.atten_path_fast(
            jdict['freq'] * u.GHz,
            jdict['temp'] * u.K, jdict['press'] * u.hPa,
            jdict['tx_height'] * u.m, jdict['rx_height'] * u.m,
            jdict['timepercent'] * u.percent,
            hprof_data,
            version=jdict['version'],
            polarization=jdict['polarization'],
            )

        self.result_ready.emit(hprof_data, results)

        print('job finished')


class MapWorker(QtCore.QObject):

    result_ready = QtCore.pyqtSignal(object, object, name='result_ready')
    job_started = QtCore.pyqtSignal(name='job_started')
    job_finished = QtCore.pyqtSignal(name='job_finished')
    job_excepted = QtCore.pyqtSignal(str, name='job_excepted')

    def __init__(self, parent):

        super().__init__()

        self.parent = parent
        self.job_waiting = False

        self.event_loop()

    @QtCore.pyqtSlot(dict)
    def on_job_triggered(self, job_dict):

        self.job_dict = job_dict.copy()
        self.job_waiting = True

    @QtCore.pyqtSlot()
    def event_loop(self):

        if self.job_waiting:
            self.do_job()

        # setup one-time counter
        self.timer = QtCore.QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.event_loop)
        self.timer.start(50)

    @QtCore.pyqtSlot()
    def do_job(self):

        self.job_waiting = False
        jdict = self.job_dict
        print('doing job', jdict)

        hprof_data = cached_height_map_data(
            jdict['tx_lon'] * u.deg, jdict['tx_lat'] * u.deg,
            jdict['size_lon'] * u.deg, jdict['size_lat'] * u.deg,
            jdict['map_reso'] * u.arcsec,
            zone_t=jdict['tx_clutter'], zone_r=jdict['rx_clutter'],
            )

        results = pathprof.atten_map_fast(
            jdict['freq'] * u.GHz,
            jdict['temp'] * u.K, jdict['press'] * u.hPa,
            jdict['tx_height'] * u.m, jdict['rx_height'] * u.m,
            jdict['timepercent'] * u.percent,
            hprof_data,
            version=jdict['version'],
            polarization=jdict['polarization'],
            )

        self.result_ready.emit(hprof_data, results)

        print('job finished')
