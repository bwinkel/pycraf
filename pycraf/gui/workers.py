#!/usr/bin/python
# -*- coding: utf-8 -*-


from PyQt5 import QtCore
from functools import lru_cache
import numpy as np
from astropy import units as u
from .. import pathprof
# from .. import conversions as cnv


__all__ = [
    'GeometryWorker', 'StatisticsWorker', 'PathProfWorker', 'MapWorker'
    ]


# Note: pathprof.height_path_data is super fast usually, so this
# is probably overkill
@lru_cache(maxsize=10, typed=False)
def cached_height_path_data(*args, **kwargs):
    return pathprof.height_path_data(*args, **kwargs)


@lru_cache(maxsize=10, typed=False)
def cached_height_path_data_generic(*args, **kwargs):
    # height_path_data_generic has no backbearings, etc.
    res = pathprof.height_path_data(*args, **kwargs)
    res['heights'][...] = 0
    return res


@lru_cache(maxsize=10, typed=False)
def cached_height_map_data(*args, **kwargs):
    return pathprof.height_map_data(*args, **kwargs)


@lru_cache(maxsize=10, typed=False)
def cached_heightprofile(*args, **kwargs):
    return pathprof.srtm_height_profile(*args, **kwargs)


class BaseWorker(QtCore.QObject):

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
    def on_clear_caches_triggered(self):

        cached_height_path_data.cache_clear()
        cached_height_path_data_generic.cache_clear()
        cached_height_map_data.cache_clear()
        cached_heightprofile.cache_clear()

    @QtCore.pyqtSlot()
    def event_loop(self):

        if self.job_waiting:
            self.job_started.emit()
            try:
                self.do_job()
            except Exception as e:
                self.job_excepted.emit(e.args[0])
            else:
                self.job_finished.emit()

        # setup one-time counter
        self.timer = QtCore.QTimer()
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.event_loop)
        self.timer.start(50)

    @QtCore.pyqtSlot()
    def do_job(self):

        raise NotImplementedError('Cannot use BaseWorker directly!')


class GeometryWorker(BaseWorker):

    @QtCore.pyqtSlot()
    def do_job(self):

        self.job_waiting = False
        jdict = self.job_dict
        print('doing job', jdict)
        print(pathprof.SrtmConf)

        hprof_data = cached_heightprofile(
            jdict['tx_lon'] * u.deg, jdict['tx_lat'] * u.deg,
            jdict['rx_lon'] * u.deg, jdict['rx_lat'] * u.deg,
            jdict['stepsize'] * u.m,
            )
        (
            lons, lats, distance, distances, heights,
            bearing, back_bearing, back_bearings
            ) = hprof_data

        pp = pathprof.PathProp(
            jdict['freq'] * u.GHz,
            jdict['temp'] * u.K, jdict['press'] * u.hPa,
            jdict['tx_lon'] * u.deg, jdict['tx_lat'] * u.deg,
            jdict['rx_lon'] * u.deg, jdict['rx_lat'] * u.deg,
            jdict['tx_height'] * u.m, jdict['rx_height'] * u.m,
            jdict['stepsize'] * u.m,
            jdict['timepercent'] * u.percent,
            version=jdict['version'],
            polarization=jdict['polarization'],
            zone_t=jdict['tx_clutter'], zone_r=jdict['rx_clutter'],
            hprof_dists=distances,
            hprof_heights=heights,
            hprof_bearing=bearing,
            hprof_backbearing=back_bearing,
            )

        results = dict(
            (k, getattr(pp, k))
            for k in dir(pp)
            if not k.startswith('_')
            )

        losses = pathprof.loss_complete(pp)
        loss_names = [
            'L_b0p', 'L_bd', 'L_bs', 'L_ba', 'L_b', 'L_b_corr',
            ]
        results.update(dict(zip(loss_names, losses)))

        self.result_ready.emit(hprof_data, results)

        print('job finished')


class StatisticsWorker(BaseWorker):

    @QtCore.pyqtSlot()
    def do_job(self):

        self.job_waiting = False
        jdict = self.job_dict
        print('doing job', jdict)
        print(pathprof.SrtmConf)

        hprof_data = cached_heightprofile(
            jdict['tx_lon'] * u.deg, jdict['tx_lat'] * u.deg,
            jdict['rx_lon'] * u.deg, jdict['rx_lat'] * u.deg,
            jdict['stepsize'] * u.m,
            )
        (
            lons, lats, distance, distances, heights,
            bearing, back_bearing, back_bearings
            ) = hprof_data

        frequency = np.array([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100])
        time_percent = np.logspace(-3, np.log10(50), 100)

        results = pathprof.losses_complete(
            frequency[:, np.newaxis] * u.GHz,
            jdict['temp'] * u.K, jdict['press'] * u.hPa,
            jdict['tx_lon'] * u.deg, jdict['tx_lat'] * u.deg,
            jdict['rx_lon'] * u.deg, jdict['rx_lat'] * u.deg,
            jdict['tx_height'] * u.m, jdict['rx_height'] * u.m,
            jdict['stepsize'] * u.m,
            time_percent[np.newaxis] * u.percent,
            version=jdict['version'],
            polarization=jdict['polarization'],
            zone_t=jdict['tx_clutter'], zone_r=jdict['rx_clutter'],
            hprof_dists=distances,
            hprof_heights=heights,
            hprof_bearing=bearing,
            hprof_backbearing=back_bearing,
            )

        self.result_ready.emit((frequency, time_percent), results)

        print('job finished')


class PathProfWorker(BaseWorker):

    @QtCore.pyqtSlot()
    def do_job(self):

        self.job_waiting = False
        jdict = self.job_dict
        print('doing job', jdict)

        if jdict['do_generic']:

            hprof_data = cached_height_path_data_generic(
                jdict['tx_lon'] * u.deg, jdict['tx_lat'] * u.deg,
                jdict['rx_lon'] * u.deg, jdict['rx_lat'] * u.deg,
                jdict['stepsize'] * u.m,
                zone_t=jdict['tx_clutter'], zone_r=jdict['rx_clutter'],
                )

        else:

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


class MapWorker(BaseWorker):

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

        # hprof_data contains only a high-res representation of height profs
        lons, lats = hprof_data['xcoords'], hprof_data['ycoords']
        hprof_data['height_map'] = pathprof.srtm_height_data(
            lons[np.newaxis] * u.deg, lats[:, np.newaxis] * u.deg
            )

        self.result_ready.emit(hprof_data, results)

        print('job finished')
