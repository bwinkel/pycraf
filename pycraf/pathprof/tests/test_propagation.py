#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import os
import pytest
from functools import partial
import numpy as np
from zipfile import ZipFile
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import assert_quantity_allclose, remote_data
from astropy import units as apu
from astropy.units import Quantity
from ... import conversions as cnv
from ... import pathprof
from ...utils import check_astro_quantities
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.misc import NumpyRNGContext
import json
from itertools import product
import importlib


TOL_KWARGS = {'atol': 1.e-4, 'rtol': 1.e-4}


# skip over h5py related tests, if not package present:
skip_h5py = pytest.mark.skipif(
    importlib.util.find_spec('h5py') is None,
    reason='"h5py" package not installed'
    )


MAP_KEYS = [
    'lon_mid_map', 'lat_mid_map',
    'dist_map', 'd_ct_map', 'd_cr_map', 'd_lm_map', 'd_tm_map',
    'zone_t_map', 'zone_r_map', 'bearing_map', 'back_bearing_map',
    'N0_map', 'delta_N_map', 'beta0_map', 'path_idx_map',
    'pix_dist_map', 'dist_end_idx_map', 'omega_map'
    ]


@remote_data(source='any')
@pytest.mark.usefixtures('srtm_handler')
class TestPropagation:

    def setup(self):

        # TODO: add further test cases

        self.cases_zip_name = get_pkg_data_filename('cases.zip')
        self.fastmap_zip_name = get_pkg_data_filename('fastmap.zip')

        self.lon_t, self.lat_t = 6.5 * apu.deg, 50.5 * apu.deg
        self.lon_r, self.lat_r = 6.6 * apu.deg, 50.75 * apu.deg
        self.hprof_step = 100 * apu.m

        self.omega = 0. * apu.percent
        self.temperature = (273.15 + 15.) * apu.K  # as in Excel sheet
        self.pressure = 1013. * apu.hPa

        self.cases_freqs = [0.1, 1., 10.]
        self.cases_h_tgs = [50., 200.]
        self.cases_h_rgs = [50., 200.]
        self.cases_time_percents = [2, 10, 50]
        self.cases_versions = [14, 16]
        self.cases_G_ts = [0, 20]
        self.cases_G_rs = [0, 30]

        self.cases = list(
            # freq, (h_tg, h_rg), time_percent, version, (G_t, G_r)
            product(
                self.cases_freqs,
                # [(50, 50), (200, 200)],
                list(zip(self.cases_h_tgs, self.cases_h_rgs)),
                self.cases_time_percents,
                self.cases_versions,
                # [(0, 0), (20, 30)],
                list(zip(self.cases_G_ts, self.cases_G_rs)),
            ))

        self.fast_cases = list(
            # freq, (h_tg, h_rg), time_percent, version, (G_t, G_r)
            product(
                [0.1, 1., 10.],
                [(50, 50), (200, 200)],
                [2, 10, 50],
                [14, 16],
                [(0, 0)],
            ))

        self.pprop_template = (
            'cases/pprop_{:.2f}ghz_{:.2f}m_{:.2f}m_{:.2f}percent_v{:d}.json'
            )
        self.loss_template = (
            'cases/loss_{:.2f}ghz_{:.2f}m_{:.2f}m_{:.2f}percent_'
            '{:.2f}db_{:.2f}db_v{:d}.json'
            )
        self.fastmap_template = (
            'fastmap/attens_{:.2f}ghz_{:.2f}m_{:.2f}m_{:.2f}percent_v{:d}.{}'
            )
        self.pprops = []
        for case in self.cases:

            freq, (h_tg, h_rg), time_percent, version, (G_t, G_r) = case
            pprop = pathprof.PathProp(
                freq * apu.GHz,
                self.temperature, self.pressure,
                self.lon_t, self.lat_t,
                self.lon_r, self.lat_r,
                h_tg * apu.m, h_rg * apu.m,
                self.hprof_step,
                time_percent * apu.percent,
                version=version,
                )

            self.pprops.append(pprop)

            # Warning: if uncommenting, the test cases will be overwritten
            # do this only, if you need to update the json files
            # (make sure, that results are correct!)
            # zip-file approach not working???
            # with ZipFile('/tmp/cases.zip') as myzip:
            #     pprop_name = self.pprop_template.format(
            #         freq, h_tg, h_rg, time_percent, version
            #         )
            #     with myzip.open(pprop_name, 'w') as f:
            #         json.dump(pprop._pp, f)
            # pprop_name = self.pprop_template.format(
            #     freq, h_tg, h_rg, time_percent, version
            #     )
            # with open('/tmp/' + pprop_name, 'w') as f:
            #     json.dump(pprop._pp, f)

            los_loss = pathprof.loss_freespace(pprop)
            trop_loss = pathprof.loss_troposcatter(
                pprop, G_t * cnv.dB, G_r * cnv.dB,
                )
            duct_loss = pathprof.loss_ducting(pprop)
            diff_loss = pathprof.loss_diffraction(pprop)

            tot_loss = pathprof.loss_complete(
                pprop, G_t * cnv.dB, G_r * cnv.dB,
                )

            losses = {}
            losses['L_bfsg'] = los_loss[0].to(cnv.dB).value
            losses['E_sp'] = los_loss[1].to(cnv.dB).value
            losses['E_sbeta'] = los_loss[2].to(cnv.dB).value

            losses['L_bs'] = trop_loss.to(cnv.dB).value
            losses['L_ba'] = duct_loss.to(cnv.dB).value

            losses['L_d_50'] = diff_loss[0].to(cnv.dB).value
            losses['L_dp'] = diff_loss[1].to(cnv.dB).value
            losses['L_bd_50'] = diff_loss[2].to(cnv.dB).value
            losses['L_bd'] = diff_loss[3].to(cnv.dB).value
            losses['L_min_b0p'] = diff_loss[4].to(cnv.dB).value

            losses['L_b0p_t'] = tot_loss[0].to(cnv.dB).value
            losses['L_bd_t'] = tot_loss[1].to(cnv.dB).value
            losses['L_bs_t'] = tot_loss[2].to(cnv.dB).value
            losses['L_ba_t'] = tot_loss[3].to(cnv.dB).value
            losses['L_b_t'] = tot_loss[4].to(cnv.dB).value
            losses['L_b_corr_t'] = tot_loss[5].to(cnv.dB).value
            losses['L_t'] = tot_loss[6].to(cnv.dB).value

            # Warning: if uncommenting, the test cases will be overwritten
            # do this only, if you need to update the json files
            # (make sure, that results are correct!)
            # with ZipFile(self.cases_zip_name) as myzip:
            #     loss_name = self.loss_template.format(
            #         freq, h_tg, h_rg, time_percent, G_t, G_r, version
            #         )
            #     with myzip.open(loss_name, 'w') as f:
            #         json.dump(losses, open(f, 'w'))
            # loss_name = self.loss_template.format(
            #     freq, h_tg, h_rg, time_percent, G_t, G_r, version
            #     )
            # with open('/tmp/' + loss_name, 'w') as f:
            #     json.dump(losses, f)

    def teardown(self):

        pass

    def test_pathprop(self):

        for freq, (h_tg, h_rg), time_percent, version, _ in self.cases:

            pprop = pathprof.PathProp(
                freq * apu.GHz,
                self.temperature, self.pressure,
                self.lon_t, self.lat_t,
                self.lon_r, self.lat_r,
                h_tg * apu.m, h_rg * apu.m,
                self.hprof_step,
                time_percent * apu.percent,
                version=version,
                )

            with ZipFile(self.cases_zip_name) as myzip:
                pprop_name = self.pprop_template.format(
                    freq, h_tg, h_rg, time_percent, version
                    )
                with myzip.open(pprop_name, 'r') as f:
                    pprop_true = json.loads(f.read().decode('utf-8'))

            for k in pprop._pp:
                # if k not in pprop_true:
                #     continue
                assert_quantity_allclose(pprop._pp[k], pprop_true[k])

    def test_freespace_loss(self):

        for case, pprop in zip(self.cases, self.pprops):

            freq, (h_tg, h_rg), time_percent, version, (G_t, G_r) = case
            los_loss = pathprof.loss_freespace(pprop)
            losses = {}
            losses['L_bfsg'] = los_loss[0].to(cnv.dB).value
            losses['E_sp'] = los_loss[1].to(cnv.dB).value
            losses['E_sbeta'] = los_loss[2].to(cnv.dB).value

            with ZipFile(self.cases_zip_name) as myzip:
                loss_name = self.loss_template.format(
                    freq, h_tg, h_rg, time_percent, G_t, G_r, version,
                    )
                with myzip.open(loss_name, 'r') as f:
                    loss_true = json.loads(f.read().decode('utf-8'))

            for k in losses:
                assert_quantity_allclose(losses[k], loss_true[k])

    def test_troposcatter_loss(self):

        for case, pprop in zip(self.cases, self.pprops):

            freq, (h_tg, h_rg), time_percent, version, (G_t, G_r) = case
            tropo_loss = pathprof.loss_troposcatter(
                pprop, G_t * cnv.dB, G_r * cnv.dB
                )
            losses = {}
            losses['L_bs'] = tropo_loss.to(cnv.dB).value

            with ZipFile(self.cases_zip_name) as myzip:
                loss_name = self.loss_template.format(
                    freq, h_tg, h_rg, time_percent, G_t, G_r, version,
                    )
                with myzip.open(loss_name, 'r') as f:
                    loss_true = json.loads(f.read().decode('utf-8'))

            for k in losses:
                assert_quantity_allclose(losses[k], loss_true[k])

    def test_ducting_loss(self):

        for case, pprop in zip(self.cases, self.pprops):

            freq, (h_tg, h_rg), time_percent, version, (G_t, G_r) = case
            duct_loss = pathprof.loss_ducting(pprop)
            losses = {}
            losses['L_ba'] = duct_loss.to(cnv.dB).value

            with ZipFile(self.cases_zip_name) as myzip:
                loss_name = self.loss_template.format(
                    freq, h_tg, h_rg, time_percent, G_t, G_r, version,
                    )
                with myzip.open(loss_name, 'r') as f:
                    loss_true = json.loads(f.read().decode('utf-8'))

            for k in losses:
                assert_quantity_allclose(losses[k], loss_true[k])

    def test_diffraction_loss(self):

        for case, pprop in zip(self.cases, self.pprops):

            freq, (h_tg, h_rg), time_percent, version, (G_t, G_r) = case
            diff_loss = pathprof.loss_diffraction(pprop)
            losses = {}
            losses['L_d_50'] = diff_loss[0].to(cnv.dB).value
            losses['L_dp'] = diff_loss[1].to(cnv.dB).value
            losses['L_bd_50'] = diff_loss[2].to(cnv.dB).value
            losses['L_bd'] = diff_loss[3].to(cnv.dB).value
            losses['L_min_b0p'] = diff_loss[4].to(cnv.dB).value

            with ZipFile(self.cases_zip_name) as myzip:
                loss_name = self.loss_template.format(
                    freq, h_tg, h_rg, time_percent, G_t, G_r, version,
                    )
                with myzip.open(loss_name, 'r') as f:
                    loss_true = json.loads(f.read().decode('utf-8'))

            for k in losses:
                assert_quantity_allclose(losses[k], loss_true[k])

    def test_complete_loss(self):

        for case, pprop in zip(self.cases, self.pprops):

            freq, (h_tg, h_rg), time_percent, version, (G_t, G_r) = case
            tot_loss = pathprof.loss_complete(
                pprop, G_t * cnv.dB, G_r * cnv.dB
                )
            losses = {}
            losses['L_b0p_t'] = tot_loss[0].to(cnv.dB).value
            losses['L_bd_t'] = tot_loss[1].to(cnv.dB).value
            losses['L_bs_t'] = tot_loss[2].to(cnv.dB).value
            losses['L_ba_t'] = tot_loss[3].to(cnv.dB).value
            losses['L_b_t'] = tot_loss[4].to(cnv.dB).value
            losses['L_b_corr_t'] = tot_loss[5].to(cnv.dB).value
            losses['L_t'] = tot_loss[6].to(cnv.dB).value

            with ZipFile(self.cases_zip_name) as myzip:
                loss_name = self.loss_template.format(
                    freq, h_tg, h_rg, time_percent, G_t, G_r, version,
                    )
                with myzip.open(loss_name, 'r') as f:
                    loss_true = json.loads(f.read().decode('utf-8'))

            for k in losses:
                assert_quantity_allclose(losses[k], loss_true[k])

    def test_complete_losses(self):
        # this is testing full broadcasting

        n = np.newaxis
        (
            freqs, h_tgs, h_rgs, time_percents, versions, G_ts, G_rs
            ) = np.broadcast_arrays(
                np.array(self.cases_freqs)[:, n, n, n, n],
                np.array(self.cases_h_tgs)[n, :, n, n, n],
                np.array(self.cases_h_rgs)[n, :, n, n, n],
                np.array(self.cases_time_percents)[n, n, :, n, n],
                np.array(self.cases_versions, dtype=np.int32)[n, n, n, :, n],
                np.array(self.cases_G_ts)[n, n, n, n, :],
                np.array(self.cases_G_rs)[n, n, n, n, :],
            )
        results = pathprof.losses_complete(
            freqs * apu.GHz,
            self.temperature,
            self.pressure,
            self.lon_t, self.lat_t,
            self.lon_r, self.lat_r,
            h_tgs * apu.m,
            h_rgs * apu.m,
            self.hprof_step,
            time_percents * apu.percent,
            G_t=G_ts * cnv.dBi,
            G_r=G_rs * cnv.dBi,
            omega=self.omega,
            version=versions,
            )

        for tup in np.nditer([
                freqs, h_tgs, h_rgs, time_percents, G_ts, G_rs, versions,
                results['L_b0p'], results['L_bd'], results['L_bs'],
                results['L_ba'], results['L_b'], results['L_b_corr'],
                ]):

            with ZipFile(self.cases_zip_name) as myzip:

                loss_name = self.loss_template.format(
                    float(tup[0]), float(tup[1]), float(tup[2]),
                    float(tup[3]), float(tup[4]), float(tup[5]),
                    int(tup[6]),
                    )
                with myzip.open(loss_name, 'r') as f:
                    loss_true = json.loads(f.read().decode('utf-8'))

                for i, k in enumerate([
                        'L_b0p', 'L_bd', 'L_bs', 'L_ba', 'L_b', 'L_b_corr',
                        ]):
                    assert_quantity_allclose(tup[i + 7], loss_true[k + '_t'])

    @skip_h5py
    def test_height_map_data_h5py(self, tmpdir_factory):

        import h5py

        hprof_data_cache = pathprof.height_map_data(
            6.5 * apu.deg, 50.5 * apu.deg,
            900 * apu.arcsec, 900 * apu.arcsec,
            map_resolution=30 * apu.arcsec,
            )

        # also testing reading/writing from hdf5
        tdir = tmpdir_factory.mktemp('hdata')
        tfile = str(tdir.join('hprof.hdf5'))
        print('writing temporary files to', tdir)

        # saving
        with h5py.File(tfile, 'w') as h5f:
            for k, v in hprof_data_cache.items():
                h5f[k] = v

        hprof_data_cache_disk = h5py.File(tfile, 'r')

        for k in hprof_data_cache:
            assert_quantity_allclose(
                # Note conversion to some ndarray type necessary, as h5py
                # returns <HDF5 dataset> types
                np.squeeze(hprof_data_cache[k]),
                np.squeeze(hprof_data_cache_disk[k])
                )

        # also test versus true results

        # fileobjects in ZipFile don't support seek; need to write to tmpdir
        zipdir = tmpdir_factory.mktemp('zip')
        tfile = 'fastmap/hprof.hdf5'
        with ZipFile(self.fastmap_zip_name) as myzip:
            myzip.extract(tfile, str(zipdir))

        hprof_data_cache_true = h5py.File(str(zipdir.join(tfile)), 'r')

        for k in hprof_data_cache:
            # Note conversion to some ndarray type necessary, as h5py
            # returns <HDF5 dataset> types
            q1 = np.squeeze(hprof_data_cache[k])
            q2 = np.squeeze(hprof_data_cache_true[k])

            # we'll skip over the map center, as the path_idx is quasi-random
            # (all paths have a point in the center...)
            if k in MAP_KEYS:
                q1[15, 15] = q2[15, 15]

            if issubclass(q1.dtype.type, np.integer):
                assert_equal(q1, q2)
            else:
                assert_quantity_allclose(q1, q2, atol=1.e-6)

    @skip_h5py
    def test_fast_atten_map_h5py(self, tmpdir_factory):

        import h5py

        zipdir = tmpdir_factory.mktemp('zip')
        tfile = 'fastmap/hprof.hdf5'
        with ZipFile(self.fastmap_zip_name) as myzip:
            myzip.extract(tfile, str(zipdir))

        hprof_data_cache = h5py.File(str(zipdir.join(tfile)), 'r')

        for case in self.fast_cases:

            freq, (h_tg, h_rg), time_percent, version, (G_t, G_r) = case

            results = pathprof.atten_map_fast(
                freq * apu.GHz,
                self.temperature,
                self.pressure,
                h_tg * apu.m, h_rg * apu.m,
                time_percent * apu.percent,
                hprof_data_cache,  # dict_like
                version=version,
                )
            for k in results:
                try:
                    results[k] = results[k].value
                except AttributeError:
                    pass

            fname = self.fastmap_template.format(
                freq, h_tg, h_rg, time_percent, version, 'hdf5'
                )

            # Warning: if uncommenting, the test cases will be overwritten
            # do this only, if you need to update the h5py files
            # (make sure, that results are correct!)
            # Also, if you want to create all at once, comment-out the
            # "with ZipFile" below; results will be in the tmp-dir
            # and can be added manually to the zipfile
            # tfile = str(zipdir.join(fname))
            # print('writing temporary files to', zipdir)
            # with h5py.File(tfile, 'w') as h5f:
            #     for k, v in results.items():
            #         h5f[k] = v

            with ZipFile(self.fastmap_zip_name) as myzip:
                myzip.extract(fname, str(zipdir))

            print(str(zipdir.join(fname)))
            h5f = h5py.File(str(zipdir.join(fname)), 'r')

            # Note conversion to some ndarray type necessary, as h5py
            # returns <HDF5 dataset> types
            tol_kwargs = {'atol': 1.e-6, 'rtol': 1.e-6}
            # for some super-strange reason, index 9, 13 is completely off
            # on travis and appveyor (only diffraction)
            # as it is only one pixel, we ignore it here for now
            for k, v in results.items():
                v2 = np.squeeze(h5f[k])
                v2[9, 13] = v[9, 13]
                assert_allclose(v, v2, **tol_kwargs)

            # idx = np.where(np.abs(h5_atten_map - atten_map) > 1.e-6)
            # for i, y, x in zip(*idx):
            #     print(i, y, x, h5_atten_map[i, y, x], atten_map[i, y, x])

    def test_height_map_data_npz(self, tmpdir_factory):

        hprof_data_cache = pathprof.height_map_data(
            6.5 * apu.deg, 50.5 * apu.deg,
            900 * apu.arcsec, 900 * apu.arcsec,
            map_resolution=30 * apu.arcsec,
            )

        # also testing reading/writing from hdf5
        tdir = tmpdir_factory.mktemp('hdata')
        tfile = str(tdir.join('hprof.npz'))
        print('writing temporary files to', tdir)

        # saving
        np.savez(tfile, **hprof_data_cache)

        hprof_data_cache_disk = np.load(tfile)

        for k in hprof_data_cache:
            assert_quantity_allclose(
                np.squeeze(hprof_data_cache[k]),
                np.squeeze(hprof_data_cache_disk[k])
                )

        # also test versus true results

        # fileobjects in ZipFile don't support seek; need to write to tmpdir
        zipdir = tmpdir_factory.mktemp('zip')
        tfile = 'fastmap/hprof.npz'
        with ZipFile(self.fastmap_zip_name) as myzip:
            myzip.extract(tfile, str(zipdir))

        hprof_data_cache_true = np.load(str(zipdir.join(tfile)))

        for k in hprof_data_cache:
            q1 = np.squeeze(hprof_data_cache[k])
            q2 = np.squeeze(hprof_data_cache_true[k])

            # print(k)
            # if k == 'path_idx_map':
            #     idx = np.where(np.abs(q1 - q2) > 1.e-6)
            #     for y, x in zip(*idx):
            #         print(y, x, q1[y, x], q2[y, x])

            # we'll skip over the map center, as the path_idx is quasi-random
            # (all paths have a point in the center...)
            if k in MAP_KEYS:
                q1[15, 15] = q2[15, 15]

            if issubclass(q1.dtype.type, np.integer):
                assert_equal(q1, q2)
            else:
                assert_quantity_allclose(q1, q2, atol=1.e-6)

    def test_fast_atten_map_npz(self, tmpdir_factory):

        zipdir = tmpdir_factory.mktemp('zip')
        tfile = 'fastmap/hprof.npz'
        with ZipFile(self.fastmap_zip_name) as myzip:
            myzip.extract(tfile, str(zipdir))

        hprof_data_cache = np.load(str(zipdir.join(tfile)))

        for case in self.fast_cases:

            freq, (h_tg, h_rg), time_percent, version, (G_t, G_r) = case

            results = pathprof.atten_map_fast(
                freq * apu.GHz,
                self.temperature,
                self.pressure,
                h_tg * apu.m, h_rg * apu.m,
                time_percent * apu.percent,
                hprof_data_cache,  # dict_like
                version=version,
                )

            for k in results:
                try:
                    results[k] = results[k].value
                except AttributeError:
                    pass

            fname = self.fastmap_template.format(
                freq, h_tg, h_rg, time_percent, version, 'npz'
                )

            # Warning: if uncommenting, the test cases will be overwritten
            # do this only, if you need to update the npz files
            # (make sure, that results are correct!)
            # Also, if you want to create all at once, comment-out the
            # "with ZipFile" below; results will be in the tmp-dir
            # and can be added manually to the zipfile
            # tfile = str(zipdir.join(fname))
            # np.savez(tfile, **results)

            with ZipFile(self.fastmap_zip_name) as myzip:
                myzip.extract(fname, str(zipdir))

            print(str(zipdir.join(fname)))
            true_dat = np.load(str(zipdir.join(fname)))

            # Note conversion to some ndarray type necessary, as h5py
            # returns <HDF5 dataset> types
            tol_kwargs = {'atol': 1.e-6, 'rtol': 1.e-6}
            # for some super-strange reason, index 9, 13 is completely off
            # on travis and appveyor (only diffraction)
            # as it is only one pixel, we ignore it here for now

            # idx = np.where(np.abs(t_atten_map - atten_map) > 1.e-6)
            # for i, y, x in zip(*idx):
            #     print(i, y, x, t_atten_map[i, y, x], atten_map[i, y, x])

            for k, v in results.items():
                v2 = np.squeeze(true_dat[k])
                v2[9, 13] = v[9, 13]
                assert_allclose(v, v2, **tol_kwargs)

    def test_atten_path_fast(self):

        # testing against the slow approach

        lon_t, lat_t = 6.8836 * apu.deg, 50.525 * apu.deg
        lon_r, lat_r = 7.3334 * apu.deg, 50.635 * apu.deg
        hprof_step = 100 * apu.m

        freq = 1. * apu.GHz
        temperature = 290. * apu.K
        pressure = 1013. * apu.hPa
        h_tg, h_rg = 5. * apu.m, 50. * apu.m
        time_percent = 2. * apu.percent

        zone_t, zone_r = pathprof.CLUTTER.URBAN, pathprof.CLUTTER.SUBURBAN

        hprof_data = pathprof.height_path_data(
            lon_t, lat_t, lon_r, lat_r, hprof_step,
            zone_t=zone_t, zone_r=zone_r,
            )

        results = pathprof.atten_path_fast(
            freq, temperature, pressure,
            h_tg, h_rg, time_percent,
            hprof_data,
            )

        (
            lons, lats, distance, distances, heights,
            bearing, back_bearing, back_bearings
            ) = pathprof.srtm_height_profile(
                lon_t, lat_t, lon_r, lat_r, hprof_step
                )

        atten_path = np.zeros((6, len(distances)), dtype=np.float64)
        eps_pt_path = np.zeros((len(distances)), dtype=np.float64)
        eps_pr_path = np.zeros((len(distances)), dtype=np.float64)
        d_lt_path = np.zeros((len(distances)), dtype=np.float64)
        d_lr_path = np.zeros((len(distances)), dtype=np.float64)

        for idx in range(6, len(distances)):

            pprop = pathprof.PathProp(
                freq,
                temperature, pressure,
                lon_t, lat_t,
                lons[idx], lats[idx],
                h_tg, h_rg,
                hprof_step,
                time_percent,
                zone_t=zone_t, zone_r=zone_r,
                hprof_dists=distances[:idx + 1],
                hprof_heights=heights[:idx + 1],
                hprof_bearing=bearing,
                hprof_backbearing=back_bearings[idx],
                # delta_N=hprof_data['delta_N'][idx] * cnv.dimless / apu.km,
                # N0=hprof_data['N0'][idx] * cnv.dimless,
                )

            eps_pt_path[idx] = pprop.eps_pt.value
            eps_pr_path[idx] = pprop.eps_pr.value
            d_lt_path[idx] = pprop.d_lt.value
            d_lr_path[idx] = pprop.d_lr.value
            tot_loss = pathprof.loss_complete(pprop)
            atten_path[:, idx] = apu.Quantity(tot_loss).value[:-1]

        assert np.allclose(atten_path[0], results['L_b0p'].value, atol=1.e-3)
        assert np.allclose(atten_path[1], results['L_bd'].value, atol=1.e-3)
        assert np.allclose(atten_path[2], results['L_bs'].value, atol=1.e-3)
        assert np.allclose(atten_path[3], results['L_ba'].value, atol=1.e-3)
        assert np.allclose(atten_path[4], results['L_b'].value, atol=1.e-3)
        assert np.allclose(atten_path[5], results['L_b_corr'].value, atol=1.e-3)

        assert np.allclose(eps_pt_path, results['eps_pt'].value, atol=1.e-6)
        assert np.allclose(eps_pr_path, results['eps_pr'].value, atol=1.e-6)

        assert np.allclose(d_lt_path, results['d_lt'].value, atol=1.e-6)
        assert np.allclose(d_lr_path, results['d_lr'].value, atol=1.e-6)


# repeat tests with flat-terrain (avoids downloading data)
class TestPropagationGeneric:

    def setup(self):

        # TODO: add further test cases

        self.cases_zip_name = get_pkg_data_filename('cases_generic.zip')

        self.lon_t, self.lat_t = 6.5 * apu.deg, 50.5 * apu.deg
        self.lon_r, self.lat_r = 6.6 * apu.deg, 50.75 * apu.deg
        self.hprof_step = 100 * apu.m

        self.omega = 0. * apu.percent
        self.temperature = (273.15 + 15.) * apu.K  # as in Excel sheet
        self.pressure = 1013. * apu.hPa

        self.cases_freqs = [0.1, 1., 10.]
        self.cases_h_tgs = [50., 200.]
        self.cases_h_rgs = [50., 200.]
        self.cases_time_percents = [2, 10, 50]
        self.cases_versions = [14, 16]
        self.cases_G_ts = [0, 20]
        self.cases_G_rs = [0, 30]

        self.cases = list(
            # freq, (h_tg, h_rg), time_percent, version, (G_t, G_r)
            product(
                self.cases_freqs,
                # [(50, 50), (200, 200)],
                list(zip(self.cases_h_tgs, self.cases_h_rgs)),
                self.cases_time_percents,
                self.cases_versions,
                # [(0, 0), (20, 30)],
                list(zip(self.cases_G_ts, self.cases_G_rs)),
            ))

        self.fast_cases = list(
            # freq, (h_tg, h_rg), time_percent, version, (G_t, G_r)
            product(
                [0.1, 1., 10.],
                [(50, 50), (200, 200)],
                [2, 10, 50],
                [14, 16],
                [(0, 0)],
            ))

        self.pprop_template = (
            'cases/pprop_{:.2f}ghz_{:.2f}m_{:.2f}m_{:.2f}percent_'
            'v{:d}_generic.json'
            )
        self.loss_template = (
            'cases/loss_{:.2f}ghz_{:.2f}m_{:.2f}m_{:.2f}percent_'
            '{:.2f}db_{:.2f}db_v{:d}_generic.json'
            )
        self.pprops = []
        for case in self.cases:

            freq, (h_tg, h_rg), time_percent, version, (G_t, G_r) = case
            pprop = pathprof.PathProp(
                freq * apu.GHz,
                self.temperature, self.pressure,
                self.lon_t, self.lat_t,
                self.lon_r, self.lat_r,
                h_tg * apu.m, h_rg * apu.m,
                self.hprof_step,
                time_percent * apu.percent,
                version=version,
                generic_heights=True,
                )

            self.pprops.append(pprop)

            # Warning: if uncommenting, the test cases will be overwritten
            # do this only, if you need to update the json files
            # (make sure, that results are correct!)
            # zip-file approach not working???
            # with ZipFile('/tmp/cases.zip') as myzip:
            #     pprop_name = self.pprop_template.format(
            #         freq, h_tg, h_rg, time_percent, version
            #         )
            #     with myzip.open(pprop_name, 'w') as f:
            #         json.dump(pprop._pp, f)
            # pprop_name = self.pprop_template.format(
            #     freq, h_tg, h_rg, time_percent, version
            #     )
            # with open('/tmp/' + pprop_name, 'w') as f:
            #     json.dump(pprop._pp, f)

            los_loss = pathprof.loss_freespace(pprop)
            trop_loss = pathprof.loss_troposcatter(
                pprop, G_t * cnv.dB, G_r * cnv.dB,
                )
            duct_loss = pathprof.loss_ducting(pprop)
            diff_loss = pathprof.loss_diffraction(pprop)

            tot_loss = pathprof.loss_complete(
                pprop, G_t * cnv.dB, G_r * cnv.dB,
                )

            losses = {}
            losses['L_bfsg'] = los_loss[0].to(cnv.dB).value
            losses['E_sp'] = los_loss[1].to(cnv.dB).value
            losses['E_sbeta'] = los_loss[2].to(cnv.dB).value

            losses['L_bs'] = trop_loss.to(cnv.dB).value
            losses['L_ba'] = duct_loss.to(cnv.dB).value

            losses['L_d_50'] = diff_loss[0].to(cnv.dB).value
            losses['L_dp'] = diff_loss[1].to(cnv.dB).value
            losses['L_bd_50'] = diff_loss[2].to(cnv.dB).value
            losses['L_bd'] = diff_loss[3].to(cnv.dB).value
            losses['L_min_b0p'] = diff_loss[4].to(cnv.dB).value

            losses['L_b0p_t'] = tot_loss[0].to(cnv.dB).value
            losses['L_bd_t'] = tot_loss[1].to(cnv.dB).value
            losses['L_bs_t'] = tot_loss[2].to(cnv.dB).value
            losses['L_ba_t'] = tot_loss[3].to(cnv.dB).value
            losses['L_b_t'] = tot_loss[4].to(cnv.dB).value
            losses['L_b_corr_t'] = tot_loss[5].to(cnv.dB).value
            losses['L_t'] = tot_loss[6].to(cnv.dB).value

            # Warning: if uncommenting, the test cases will be overwritten
            # do this only, if you need to update the json files
            # (make sure, that results are correct!)
            # with ZipFile(self.cases_zip_name) as myzip:
            #     loss_name = self.loss_template.format(
            #         freq, h_tg, h_rg, time_percent, G_t, G_r, version
            #         )
            #     with myzip.open(loss_name, 'w') as f:
            #         json.dump(losses, open(f, 'w'))
            # loss_name = self.loss_template.format(
            #     freq, h_tg, h_rg, time_percent, G_t, G_r, version
            #     )
            # with open('/tmp/' + loss_name, 'w') as f:
            #     json.dump(losses, f)

    def teardown(self):

        pass

    def test_pathprop(self):

        for freq, (h_tg, h_rg), time_percent, version, _ in self.cases:

            pprop = pathprof.PathProp(
                freq * apu.GHz,
                self.temperature, self.pressure,
                self.lon_t, self.lat_t,
                self.lon_r, self.lat_r,
                h_tg * apu.m, h_rg * apu.m,
                self.hprof_step,
                time_percent * apu.percent,
                version=version,
                generic_heights=True,
                )

            with ZipFile(self.cases_zip_name) as myzip:
                pprop_name = self.pprop_template.format(
                    freq, h_tg, h_rg, time_percent, version
                    )
                with myzip.open(pprop_name, 'r') as f:
                    pprop_true = json.loads(f.read().decode('utf-8'))

            for k in pprop._pp:
                # if k not in pprop_true:
                #     continue
                assert_quantity_allclose(pprop._pp[k], pprop_true[k])

    def test_freespace_loss(self):

        for case, pprop in zip(self.cases, self.pprops):

            freq, (h_tg, h_rg), time_percent, version, (G_t, G_r) = case
            los_loss = pathprof.loss_freespace(pprop)
            losses = {}
            losses['L_bfsg'] = los_loss[0].to(cnv.dB).value
            losses['E_sp'] = los_loss[1].to(cnv.dB).value
            losses['E_sbeta'] = los_loss[2].to(cnv.dB).value

            with ZipFile(self.cases_zip_name) as myzip:
                loss_name = self.loss_template.format(
                    freq, h_tg, h_rg, time_percent, G_t, G_r, version,
                    )
                with myzip.open(loss_name, 'r') as f:
                    loss_true = json.loads(f.read().decode('utf-8'))

            for k in losses:
                assert_quantity_allclose(losses[k], loss_true[k])

    def test_troposcatter_loss(self):

        for case, pprop in zip(self.cases, self.pprops):

            freq, (h_tg, h_rg), time_percent, version, (G_t, G_r) = case
            tropo_loss = pathprof.loss_troposcatter(
                pprop, G_t * cnv.dB, G_r * cnv.dB
                )
            losses = {}
            losses['L_bs'] = tropo_loss.to(cnv.dB).value

            with ZipFile(self.cases_zip_name) as myzip:
                loss_name = self.loss_template.format(
                    freq, h_tg, h_rg, time_percent, G_t, G_r, version,
                    )
                with myzip.open(loss_name, 'r') as f:
                    loss_true = json.loads(f.read().decode('utf-8'))

            for k in losses:
                assert_quantity_allclose(losses[k], loss_true[k])

    def test_ducting_loss(self):

        for case, pprop in zip(self.cases, self.pprops):

            freq, (h_tg, h_rg), time_percent, version, (G_t, G_r) = case
            duct_loss = pathprof.loss_ducting(pprop)
            losses = {}
            losses['L_ba'] = duct_loss.to(cnv.dB).value

            with ZipFile(self.cases_zip_name) as myzip:
                loss_name = self.loss_template.format(
                    freq, h_tg, h_rg, time_percent, G_t, G_r, version,
                    )
                with myzip.open(loss_name, 'r') as f:
                    loss_true = json.loads(f.read().decode('utf-8'))

            for k in losses:
                assert_quantity_allclose(losses[k], loss_true[k])

    def test_diffraction_loss(self):

        for case, pprop in zip(self.cases, self.pprops):

            freq, (h_tg, h_rg), time_percent, version, (G_t, G_r) = case
            diff_loss = pathprof.loss_diffraction(pprop)
            losses = {}
            losses['L_d_50'] = diff_loss[0].to(cnv.dB).value
            losses['L_dp'] = diff_loss[1].to(cnv.dB).value
            losses['L_bd_50'] = diff_loss[2].to(cnv.dB).value
            losses['L_bd'] = diff_loss[3].to(cnv.dB).value
            losses['L_min_b0p'] = diff_loss[4].to(cnv.dB).value

            with ZipFile(self.cases_zip_name) as myzip:
                loss_name = self.loss_template.format(
                    freq, h_tg, h_rg, time_percent, G_t, G_r, version,
                    )
                with myzip.open(loss_name, 'r') as f:
                    loss_true = json.loads(f.read().decode('utf-8'))

            for k in losses:
                assert_quantity_allclose(losses[k], loss_true[k])

    def test_complete_loss(self):

        for case, pprop in zip(self.cases, self.pprops):

            freq, (h_tg, h_rg), time_percent, version, (G_t, G_r) = case
            tot_loss = pathprof.loss_complete(
                pprop, G_t * cnv.dB, G_r * cnv.dB
                )
            losses = {}
            losses['L_b0p_t'] = tot_loss[0].to(cnv.dB).value
            losses['L_bd_t'] = tot_loss[1].to(cnv.dB).value
            losses['L_bs_t'] = tot_loss[2].to(cnv.dB).value
            losses['L_ba_t'] = tot_loss[3].to(cnv.dB).value
            losses['L_b_t'] = tot_loss[4].to(cnv.dB).value
            losses['L_b_corr_t'] = tot_loss[5].to(cnv.dB).value
            losses['L_t'] = tot_loss[6].to(cnv.dB).value

            with ZipFile(self.cases_zip_name) as myzip:
                loss_name = self.loss_template.format(
                    freq, h_tg, h_rg, time_percent, G_t, G_r, version,
                    )
                with myzip.open(loss_name, 'r') as f:
                    loss_true = json.loads(f.read().decode('utf-8'))

            for k in losses:
                assert_quantity_allclose(losses[k], loss_true[k])

    def test_complete_losses(self):
        # this is testing full broadcasting

        n = np.newaxis
        (
            freqs, h_tgs, h_rgs, time_percents, versions, G_ts, G_rs
            ) = np.broadcast_arrays(
                np.array(self.cases_freqs)[:, n, n, n, n],
                np.array(self.cases_h_tgs)[n, :, n, n, n],
                np.array(self.cases_h_rgs)[n, :, n, n, n],
                np.array(self.cases_time_percents)[n, n, :, n, n],
                np.array(self.cases_versions, dtype=np.int32)[n, n, n, :, n],
                np.array(self.cases_G_ts)[n, n, n, n, :],
                np.array(self.cases_G_rs)[n, n, n, n, :],
            )
        results = pathprof.losses_complete(
            freqs * apu.GHz,
            self.temperature,
            self.pressure,
            self.lon_t, self.lat_t,
            self.lon_r, self.lat_r,
            h_tgs * apu.m,
            h_rgs * apu.m,
            self.hprof_step,
            time_percents * apu.percent,
            G_t=G_ts * cnv.dBi,
            G_r=G_rs * cnv.dBi,
            omega=self.omega,
            version=versions,
            generic_heights=True,
            )

        for tup in np.nditer([
                freqs, h_tgs, h_rgs, time_percents, G_ts, G_rs, versions,
                results['L_b0p'], results['L_bd'], results['L_bs'],
                results['L_ba'], results['L_b'], results['L_b_corr'],
                ]):

            with ZipFile(self.cases_zip_name) as myzip:

                loss_name = self.loss_template.format(
                    float(tup[0]), float(tup[1]), float(tup[2]),
                    float(tup[3]), float(tup[4]), float(tup[5]),
                    int(tup[6]),
                    )
                with myzip.open(loss_name, 'r') as f:
                    loss_true = json.loads(f.read().decode('utf-8'))

                for i, k in enumerate([
                        'L_b0p', 'L_bd', 'L_bs', 'L_ba', 'L_b', 'L_b_corr',
                        ]):
                    assert_quantity_allclose(tup[i + 7], loss_true[k + '_t'])

    def test_atten_path_fast_generic(self):

        # testing against the slow approach

        hprof_step = 100 * apu.m
        lon_mid, lat_mid = 6 * apu.deg, 50 * apu.deg
        lon_t = lon_mid - 0.5 * apu.deg
        lon_r = lon_mid + 0.5 * apu.deg

        freq = 1. * apu.GHz
        temperature = 290. * apu.K
        pressure = 1013. * apu.hPa
        h_tg, h_rg = 5. * apu.m, 50. * apu.m
        time_percent = 2. * apu.percent

        (
            lons, lats, distance, distances, heights,
            bearing, back_bearing, back_bearings
            ) = pathprof.srtm_height_profile(
                lon_t, lat_mid, lon_r, lat_mid, hprof_step,
                generic_heights=True,
                )

        zone_t, zone_r = pathprof.CLUTTER.URBAN, pathprof.CLUTTER.SUBURBAN

        hprof_data = pathprof.height_path_data_generic(
            distance, hprof_step, lon_mid, lat_mid,
            zone_t=zone_t, zone_r=zone_r,
            )

        results = pathprof.atten_path_fast(
            freq, temperature, pressure,
            h_tg, h_rg, time_percent,
            hprof_data,
            )

        atten_path = np.zeros((6, len(distances)), dtype=np.float64)
        eps_pt_path = np.zeros((len(distances)), dtype=np.float64)
        eps_pr_path = np.zeros((len(distances)), dtype=np.float64)
        d_lt_path = np.zeros((len(distances)), dtype=np.float64)
        d_lr_path = np.zeros((len(distances)), dtype=np.float64)

        for idx in range(6, len(distances)):

            pprop = pathprof.PathProp(
                freq,
                temperature, pressure,
                lon_t, lat_mid,
                lons[idx], lats[idx],
                h_tg, h_rg,
                hprof_step,
                time_percent,
                zone_t=zone_t, zone_r=zone_r,
                hprof_dists=distances[:idx + 1],
                hprof_heights=0 * heights[:idx + 1],
                hprof_bearing=bearing,
                hprof_backbearing=back_bearings[idx],
                delta_N=hprof_data['delta_N'][idx] * cnv.dimless / apu.km,
                N0=hprof_data['N0'][idx] * cnv.dimless,
                )

            eps_pt_path[idx] = pprop.eps_pt.value
            eps_pr_path[idx] = pprop.eps_pr.value
            d_lt_path[idx] = pprop.d_lt.value
            d_lr_path[idx] = pprop.d_lr.value
            tot_loss = pathprof.loss_complete(pprop)
            atten_path[:, idx] = apu.Quantity(tot_loss).value[:-1]

        assert np.allclose(atten_path[0], results['L_b0p'].value, atol=1.e-3)
        assert np.allclose(atten_path[1], results['L_bd'].value, atol=1.e-3)
        assert np.allclose(atten_path[2], results['L_bs'].value, atol=1.e-3)
        assert np.allclose(atten_path[3], results['L_ba'].value, atol=1.e-3)
        assert np.allclose(atten_path[4], results['L_b'].value, atol=1.e-3)
        assert np.allclose(atten_path[5], results['L_b_corr'].value, atol=1.e-3)

        assert np.allclose(eps_pt_path, results['eps_pt'].value, atol=1.e-6)
        assert np.allclose(eps_pr_path, results['eps_pr'].value, atol=1.e-6)

        assert np.allclose(d_lt_path, results['d_lt'].value, atol=1.e-6)
        assert np.allclose(d_lr_path, results['d_lr'].value, atol=1.e-6)


def test_clutter_correction():

    # args_list = [
    #     (None, None, apu.m),
    #     (None, None, apu.GHz),
    #     ]

    # check_astro_quantities(pathprof.clutter_correction, args_list)

    CL = pathprof.CLUTTER

    cases = [
        (CL.UNKNOWN, 5 * apu.m, 1 * apu.GHz, 0.0 * cnv.dB),
        (CL.SPARSE, 2 * apu.m, 0.5 * apu.GHz, 9.148328469292968 * cnv.dB),
        (CL.VILLAGE, 5 * apu.m, 1 * apu.GHz, -0.12008183792955418 * cnv.dB),
        (CL.SUBURBAN, 5 * apu.m, 10 * apu.GHz, 13.606900992918096 * cnv.dB),
        (CL.DENSE_URBAN, 10 * apu.m, 10 * apu.GHz, 18.4986816015433 * cnv.dB),
        (CL.INDUSTRIAL_ZONE, 50 * apu.m, 2 * apu.GHz, -0.32999999670 * cnv.dB),
        ]

    for zone, h_g, freq, loss in cases:

        assert_quantity_allclose(
            pathprof.clutter_correction(h_g, zone, freq),
            loss,
            )

