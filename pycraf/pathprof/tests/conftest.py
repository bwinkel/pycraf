#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pytest
from ... import pathprof


@pytest.fixture(scope='session')
def srtm_temp_dir(tmpdir_factory):

    tdir = tmpdir_factory.mktemp('srtmdata')
    return str(tdir)


@pytest.yield_fixture()
def srtm_handler(srtm_temp_dir):

    with pathprof.srtm.SrtmConf.set(
            srtm_dir=srtm_temp_dir,
            server='nasa_v2.1',
            download='missing',
            ):

        yield
