# This file is used to configure the behavior of pytest when using the Astropy
# test infrastructure. It needs to live inside the package in order for it to
# get picked up when running the tests inside an interpreter using
# packagename.test

try:
    from pytest_astropy_header.display import (
        PYTEST_HEADER_MODULES, TESTED_VERSIONS
        )
    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False


def pytest_configure(config):
    if ASTROPY_HEADER:
        config.option.astropy_header = True

        # Customize the following lines to add/remove entries from the
        # list of packages for which version numbers are displayed when
        # running the tests.
        # PYTEST_HEADER_MODULES['Cython'] = 'Cython'  # noqa
        # PYTEST_HEADER_MODULES['Numpy'] = 'numpy'  # noqa
        # PYTEST_HEADER_MODULES['Astropy'] = 'astropy'  # noqa
        # PYTEST_HEADER_MODULES['Scipy'] = 'scipy'  # noqa
        # PYTEST_HEADER_MODULES['Matplotlib'] = 'matplotlib'  # noqa
        # PYTEST_HEADER_MODULES.pop('h5py', None)  # noqa

        from .version import __version__
        TESTED_VERSIONS['pycraf'] = __version__


# want the following two fixtures in multiple sub-packages

import pytest
from . import pathprof

# def pytest_addoption(parser):
#     parser.addoption(
#         '--do-gui-tests', action='store_true', help='Do GUI tests.'
#         )


# def pytest_runtest_setup(item):
#     if 'do_gui_tests' in item.keywords and not item.config.getoption('--do-gui-tests'):
#         pytest.skip('GUI tests are only executed if user provides "--do-gui-tests" command line option')


@pytest.fixture(scope='session')
def srtm_temp_dir(tmp_path_factory):

    tdir = tmp_path_factory.mktemp('srtmdata')
    return str(tdir)


@pytest.fixture(scope='class')
def srtm_handler(srtm_temp_dir):
    print("srtm_handler")

    with pathprof.srtm.SrtmConf.set(
            srtm_dir=srtm_temp_dir,
            server='viewpano',
            download='missing',
            interp='linear',
            ):

        yield
