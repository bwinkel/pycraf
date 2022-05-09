#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# adapted from https://github.com/astropy/astropy
# Note: This file needs to be Python 2 / <3.6 compatible, so that the nice
# "This package only supports Python 3.x+" error prints without syntax errors etc.


'''
Note: if you get an error:

> error: [Errno 2] Could not find C/C++ file pycraf/pathprof/cyprop.(c/cpp)
> for Cython file pycraf/pathprof/cyprop.pyx when building extension
> pycraf.pathprof.cyprop. Cython must be installed to build from a git
> checkout.: 'pycraf/pathprof/cyprop.c'

Delete the file "pycraf/cython_version.py"

'''

import sys


TEST_HELP = """
Note: running tests is no longer done using 'python setup.py test'. Instead
you will need to run:
    tox -e test
If you don't already have tox installed, you can install it with:
    pip install tox
If you only want to run part of the test suite, you can also use pytest
directly with::
    pip install -e .[test]
    pytest --pyargs pycraf --remote-data=any
    # or individual tests:
    pytest --pyargs pycraf --remote-data=any -k <test_func_name/module_name/etc.>
    # with docstrings (note: --doctest-modules turns off doctest-plus):
    pytest --pyargs pycraf --remote-data=any --doctest-modules --ignore-glob="*/setup_package.py"
    # with doctests (in project dir)
    pytest -rsx --doctest-rst --remote-data=any docs

For more information, see:
  https://docs.astropy.org/en/latest/development/testguide.html#running-tests
"""

if 'test' in sys.argv:
    print(TEST_HELP)
    sys.exit(1)


DOCS_HELP = """
Note: building the documentation is no longer done using
'python setup.py build_docs'. Instead you will need to run:
    tox -e build_docs
If you don't already have tox installed, you can install it with:
    pip install tox
You can also build the documentation with Sphinx directly using::
    pip install -e .[docs]
    cd docs
    # make clean  # to rebuild everything
    make html
    # alternatively (in project dir):
    sphinx-build docs docs/_build/html -b html
    sphinx-build docs docs/_build/html -b html -W  # fail on warnings

For more information, see:
  https://docs.astropy.org/en/latest/install.html#builddocs
"""

if 'build_docs' in sys.argv or 'build_sphinx' in sys.argv:
    print(DOCS_HELP)
    sys.exit(1)


# Only import these if the above checks are okay
# to avoid masking the real problem with import error.
from setuptools import setup  # noqa
from extension_helpers import get_extensions  # noqa

# import numpy as np
# np.import_array()

setup(ext_modules=get_extensions())



# import glob
# import os


# try:
#     from configparser import ConfigParser
# except ImportError:
#     from ConfigParser import ConfigParser

# # Get some values from the setup.cfg
# conf = ConfigParser()
# conf.read(['setup.cfg'])
# metadata = dict(conf.items('metadata'))

# PACKAGENAME = metadata.get('package_name', 'pycraf')
# DESCRIPTION = metadata.get('description', 'pycraf')
# AUTHOR = metadata.get('author', 'Benjamin Winkel')
# AUTHOR_EMAIL = metadata.get('author_email', 'bwinkel@mpifr.de')
# LICENSE = metadata.get('license', 'GPLv3')
# URL = metadata.get('url', 'https://github.com/bwinkel/pycraf')
# __minimum_python_version__ = metadata.get("minimum_python_version", "2.7")

# # Enforce Python version check - this is the same check as in __init__.py but
# # this one has to happen before importing ah_bootstrap.
# if sys.version_info < tuple((int(val) for val in __minimum_python_version__.split('.'))):
#     sys.stderr.write("ERROR: pycraf requires Python {} or later\n".format(__minimum_python_version__))
#     sys.exit(1)

# # Import ah_bootstrap after the python version validation

# import ah_bootstrap
# from setuptools import setup

# # A dirty hack to get around some early import/configurations ambiguities
# if sys.version_info[0] >= 3:
#     import builtins
# else:
#     import __builtin__ as builtins
# builtins._ASTROPY_SETUP_ = True

# from astropy_helpers.setup_helpers import (register_commands, get_debug_option,
#                                            get_package_info)
# from astropy_helpers.git_helpers import get_git_devstr
# from astropy_helpers.version_helpers import generate_version_py


# # order of priority for long_description:
# #   (1) set in setup.cfg,
# #   (2) load LONG_DESCRIPTION.rst,
# #   (3) load README.rst,
# #   (4) package docstring
# readme_glob = 'README*'
# _cfg_long_description = metadata.get('long_description', '')
# if _cfg_long_description:
#     LONG_DESCRIPTION = _cfg_long_description

# elif os.path.exists('LONG_DESCRIPTION.rst'):
#     with open('LONG_DESCRIPTION.rst') as f:
#         LONG_DESCRIPTION = f.read()

# elif len(glob.glob(readme_glob)) > 0:
#     with open(glob.glob(readme_glob)[0]) as f:
#         LONG_DESCRIPTION = f.read()

# else:
#     # Get the long description from the package's docstring
#     __import__(PACKAGENAME)
#     package = sys.modules[PACKAGENAME]
#     LONG_DESCRIPTION = package.__doc__

# # Store the package name in a built-in variable so it's easy
# # to get from other parts of the setup infrastructure
# builtins._ASTROPY_PACKAGE_NAME_ = PACKAGENAME

# # VERSION should be PEP440 compatible (http://www.python.org/dev/peps/pep-0440)
# VERSION = metadata.get('version', '0.24.0')

# # Indicates if this version is a release version
# RELEASE = 'dev' not in VERSION

# if not RELEASE:
#     VERSION += get_git_devstr(False)

# # Populate the dict of setup command overrides; this should be done before
# # invoking any other functionality from distutils since it can potentially
# # modify distutils' behavior.
# cmdclassd = register_commands(PACKAGENAME, VERSION, RELEASE)

# # Freeze build information in version.py
# generate_version_py(PACKAGENAME, VERSION, RELEASE,
#                     get_debug_option(PACKAGENAME))

# # Treat everything in scripts except README* as a script to be installed
# scripts = [fname for fname in glob.glob(os.path.join('scripts', '*'))
#            if not os.path.basename(fname).startswith('README')]


# # Get configuration information from all of the various subpackages.
# # See the docstring for setup_helpers.update_package_files for more
# # details.
# package_info = get_package_info()

# # Add the project-global data
# package_info['package_data'].setdefault(PACKAGENAME, [])
# package_info['package_data'][PACKAGENAME].append('data/*')
# package_info['package_data'][PACKAGENAME].append('pathprof/data/*')
# package_info['package_data'][PACKAGENAME].append('itudata/*.*')
# package_info['package_data'][PACKAGENAME].append('itudata/p.452-16/*')
# package_info['package_data'][PACKAGENAME].append('itudata/p.452-16/R-REC-P.452-16-201507/*')
# package_info['package_data'][PACKAGENAME].append('itudata/p.676-10/*')
# package_info['package_data'][PACKAGENAME].append('itudata/ra.769-2/*')

# # Define entry points for command-line scripts
# entry_points = {}
# entry_points['console_scripts'] = []

# if conf.has_section('entry_points'):
#     entry_point_list = conf.items('entry_points')
#     for entry_point in entry_point_list:
#         entry_points['console_scripts'].append('{0} = {1}'.format(
#             entry_point[0], entry_point[1]))

# # Define entry points for GUI scripts
# entry_points['gui_scripts'] = []

# if conf.has_section('entry_points_gui'):
#     entry_point_list = conf.items('entry_points_gui')
#     for entry_point in entry_point_list:
#         entry_points['gui_scripts'].append('{0} = {1}'.format(
#             entry_point[0], entry_point[1]))

# # Include all .c files, recursively, including those generated by
# # Cython, since we can not do this in MANIFEST.in with a "dynamic"
# # directory name.
# c_files = []
# for root, dirs, files in os.walk(PACKAGENAME):
#     for filename in files:
#         if filename.endswith('.c'):
#             c_files.append(
#                 os.path.join(
#                     os.path.relpath(root, PACKAGENAME), filename))
# package_info['package_data'][PACKAGENAME].extend(c_files)

# # Note that requires and provides should not be included in the call to
# # ``setup``, since these are now deprecated. See this link for more details:
# # https://groups.google.com/forum/#!topic/astropy-dev/urYO8ckB2uM

# setup(name=PACKAGENAME,
#       version=VERSION,
#       description=DESCRIPTION,
#       scripts=scripts,
#       install_requires=[s.strip() for s in metadata.get('install_requires', 'astropy').split(',')],
#       author=AUTHOR,
#       author_email=AUTHOR_EMAIL,
#       license=LICENSE,
#       url=URL,
#       long_description=LONG_DESCRIPTION,
#       cmdclass=cmdclassd,
#       zip_safe=False,
#       use_2to3=False,
#       entry_points=entry_points,
#       python_requires='>={}'.format(__minimum_python_version__),
#       **package_info
# )
