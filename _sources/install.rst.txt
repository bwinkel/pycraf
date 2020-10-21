************
Installation
************

Requirements
============

pycraf has the following strict requirements:

- `Python <http://www.python.org/>`__ 3.6 or later

- `setuptools <https://pythonhosted.org/setuptools/>`__: Used for the package
  installation.

- `Cython <http://cython.org/>`__ 0.29 or later

- `NumPy <http://www.numpy.org/>`__ 1.14 or later

- `SciPy <https://scipy.org/>`__: 0.19 or later

- `astropy <http://www.astropy.org/>`__: 3.0 or later

- `pytest <https://pypi.python.org/pypi/pytest>`__ 2.6 or later


There are a few optional packages, which are necessary for some functionality:

- `matplotlib <http://matplotlib.org/>`__ 1.5 or later: To provide plotting
  functionality that `~pycraf.pathprof` enhances.

- `pyproj <https://pypi.python.org/pypi/pyproj>`__ 2.0 or later: This is a
  requirement for the `~pycraf.geospatial` package.

- `sgp4 <https://pypi.python.org/pypi/sgp4>`__ 2.0 or later: This is a
  requirement for the `~pycraf.satellite` package.

Older versions of these packages may work, but no support will be provided.

Installing pycraf
==================

There are various ways to install `pycraf`. The easiest and cleanest approach
would be to use the `Anaconda/Miniconda Python distribution
<https://www.anaconda.com/distribution/>`_, because it allows to download
a binary package, which is well-tested against all dependency packages.

Using Anaconda
--------------
After installing Anaconda, one can run the `conda package manager
<https://docs.conda.io/en/latest/>`_::

    conda install pycraf -c conda-forge

.. note::

    It is always a good idea to keep different projects separated and conda
    allows to easily create virtual environments. To set one up for `pycraf`::

        conda create -n pycraf-env python=3.6 pycraf

    and to use it::

        conda activate pycraf-env

    (In conda versions before 4.6, one has to source the activate shell
    script instead. See `conda manual <https://docs.conda.io/en/latest/>`_.)

Using pip
-------------

To install pycraf with `pip <http://www.pip-installer.org/en/latest/>`__, simply run

.. code-block:: bash

    pip install pycraf

.. note::

    You may need a C compiler (``gcc``) with OpenMP support to be installed
    for the installation to succeed.

.. note::

    Use the ``--no-deps`` flag if you already have dependency packages
    installed, since otherwise pip will sometimes try to "help" you
    by upgrading your installation, which may not always be desired.

.. note::

    If you get a ``PermissionError`` this means that you do not have the
    required administrative access to install new packages to your Python
    installation.  In this case you may consider using the ``--user`` option
    to install the package into your home directory.  You can read more
    about how to do this in the `pip documentation
    <http://www.pip-installer.org/en/1.2.1/other-tools.html#using-pip-with-the-user-scheme>`__.

    We recommend to use a Python distribution, such as `Anaconda
    <https://www.continuum.io/downloads>`_, especially, if you are on
    :ref:`windows_install`.

    Do **not** install pycraf or other third-party packages as
    Administrator/Root unless you are fully aware of the risks.

.. _source_install:

Installation from source
------------------------

There are two options, if you want to build pycraf from sources. Either, you
install the tar-ball (`*.tar.gz` file) from `PyPI
<https://pypi.python.org/pypi/pycraf>`_ and extract it to the directory of
your choice, or, if you always want to stay up-to-date, clone the git
repository:

.. code-block:: bash

    git clone https://github.com/bwinkel/pycraf

Then go into the pycraf source directory and run:

.. code-block:: bash

    python setup.py install

Again, consider the ``--user`` option or even better use a python distribution
such as `Anaconda <https://www.continuum.io/downloads>`_ to avoid messing up
the system-wide Python installation.

.. note::

    If you use `Anaconda` and want to install the `sgp4` and `pyproj`
    packages, you'll have to use a different channel

    .. code-block:: bash

        conda install -c conda-forge sgp4 pyproj


.. _windows_install:

Installation on Windows
-----------------------

Note that for Windows machines we provide a binary wheel (Python 3.5+ only).
However, the `pyproj <https://pypi.python.org/pypi/pyproj>`_ package is a
dependency and unfortunately, the official
`pyproj <https://pypi.python.org/pypi/pyproj>`__ repository on PyPI contains
only the sources. You can download a
suitable wheel from `Christoph Gohlke's package site
<http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyproj>`__. Then use

.. code-block:: bash

    pip install [path-to-wheel]/pyproj‑*.whl

If you're using `Anaconda <https://www.continuum.io/downloads>`__
(recommended), it gets much simpler

.. code-block:: bash

    conda install -c conda-forge pyproj
    pip install pycraf

.. note::

    If you are desperate, you can install pycraf from source even on Windows.
    You'll need to install a suitable C-compiler; <see here
    <https://matthew-brett.github.io/pydagogue/python_msvc.html#visual-studio-versions-used-to-compile-distributed-python-binaries>`__. The pycraf
    package needs Python 3.5 or later, which means VC++ Version 14 is
    mandatory. The easiest way to obtain it, is by installing the
    `Visual C++ 2015 Build Tools
    <http://landinghub.visualstudio.com/visual-cpp-build-tools>`__ which is
    "only" 4 GBytes large...


.. _macos_install:

Installation on MacOS
---------------------

Installation on MacOS can be a bit tricky, because the standard C compiler
does not support OpenMP. We provide wheels on PyPI, such that you can

.. code-block:: bash

    pip install pycraf

however, you need to have the LLVM C compiler (see below), otherwise you'll
likely get an error message that a library (such as "libgomp") is not
found, when you import pycraf in Python.

Also, if you want to install from source, you must have a C compiler. There
are basically two options, using LLVM or the gcc suite.

LLVM
~~~~

.. code-block:: bash

    brew update
    brew install llvm

    export CC="/usr/local/opt/llvm/bin/clang"
    export LDFLAGS="-L/usr/local/opt/llvm/lib -Wl,-rpath,/usr/local/opt/llvm/lib"
    export CPPFLAGS="-I/usr/local/opt/llvm/include"

Then follow the instructions in :ref:`source_install`.

gcc
~~~

.. code-block:: bash

    brew install gcc6  # or gcc7
    brew link --overwrite gcc@6  # or gcc@7

Then follow the instructions in :ref:`source_install`.

.. note::

    The MacOS wheel, which we provide on PyPI (for pip installation)
    was built using LLVM. So it may happen that you run into binary
    incompatibilities if you use a different compiler suite on your computer.
    In such cases it may be necessary to build pycraf from source using
    your own compiler. Sometimes even different compiler versions
    (e.g. gcc 6.3 instead of gcc 6.4) can lead to problems.
    Please write a ticket, if you run into trouble.

.. note::

    Again, if you're on Anaconda, things get (often) much simpler:

     .. code-block:: bash

        conda install -c conda-forge openmp


.. _testing_installed_pycraf:

Testing an installed pycraf
=============================

The easiest way to test if your installed version of pycraf is running
correctly, is to use the `~pycraf.test()` function::

    import pycraf
    pycraf.test()

To run the tests for one sub-package, e.g., `conversions`, only::

    import pycraf
    pycraf.test('conversions')

The tests should run and print out any failures, which you can report at
the `pycraf issue tracker <http://github.com/bwinkel/pycraf/issues>`__.

.. note::

    This way of running the tests may not work if you do it in the
    pycraf source distribution directory.

.. note::

    By default, the `test` function will skip over tests that require
    data from the internet. One can include them by::

        import pycraf
        pycraf.test(remote_data='any')

    This will *always* download SRTM data (few tiles only) to test the
    auto-download functionality! Do this only, if you can afford the
    network traffic.

If you prefer testing on the command line and usually work with the source
code, you can also do

.. code-block:: bash

    python setup.py test

    # to run tests from a sub-package
    python setup.py test -P conversions

    # include tests, which need to download data (will slow down tests)
    python setup.py test --remote-data=any

.. _srtm_data:

Using SRTM data
==================

To make full use of the path attenuation calculations provided by pycraf,
you will need to use NASA's Shuttle Radar Topography Mission
`(SRTM) data <https://www2.jpl.nasa.gov/srtm/>`__ for height-profile
generation. Please see :ref:`working_with_srtm` for further details.
