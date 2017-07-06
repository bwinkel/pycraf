************
Installation
************

Requirements
============

pycraf has the following strict requirements:

- `Python <http://www.python.org/>`_ 3.5 or 3.6

- `Numpy`_ |minimum_numpy_version| or later

- `pytest`_ 2.8 or later

- `h5py <http://h5py.org/>`_: To read/write path geometry data created by
  :function:`~pycraf.pathprof.height_profile_data` from/to HDF5 files.

- `scipy`_: Used in various routines.

- `matplotlib <http://matplotlib.org/>`_ 1.5 or later: To provide plotting functionality that `pycraf.pathprof.helper` enhances.

- `setuptools <https://pythonhosted.org/setuptools/>`_: Used for the package installation.

Installing pycraf
==================

Using pip
-------------

To install pycraf with `pip <http://www.pip-installer.org/en/latest/>`_, simply run::

    pip install pycraf

.. note::

    You will need a C compiler (e.g. ``gcc`` or ``clang``) to be installed (see
    `Building from source`_ below) for the installation to succeed.

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
    <http://www.pip-installer.org/en/1.2.1/other-tools.html#using-pip-with-the-user-scheme>`_.

    Alternatively, if you intend to do development on other software that uses
    pycraf, such as an affiliated package, consider installing pycraf into a
    :ref:`virtualenv<using-virtualenv>`.

    We recommend to use a Python distribution, such as `Anaconda <https://www.continuum.io/downloads>`_, especially, if you are on :ref:`windows_install`.

    Do **not** install pycraf or other third-party packages using ``sudo``
    unless you are fully aware of the risks.


.. _windows_install:

Installation on Windows
~~~~~~~~~~~~~~~~~~~~~~~

Note, for Windows machines we provide a binary wheel (Python 3.5+ only).
However, the `pyproj <https://pypi.python.org/pypi/pyproj?>`_ package is a
dependency and unfortunately, the official
`pyproj <https://pypi.python.org/pypi/pyproj?>`_ repository on PyPI contains
only the sources. You can download a
suitable wheel from `Christoph Gohlke's package site <http://www.lfd.uci.edu/~gohlke/pythonlibs/#pyproj>`_. Then use::

    pip install [path-to-wheel]\pyproj‑*.whl

If you're using `Anaconda <https://www.continuum.io/downloads>`_
(recommended), it gets much simpler::

    conda install -c conda-forge pyproj

.. _testing_installed_pycraf:

Testing an installed pycraf
----------------------------

The easiest way to test your installed version of pycraf is running
correctly is to use the :ref:`~pycraf.test()` function::

    import pycraf
    pycraf.test()

The tests should run and print out any failures, which you can report at
the `pycraf issue tracker <http://github.com/bwinkel/pycraf/issues>`_.

.. note::

    This way of running the tests may not work if you do it in the
    pycraf source distribution.  See :ref:`sourcebuildtest` for how to
    run the tests from the source code directory, or :ref:`running-tests`
    for more details.

