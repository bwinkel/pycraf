******
pycraf
******

- *Version:* 1.1.0
- *Author:* Benjamin Winkel, Marta Bautista & Federico Di Vruno
- *User manual:* `stable <https://bwinkel.github.io/pycraf/>`__ |
  `developer <https://bwinkel.github.io/pycraf/latest/>`__

.. image:: https://img.shields.io/pypi/v/pycraf.svg
    :target: https://pypi.python.org/pypi/pycraf
    :alt: PyPI tag

.. image:: https://img.shields.io/badge/license-GPL-blue.svg
    :target: https://www.github.com/bwinkel/pycraf/blob/master/COPYING
    :alt: License

.. image:: https://zenodo.org/badge/doi/10.5281/zenodo.1244192.svg
    :target: https://doi.org/10.5281/zenodo.1244192
    :alt: Zenodo DOI

The pycraf Python package provides functions and procedures for
various tasks in spectrum-management compatibility studies. A typical example
would be to calculate the interference levels at a radio telescope produced
from a radio broadcasting tower.

Releases are `registered on PyPI <http://pypi.python.org/pypi/pycraf>`_,
and development is occurring at the
`project's github page <http://github.com/bwinkel/pycraf/>`_.

Project Status
==============

.. image:: https://travis-ci.org/bwinkel/pycraf.svg?branch=master
    :target: https://travis-ci.org/bwinkel/pycraf
    :alt: Pycrafs's Travis CI Status

.. image:: https://ci.appveyor.com/api/projects/status/tj7swn14t6bek3jr?svg=true
    :target: https://ci.appveyor.com/project/bwinkel/pycraf
    :alt: Pycrafs's AppVeyor CI Status

.. image:: https://coveralls.io/repos/github/bwinkel/pycraf/badge.svg?branch=master
    :target: https://coveralls.io/github/bwinkel/pycraf?branch=master
    :alt: Pycrafs's Coveralls Status

`pycraf` is still in the early-development stage. While much of the
functionality is already working as intended, the API is not yet stable.
Nevertheless, we kindly invite you to use and test the library and we are
grateful for feedback. Note, that work on the documentation is still ongoing.

Features
========

- Full implementation of `ITU-R Rec. P.452-16 <https://www.itu.int/rec/R-REC-P.452-16-201507-I/en>`_ that allows to calculate path
  attenuation for the distance between interferer and victim service. Supports
  to load NASA's `Shuttle Radar Topography Mission (SRTM) <https://www2.jpl.nasa.gov/srtm/>`_ data for height-profile generation.
- Full implementation of `ITU-R Rec. P.676-10 <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_, which provides two atmospheric
  models to calculate the attenuation for paths through Earth's atmosphere.
- Provides various antenna patterns necessary for compatibility studies (e.g.,
  RAS, IMT, fixed-service links).
- Functions to convert power flux densities, field strengths, transmitted and
  received powers at certain distances and frequencies into each other.

Usage
=====

Examples and Documentation
--------------------------

We provide `an online documentation and API reference <https://bwinkel.github.io/pycraf/>`_. Furthermore, you can find tutorials and HowTos in
the `notebooks <http://nbviewer.jupyter.org/github/bwinkel/pycraf/blob/master/notebooks/>`_
directory on the pycraf repository.

Testing
-------

After installation (see below) you can test, if everything works as intended::

    import pycraf

    pycraf.test()

By default, the `test` function will skip over tests that require
data from the internet. One can include them by::

    pycraf.test(remote_data='any')

This will *always* download SRTM data (few tiles only) to test the
auto-download functionality! Do this only, if you can afford the network
traffic.


License
=======

Several licenses apply; see the `license directory <https://github.com/bwinkel/pycraf/blob/master/licenses/>`_ in the repository. The pycraf Python package
itself is published under `GPL v3 <https://github.com/bwinkel/pycraf/blob/master/licenses/COPYING>`_, an open-source license.

For some of the functionality provided in pycraf, data files provided by the
ITU are necessary. For example, the atmospheric model in the *pycraf.atm*
subpackage implements the algorithm described in `ITU-R Recommendation P.676 <https://www.itu.int/rec/R-REC-P.676-10-201309-S/en>`_.
Annex 1 of this Recommendation makes use of spectroscopic information of the
oxygen and water vapour lines given in Tables 1 and 2 of P.676. Another
example are the radiometeorological data files that are distributed alongside
`ITU-R Rec. P.452-16 <https://www.itu.int/rec/R-REC-P.452-16-201507-I/en>`_

ITU kindly gave us permission to include data files into pycraf that are
distributed with the Recommendations on the ITU servers. This makes it possible
to just use pycraf without the need to manually download necessary data files.
However, these data files are not free for commercial use. For details, please
see the `LICENSE.ITU <https://www.github.com/bwinkel/pycraf/blob/master/licenses/LICENSE.ITU>`_ file.

Some of the examples/images in the pycraf documentation and tutorial notebooks
make use of `Copernicus <https://www.copernicus.eu/en>`_ data. For these, the
conditions in `COPERNICUS.EU <https://www.github.com/bwinkel/pycraf/blob/master/COPERNICUS.EU>`_ apply.

Since pycraf uses the `Astropy Package Template <https://github.com/astropy/package-template>`_ for packaging, we also refer to the associated  `license <https://github.com/bwinkel/pycraf/blob/master/licenses/LICENSE_ASTROPY_PACKAGE_TEMPLATE.rst>`_.


Installation
============

We strongly recommend to use the `Anaconda Python distribution
<https://www.anaconda.com/distribution/>`_, as it allows to download `pycraf`
binaries for all major platforms (Linux, OSX, Windows). After installing
Anaconda/Miniconda, one can use the `conda` package manager to install it::

    conda install pycraf -c conda-forge

Of course, it is always a good idea to do this in its own environment, such
that you don't mess up with your standard environment::

    conda create -n pycraf-env python=3.6 pycraf

Otherwise, the easiest way to install pycraf is via pip::

    pip install pycraf

The installation is also possible from source. Download the tar.gz-file,
extract (or clone from GitHub) and simply execute::

    python setup.py install

Dependencies
------------

We kept the dependencies as minimal as possible. The following packages are
required:

* Python 3.6 or later
* setuptools
* cython 0.29 or later
* numpy 1.14 or later
* astropy 3.0 or later
* scipy 0.19 or later
* pytest 2.6 or later

The following packages are optional, and you will need them for certain
features and to build the docs:

* matplotlib 2.0 or later; for some plot helpers
* pyproj 2.0 or later; for the `geospatial` subpackage
* sgp4 2.0 or later; for the `satellite` subpackage

For further details, we refer to the online documention `installation
instructions <https://bwinkel.github.io/pycraf/install.html>`_. It also
includes some hints for running pycraf on Windows or MacOS. Older versions
of the packages may work, but no support will be provided.

SRTM data
---------

To make full use of the path attenuation calculations provided by pycraf
(implements `ITU-R Rec. P.452 <https://www.itu.int/rec/R-REC-P.452-16-201507-I/en>`_), we recommend to use NASA's
`Shuttle Radar Topography Mission (SRTM) <https://www2.jpl.nasa.gov/srtm/>`_
data for height-profile generation. pycraf can work with so-called *.hgt*
files, a very simple binary format. Each *.hgt* file, a so-called tile, just
contains 1201x1201 16-bit integers. From the file naming scheme, one can infer
the associated coordinates. Most tiles contain one square-degree.

Unfortunately, we cannot provide SRTM data as part of the package, due to the
large file sizes and legal reasons. But once you downloaded the necessary
tiles (all or only a subset appropriate for your region), simply define the
environment variable *SRTMDATA*, let it point to the folder containing the
tiles, and pycraf will find the files when it is imported from Python.

On windows::

    set SRTMDATA=C:\[path-to-srtm]\

On Linux/MacOS (sh-like)::

    export SRTMDATA=[path-to-srtm]/

There is also the possibility to change the path to the SRTM directory during
run-time (see documentation).

Acknowledgments
===============
We are very grateful for the kind support from ITU study groups and ITU's
legal department.

This code is makes use of the excellent work provided by the
`Astropy <http://www.astropy.org/>`_ community. pycraf uses the Astropy package and also the
`Astropy Package Template <https://github.com/astropy/package-template>`_
for the packaging.

Who do I talk to?
=================

If you encounter any problems or have questions, do not hesitate to raise an
issue or make a pull request. Moreover, you can contact the devs directly:

- *bwinkel@mpifr.de*
