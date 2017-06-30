# Introduction #

- *Version*: 0.2
- *Authors*: Benjamin Winkel

# Disclaimer #
`pycraf` is still in the early-development stage. While much of the
functionality is already working as intended, the API is not yet stable.
Nevertheless, we kindly invite you to use and test the library and we are
grateful for feedback. Note, that the documentation is still missing (but at
least docstrings are provided).

# Purpose #
`pycraf` is a Python package that provides functions and procedures for various
tasks in spectrum-management compatibility studies. A typical example would be
to calculate the interference levels at a radio telescope produced from a radio
broadcasting tower.

# Features #

* Full implementation of ITU-R Rec. P.452 that allows to calculate path
  attenuation for the distance between interferer and victim service. Supports
  to load NASA's [Shuttle Radar Topography Mission
  (SRTM)](https://www2.jpl.nasa.gov/srtm/) data for height-profile generation.
* Full implementation of ITU-R Rec. P.676, which provides two atmospheric
  models to calculate the attenuation for paths through Earth's atmosphere.
* Provides various antenna patterns necessary for compatibility studies (e.g.,
  RAS, IMT, fixed-service links).
* Functions to convert power flux densities, field strengths, transmitted and
  received powers at certain distances and frequencies into each other.

# License #

`pycraf` is published under GPL v3, an open-source license.

For some of the functionality provided in pycraf, data files provided by the
ITU are necessary. For example, the atmospheric model in the pycraf.atm
subpackage implements the algorithm described in ITU-R Recommendation P.676.
Annex 1 of this Recommendation makes use of spectroscopic information of the
oxygen and water vapour lines given in Tables 1 and 2 of P.676.

ITU kindly gave us permission to include data files into pycraf that are
distributed with the Recommendations on the ITU servers. This makes it possible
to just use pycraf without the need to manually download necessary data files.
However, these data files are not free for commercial use. For details, please
see the LICENSE.ITU file in the `itudata` directory of the package.

We are very grateful for the kind support from ITU study groups and ITU's legal
department.


# Usage #

## Installation ##

The easiest way to install pycraf is via `pip`:

```
pip install pycraf
```

The installation is also possible from source. Download the tar.gz-file,
extract (or clone from GitHub) and simply execute

```
python setup.py install
```

### SRTM data ###

To make full use of the path attenuation calculations provided by `pycraf`
(implements ITU-R Rec. P.452), we recommend to use NASA's [Shuttle Radar
Topography Mission (SRTM)](https://www2.jpl.nasa.gov/srtm/) data for height-
profile generation. `pycraf` can work with so-called `.hgt` files, a very
simple binary format. Each `.hgt` file, a so-called tile, just contains
1201x1201 16-bit integers. From the file naming scheme, one can infer the
associated coordinates. Most tiles contain one square-degree.

Unfortunately, we cannot provide SRTM data as part of the package, due to the
large file sizes and legal reasons. But once you downloaded the necessary tiles
(all or only a subset appropriate for your region), simply define the
environment variable `$SRTMDATA`, let it point to the folder containing the
tiles, and `pycraf` will find the files when it is imported from Python.


## Dependencies ##

We kept the dependencies as minimal as possible. The following packages are
required:
* `numpy 1.8` or later
* `astropy 1.1` or later
* `pyproj 1.9` or later

## Examples and Documentation ##

Will follow asap. Sorry for the inconvenience!

## Who do I talk to? ##

If you encounter any problems or have questions, do not hesitate to raise an
issue or make a pull request. Moreover, you can contact the devs directly:

* <bwinkel@mpifr.de>
