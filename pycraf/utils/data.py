# Licensed under a 3-clause BSD style license - see LICENSE.rst

""" This module contains helper functions for accessing, downloading, and
caching data files.
"""

import os
from ..utils.introspection import find_current_module, resolve_name

try:
    import pathlib
except ImportError:
    HAS_PATHLIB = False
else:
    HAS_PATHLIB = True

__all__ = ['get_pkg_data_filename']


def get_pkg_data_filename(data_name, package=None):
    """
    Retrieves a data file from the standard locations for the package and
    provides a local filename for the data.

    Parameters
    ----------
    data_name : str
        Name/location of the desired data file.  One of the following:

            * The name of a data file included in the source
              distribution.  The path is relative to the module
              calling this function.  For example, if calling from
              ``astropy.pkname``, use ``'data/file.dat'`` to get the
              file in ``astropy/pkgname/data/file.dat``.  Double-dots
              can be used to go up a level.  In the same example, use
              ``'../data/file.dat'`` to get ``astropy/data/file.dat``.

    package : str, optional
        If specified, look for a file relative to the given package, rather
        than the default of looking relative to the calling module's package.

    Raises
    ------
    IOError
        If problems occur writing or reading a local file.

    Returns
    -------
    filename : str
        A file path on the local file system corresponding to the data
        requested in ``data_name``.

    Examples
    --------

    This will retrieve the contents of the data file for the `astropy.wcs`
    tests::

        >>> from astropy.utils.data import get_pkg_data_filename
        >>> fn = get_pkg_data_filename('data/3d_cd.hdr',
        ...                            package='astropy.wcs.tests')
        >>> with open(fn) as f:
        ...     fcontents = f.read()
        ...

    """

    data_name = os.path.normpath(data_name)

    datafn = _find_pkg_data_path(data_name, package=package)
    if os.path.isdir(datafn):
        raise IOError("Tried to access a data file that's actually "
                      "a package data directory")
    elif os.path.isfile(datafn):  # local file
        return datafn
    else:
        raise IOError('File {} not found!'.format(datafn))


def _is_inside(path, parent_path):
    # We have to try realpath too to avoid issues with symlinks, but we leave
    # abspath because some systems like debian have the absolute path (with no
    # symlinks followed) match, but the real directories in different
    # locations, so need to try both cases.
    return os.path.abspath(path).startswith(os.path.abspath(parent_path)) \
        or os.path.realpath(path).startswith(os.path.realpath(parent_path))


def _find_pkg_data_path(data_name, package=None):
    """
    Look for data in the source-included data directories and return the
    path.
    """

    if package is None:
        module = find_current_module(1, True)

        if module is None:
            # not called from inside an astropy package.  So just pass name
            # through
            return data_name

        if not hasattr(module, '__package__') or not module.__package__:
            # The __package__ attribute may be missing or set to None; see
            # PEP-366, also astropy issue #1256
            if '.' in module.__name__:
                package = module.__name__.rpartition('.')[0]
            else:
                package = module.__name__
        else:
            package = module.__package__
    else:
        module = resolve_name(package)

    rootpkgname = package.partition('.')[0]

    rootpkg = resolve_name(rootpkgname)

    module_path = os.path.dirname(module.__file__)
    path = os.path.join(module_path, data_name)

    root_dir = os.path.dirname(rootpkg.__file__)
    if not _is_inside(path, root_dir):
        raise RuntimeError("attempted to get a local data file outside "
                           "of the {} tree.".format(rootpkgname))

    return path
