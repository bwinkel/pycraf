from distutils.core import setup
from distutils.extension import Extension
from distutils.version import StrictVersion, LooseVersion
from Cython.Distutils import build_ext
import sys
import numpy

# Dependency checking
dependencies = [['numpy', '1.7'], ['scipy', '0.10'], ['pyfits', '3.0']]

for (pkg, minversion) in dependencies:
    try:
        m = __import__(pkg)
        if minversion is not None:
            if StrictVersion(m.__version__) < StrictVersion(minversion):
                if LooseVersion(m.__version__) < LooseVersion(minversion):
                    raise ValueError
                warnings.warn(
                    'Version', m.__version__,
                    'of package', pkg,
                    'might not be sufficient')
    except ImportError:
        print 'Package', pkg, 'not present.'
        sys.exit(1)
    except ValueError:
        print 'Package', pkg, 'has version', m.__version__
        print 'Version', minversion, 'required.'
        sys.exit(1)


# This is a list of files to install, and where
# (relative to the 'root' dir, where setup.py is)
# You could be more specific.

setup(
    name="pycraf",
    version="0.1",
    description="pycraf",
    author="Benjamin Winkel",
    author_email="bwinkel@mpifr.de",
    url="http://www.astro.uni-bonn.de/~bwinkel",
    # Name the folder where your packages live:
    # (If you have other packages (dirs) or modules (py files) then
    # put them into the package directory - they will be found
    # recursively.)
    packages=[
        'pycraf', 'pycraf.conversions', 'pycraf.atm', 'pycraf.protection'
        ],
    # 'package' package must contain files (see list above)
    # I called the package 'package' thus cleverly confusing the whole issue...
    package_dir={
        'pycraf': 'src',
        'pycraf.conversions': 'src/conversions',
        'pycraf.atm': 'src/atm',
        'pycraf.protection': 'src/protection',
        },
    cmdclass={'build_ext': build_ext},
    ext_modules=[
        ],
    # This dict maps the package name =to=> directories
    # It says, package *needs* these files.
    # note, wildcards are allowed
    package_data={
        'pycraf': [
            'atm/data/R-REC-P.676-10-201309_table1.csv',
            'atm/data/R-REC-P.676-10-201309_table2.csv',
            'protection/data/ra_769_table1_limits_continuum.csv',
            'protection/data/ra_769_table2_limits_spectroscopy.csv',
            ]
        },
    long_description="""pycraf ... the CRAF library.
    contains useful functions for the daily life"""
)
