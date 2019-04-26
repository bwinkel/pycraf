
:tocdepth: 3

#####################
pycraf Documentation
#####################

Welcome to the pycraf documentation. The pycraf Python package provides
functions and procedures for various tasks related to spectrum-management
compatibility studies. This includes an implementation of `ITU-R
Recommendation P.452-16 <https://www.itu.int/rec/R-REC-P.452-16-201507-I/en>`_
that allows to calculate path attenuation for the distance between an
interferer and the victim service.

.. _getting-started:

***************
Getting Started
***************

.. toctree::
   :maxdepth: 1

   install
   importing_pycraf
   Tutorials <http://nbviewer.jupyter.org/github/bwinkel/pycraf/blob/master/notebooks/>
   pathprof/working_with_srtm

******************
User Documentation
******************

Available modules
-----------------

.. toctree::
   :maxdepth: 1

   conversions/index
   atm/index
   pathprof/index
   antenna/index
   protection/index
   geospatial/index
   satellite/index
   geometry/index
   mc/index
   utils/index
   gui/index

.. automodapi:: pycraf
    :no-heading:
    :no-main-docstr:



***************
Project details
***************

.. toctree::
   :maxdepth: 1

   license

***************
Acknowledgments
***************

This code makes use of the excellent work provided by the
`Astropy <http://www.astropy.org/>`__ community. pycraf uses the Astropy package and also the
`Astropy Package Template <https://github.com/astropy/package-template>`__
for the packaging.
`ITU-R Recommendation P.452-16
<https://www.itu.int/rec/R-REC-P.452-16-201507-I/en>`__


.. |apu| replace:: `astropy.units`
