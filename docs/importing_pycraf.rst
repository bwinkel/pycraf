*********************************
Importing pycraf and subpackages
*********************************

Using pycraf in Python is as simple as::

    >>> import pycraf

Then one can access the subpackages via::

    >>> pycraf.<subpackage>  # doctest: +SKIP

and call associated functions, e.g.::

    >>> pycraf.conversions.free_space_loss(...)  # doctest: +SKIP

It is also possible to do::

    >>> from pycraf import <subpackage>  # doctest: +SKIP

Or::

    >>> from pycraf.<subpackage> as <abbrev>  # doctest: +SKIP
    >>> from pycraf.<subpackage> import <function>  # doctest: +SKIP

For example::

    >>> from pycraf import conversions as cnv
    >>> from pycraf.conversions import free_space_loss

********************************
Getting started with subpackages
********************************

Because different subpackages have very different functionality, each subpackage has its own
getting started guide. These can be found by browsing the sections listed in the :ref:`user-docs`.

You can also look at docstrings for a
particular package or object, or access their documentation using the
`~pycraf.utils.misc.find_api_page` function. For example, ::

    >>> from pycraf import find_api_page  # doctest: +SKIP
    >>> from pycraf.pathprof import PathProp  # doctest: +SKIP
    >>> find_api_page(PathProp)  # doctest: +SKIP

will bring up the documentation for the `~pycraf.pathprof.PathProp` class
in your browser.
