*********************************
Importing pycraf and subpackages
*********************************

Using pycraf in Python is as simple as::

    >>> import pycraf

Then one can access the subpackages via::

    pycraf.<subpackage>

and call associated functions, e.g.::

    pycraf.conversions.free_space_loss(...)

It is also possible to do::

    from pycraf import <subpackage>

Or, if you want to avoid typing a lot::

    from pycraf.<subpackage> as <abbreviation>
    from pycraf.<subpackage> import <function>

For example::

    from pycraf import conversions as cnv
    cnv.free_space_loss(...)

    from pycraf.conversions import free_space_loss
    free_space_loss(...)



