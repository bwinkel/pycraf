#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

try:

    # Create the test function for self test
    from astropy.tests.runner import TestRunner
    test = TestRunner.make_test_runner_in(os.path.dirname(__file__))

except ImportError:

    def test():

        import warnings
        warnings.warn(
            'Package "astropy" is needed for using the "test()" function'
            )

test.__test__ = False
__all__ = ['test']
