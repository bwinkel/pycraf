#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst
# Adapted from AstroPy-Project

from __future__ import (
    absolute_import, unicode_literals, division, print_function
    )


__all__ = ['ranged_quantity_input']

from collections import namedtuple
import numpy as np
from astropy.utils.decorators import wraps
from astropy.utils.compat import funcsigs

from astropy.units.core import UnitsError, add_enabled_equivalencies
from astropy.units import Unit, Quantity
# import ipdb


class RangedQuantityInput(object):

    @classmethod
    def as_decorator(cls, func=None, **kwargs):
        """
        A decorator for validating the units of arguments to functions.

        Unit specifications can be provided as keyword arguments to the decorator,
        or by using Python 3's function annotation syntax. Arguments to the decorator
        take precedence over any function annotations present.

        A `~astropy.units.UnitsError` will be raised if the unit attribute of
        the argument is not equivalent to the unit specified to the decorator
        or in the annotation.
        If the argument has no unit attribute, i.e. it is not a Quantity object, a
        `ValueError` will be raised.

        Where an equivalency is specified in the decorator, the function will be
        executed with that equivalency in force.

        Notes
        -----

        The checking of arguments inside variable arguments to a function is not
        supported (i.e. \*arg or \**kwargs).

        Examples
        --------

        Python 2 and 3::

            import astropy.units as u
            @u.ranged_quantity_input(myangle=(minval, maxval, u.arcsec))
            def myfunction(myangle):
                return myangle**2

        Using equivalencies::

            import astropy.units as u
            @u.ranged_quantity_input(myenergy=(minval, maxval, u.eV), equivalencies=u.mass_energy())
            def myfunction(myenergy):
                return myenergy**2

        """
        self = cls(**kwargs)
        if func is not None and not kwargs:
            return self(func)
        else:
            return self

    def __init__(self, func=None, **kwargs):
        self.kwargs = dict(kwargs)
        self.equivalencies = kwargs.pop('equivalencies', [])
        self.strip_input_units = kwargs.pop('strip_input_units', False)
        self.output_unit = kwargs.pop('output_unit', None)
        self.allow_none = kwargs.pop('allow_none', None)
        self.decorator_kwargs = kwargs

    def __call__(self, wrapped_function):

        # Extract the function signature for the function we are wrapping.
        wrapped_signature = funcsigs.signature(wrapped_function)

        # Define a new function to return in place of the wrapped one
        @wraps(wrapped_function)
        def wrapper(*func_args, **func_kwargs):
            # Bind the arguments to our new function to the signature of the original.
            bound_args = wrapped_signature.bind(*func_args, **func_kwargs)

            # Iterate through the parameters of the original signature
            for param in wrapped_signature.parameters.values():
                # We do not support variable arguments (*args, **kwargs)
                if param.kind in (funcsigs.Parameter.VAR_KEYWORD,
                                  funcsigs.Parameter.VAR_POSITIONAL):
                    continue
                # Catch the (never triggered) case where bind relied on a default value.
                if param.name not in bound_args.arguments and param.default is not param.empty:
                    bound_args.arguments[param.name] = param.default

                # Get the value of this parameter (argument to new function)
                arg = bound_args.arguments[param.name]

                # Get target unit, either from decorator kwargs or annotations
                if param.name in self.decorator_kwargs:
                    (
                        target_min, target_max, target_unit
                        ) = self.decorator_kwargs[param.name]
                else:
                    continue

                # If the target unit is empty, then no unit was specified so we
                # move past it
                if target_unit is not funcsigs.Parameter.empty:

                    # skip over None values, if desired
                    if arg is None and self.allow_none:
                        continue

                    try:
                        equivalent = arg.unit.is_equivalent(target_unit,
                                                  equivalencies=self.equivalencies)

                        if not equivalent:
                            raise UnitsError("Argument '{0}' to function '{1}'"
                                             " must be in units convertible to"
                                             " '{2}'.".format(param.name,
                                                     wrapped_function.__name__,
                                                     target_unit.to_string()))

                    # Either there is no .unit or no .is_equivalent
                    except AttributeError:
                        if hasattr(arg, "unit"):
                            error_msg = "a 'unit' attribute without an 'is_equivalent' method"
                        else:
                            error_msg = "no 'unit' attribute"
                        raise TypeError("Argument '{0}' to function '{1}' has {2}. "
                              "You may want to pass in an astropy Quantity instead."
                                 .format(param.name, wrapped_function.__name__, error_msg))

                    # test value range

                    if target_min is not None:
                        quantity = bound_args.arguments[param.name]
                        value = quantity.to(target_unit).value
                        if np.any(value < target_min):
                            raise AssertionError(
                                "Argument '{0}' to function '{1}' out of "
                                "range (allowed {2} to {3} {4}).".format(
                                    param.name, wrapped_function.__name__,
                                    target_min, target_max, target_unit,
                                    )
                                )

                    if target_max is not None:
                        quantity = bound_args.arguments[param.name]
                        value = quantity.to(target_unit).value
                        if np.any(value > target_max):
                            raise AssertionError(
                                "Argument '{0}' to function '{1}' out of "
                                "range (allowed {2} to {3} {4}).".format(
                                    param.name, wrapped_function.__name__,
                                    target_min, target_max, target_unit,
                                    )
                                )
                    if self.strip_input_units:
                        bound_args.arguments[param.name] = (
                            bound_args.arguments[param.name].to(
                                target_unit
                                ).value
                            )

            # Call the original function with any equivalencies in force.
            with add_enabled_equivalencies(self.equivalencies):
                # result = wrapped_function(*func_args, **func_kwargs)
                result = wrapped_function(
                    *bound_args.args, **bound_args.kwargs
                    )

            # import ipdb; ipdb.set_trace()
            if self.output_unit is not None:
                # # test, if return values are tuple-like
                try:
                    # make namedtuples work (as well as tuples)

                    if hasattr(result, '_fields'):
                        cls = result.__class__
                        return cls(*(
                            # r if u is None else Quantity(r, u)
                            # r if u is None else Quantity(r, u, subok=True)
                            r if u is None else r * u  # astropy bug
                            for r, u in zip(result, self.output_unit)
                            ))
                    else:
                        return tuple(
                            # r if u is None else Quantity(r, u)
                            # r if u is None else Quantity(r, u, subok=True)
                            r if u is None else r * u  # astropy bug
                            for r, u in zip(result, self.output_unit)
                            )
                except TypeError:

                    return (
                        result
                        if self.output_unit is None else
                        # Quantity(result, self.output_unit)
                        # Quantity(result, self.output_unit, subok=True)
                        result * self.output_unit  # astropy bug
                        )
            else:
                return result

        return wrapper

ranged_quantity_input = RangedQuantityInput.as_decorator
