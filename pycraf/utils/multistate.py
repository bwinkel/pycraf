#!/usr/bin/env python
# -*- coding: utf-8 -*-


__all__ = ['MultiState']


class _MultiMeta(type):

    def __new__(cls, *args):

        if not isinstance(args[2]['_attributes'], tuple):
            raise RuntimeError(
                '"_attributes" must be a tuple!'
                )

        if 'set' in args[2]['_attributes']:
            raise RuntimeError(
                'Option "set" not in list of allowed attributes!'
                )

        for attr in args[2]['_attributes']:

            if attr not in args[2]:
                raise RuntimeError(
                    'Must set default value for option "{}" during subclass '
                    'creation!'.format(attr)
                    )

        return super().__new__(cls, *args)

    def __setattr__(cls, attr, value):

        if attr == '__annotations__':
            # apparently numpy's autodoc is trying to store this in the class
            super().__setattr__(attr, value)
        else:
            raise RuntimeError(
                'Setting attributes directy is not allowed. Use "set" method!'
                )

    def __repr__(cls):
        if hasattr(cls, '__repr__'):
            return getattr(cls, '__repr__')()
        else:
            return super().__repr__()

    def __str__(cls):
        if hasattr(cls, '__str__'):
            return getattr(cls, '__str__')()
        else:
            return super().__repr__()


class MultiState(object, metaclass=_MultiMeta):
    '''
    Multi state subclasses are used to manage global items.

    `MultiState` is a so-called Singleton, which means that no instances
    can be created. This way, it is guaranteed that only one `MultiState`
    entity exists. Thus it offers the possibilty to handle a "global"
    state, because from anywhere in the code, one could query the
    current state and react to it. One usecase would be to
    temporarily allow downloading of files, otherwise being forbidden.

    The `MultiState` class is not useful on its own, but you have to
    subclass it::

        >>> from pycraf.utils import MultiState
        >>> class MyState(MultiState):
        ...
        ...     # define list of allowed attributes
        ...     _attributes = ('foo', 'bar')
        ...
        ...     # set default values
        ...     foo = 1
        ...     bar = "guido"

    It is mandatory to provide a tuple of allowed attributes and to
    create these instance attributes and assign default values.
    During class creation, this will be validated, for convenience.

    After defining the state class, one can do the following::

        >>> MyState.foo
        1
        >>> MyState.set(foo=2)
        <MultiState MyState>
        >>> MyState.foo
        2

    The `set` method returns a context manager,::

        >>> with MyState.set(foo="dave", bar=10):
        ...     print(MyState.foo, MyState.bar)
        dave 10

        >>> print(MyState.foo, MyState.bar)
        2 guido

    which makes it possible to temporarily change the state object and
    go back to the original value, once the `with` scope has ended.

    Note, that one cannot set the attributes directly (to ensure
    that the validation method is always run)::

        >>> MyState.foo = 0
        Traceback (most recent call last):
        ...
        RuntimeError: Setting attributes directy is not allowed. Use "set" method!

    Subclasses will generally override `validate` to convert from any
    of the acceptable inputs (such as strings) to the appropriate
    internal objects::

        class MyState(MultiState):

            # define list of allowed attributes
            _attributes = ('foo', 'bar')

            # set default values
            foo = 1
            bar = "guido"

            @classmethod
            def validate(cls, **kwargs):
                assert isinstance(kwargs['foo'], int)
                # etc.
                return kwargs

    Notes
    -----
    This class was adapted from the `~astropy.utils.state.ScienceState` class.
    '''

    _attributes = tuple()

    def __init__(self):
        raise RuntimeError('This class is a singleton.  Do not instantiate.')

    @classmethod
    def set(cls, *, _do_validate=True, **kwargs):
        """
        Set the current science state value.
        """

        for k in kwargs:

            if k not in cls._attributes:
                raise ValueError(
                    'Option "{}" not in list of allowed attributes!'.format(k)
                    )

        class _Context(object):
            def __init__(self, parent, attrs):
                self._parent = parent
                self._values = {}
                for k in attrs:
                    self._values[k] = getattr(parent, k)

            def __enter__(self):
                pass

            def __exit__(self, type, value, tb):
                cls.hook(**self._values)
                for k, v in self._values.items():
                    self._parent.__class__.__class__.__setattr__(
                        self._parent, k, v
                        )

            def __repr__(self):
                return ('<MultiState {0}>'.format(self._parent.__name__))

        ctx = _Context(cls, cls._attributes)
        if _do_validate:
            kwargs = cls.validate(**kwargs)
        cls.hook(**kwargs)
        for k, v in kwargs.items():
            cls.__class__.__class__.__setattr__(cls, k, v)

        return ctx

    @classmethod
    def validate(cls, **kwargs):
        '''
        Validate the keyword arguments and return the (converted) kwargs
        dictionary.

        You should override this method if you want to enable validation
        of your attributes.

        Notes
        -----
        One doesn't need to validate the following things, as it is already
        take care of by the `MultiState` class:

        - Check that each argument is assigned with a default.
        - Check that the kwargs keys are in _attributes tuple.
        '''

        return kwargs

    @classmethod
    def hook(cls, **kwargs):
        '''
        A hook which is called everytime when attributes are about to change.

        You should override this method if you want to enable pre-processing
        or monitoring of your attributes. For example, one could use this
        to react to attribute changes::

            >>> from pycraf.utils import MultiState

            >>> class MyState(MultiState):
            ...
            ...     _attributes = ('foo', 'bar')
            ...     foo = 1
            ...     bar = "guido"
            ...
            ...     @classmethod
            ...     def hook(cls, **kwargs):
            ...         if 'bar' in kwargs:
            ...             if kwargs['bar'] != cls.bar:
            ...                 print('{} about to change: {} --> {}'.format(
            ...                     'bar', kwargs['bar'], cls.bar
            ...                     ))
            ...                 # do stuff ...

            >>> _ = MyState.set(bar="david")
            bar about to change: david --> guido
            >>> _ = MyState.set(bar="david")
            >>> _ = MyState.set(bar="guido")
            bar about to change: guido --> david

            >>> with MyState.set(bar="david"):
            ...     pass
            bar about to change: david --> guido
            bar about to change: guido --> david
        '''

        pass
