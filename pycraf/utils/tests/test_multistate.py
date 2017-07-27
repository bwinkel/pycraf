#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pytest
from ...utils import MultiState


class TestMultiState:

    def setup(self):

        class MyState(MultiState):

            _attributes = ('foo', 'bar')
            foo = 'guido'
            bar = 1

        self.my_state = MyState

    def test_construction_guards(self):

        # _attributes must be a tuple!
        with pytest.raises(RuntimeError):

            class Tmp(MultiState):

                _attributes = ['bla']
                bla = 1

        # all attributes must be assigned with a default value
        with pytest.raises(RuntimeError):

            class Tmp(MultiState):

                _attributes = ('bla')

        # must not try to use "set" as an attribute (would override method)
        with pytest.raises(RuntimeError):

            class Tmp(MultiState):

                _attributes = ('set')
                set = 1

    def test_getter(self):

        assert self.my_state.foo == 'guido'
        assert self.my_state.bar == 1


    def test_setter(self):

        # must not set attributes directly
        with pytest.raises(RuntimeError):
            self.my_state.foo = 'bar'

        # must not instantiate class
        with pytest.raises(RuntimeError):
            self.my_state()

        with self.my_state.set(foo='foo'):
            assert self.my_state.foo == 'foo'
            assert self.my_state.bar == 1

        with self.my_state.set(bar='missing'):
            assert self.my_state.foo == 'guido'
            assert self.my_state.bar == 'missing'

        with self.my_state.set(foo='bar', bar='always'):
            assert self.my_state.foo == 'bar'
            assert self.my_state.bar == 'always'

        foo = self.my_state.foo
        self.my_state.set(foo='foo')
        assert self.my_state.foo != foo
        assert self.my_state.foo == 'foo'

        self.my_state.set(foo=foo)
        assert self.my_state.foo == foo
        assert self.my_state.foo != 'foo'

    def test_context_manager(self):

        foo = self.my_state.foo
        bar = self.my_state.bar

        with self.my_state.set(foo='bar', bar='always'):
            assert self.my_state.foo == 'bar'
            assert self.my_state.bar == 'always'

        assert self.my_state.foo == foo
        assert self.my_state.bar == bar

    # def test_validation(self):

    #     with pytest.raises(TypeError):
    #         with self.my_state.set(1):
    #             pass
