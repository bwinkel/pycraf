#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys


def main(args=None):
    '''Launch pycraf GUI.'''
    if args is None:
        args = sys.argv[1:]

    from pycraf import gui
    gui.start_gui()


if __name__ == '__main__':
    main()
