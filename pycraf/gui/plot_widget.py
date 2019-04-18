#!/usr/bin/python
# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtGui, QtWidgets
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar
    )
from matplotlib.figure import Figure
from matplotlib.colorbar import make_axes
import sys


__all__ = ['CustomToolbar', 'PlotWidget']


class CustomToolbar(NavigationToolbar):
    '''
    Customize Navigation Toolbar.

    Note:
    One can use the following code snipped (e.g., in __init__) to find
    out which Actions are available (and to remove some, if necessary).

    for i, da in enumerate(defaultactions):
        print i, da.toolTip(), da.statusTip(), str(da.data().toString())

    The following is available by default:
    allActions = [
        'Reset', 'Back', 'Forward', 'Pan', 'Zoom',
        'Configure', 'Save', 'Edit'
        ]
    (plus some separators)

    '''

    message = QtCore.pyqtSignal(str, name='message')

    def __init__(self, plotCanvas, parent, coordinates=True):
        super(CustomToolbar, self).__init__(plotCanvas, parent, coordinates)
        # defaultactions = self.actions()

        # wanted_actions = ['Reset', 'Pan', 'Zoom', 'Save']

        # Note, the coordinate display field is not an action,
        # but is invisible if separators were removed
        # we couldn't find out, how to obtain it by any other
        # means than to include all elements without a tooltip :-(
        # wanted_actions = [
        #     da for da in defaultactions
        #     if any([
        #         wa in da.toolTip() or len(da.toolTip()) < 1
        #         for wa in wanted_actions
        #         ])
        #     ]

        # for da in self.actions()[::-1]:
        #     if da not in wanted_actions:
        #         self.removeAction(da)

        self._msghandler = None

    def set_msghandler(self, p):
        '''
        Install a custom message handler, see 'set_message'.
        '''

        self._msghandler = p

    def set_message(self, s):
        '''
        Write custom messages into the locLabel.

        (The locLabel is where coordinates usually appear).

        Note: for the lazy folks, one can register a msghandler (e.g., once
        before all subsequent calls to set_message) that then performs
        some transformations. This is used in online plotter to install
        a function that converts between frequency and velocity etc.
        '''

        # TODO, can we replace this with the new API? Or is it touching
        # some internals of the NavigationToolbar parent?
        # self.emit(QtCore.SIGNAL("message"), s)
        self.message.emit(s)

        if self._msghandler is not None:
            s = self._msghandler(s)

        if self.coordinates:
            self.locLabel.setText(s.replace(', ', '\n'))


class PlotWidget(QtWidgets.QWidget):
    '''
    Matplotlib widget that encapsulates FigureCanvasQTAgg for convenience.

    Note: If you use the twin-x/y feature, for some strange reason, if one
    creates the twins before one actually plots data (which is the case here),
    the autoscaling doesn't work anymore. So, in that case, you need to do
    set_x/ylim manually!
    '''
    sharex = None
    sharey = None

    def __init__(
            self, parent=None,
            dpi=100, subplotx=1, subploty=1,
            sharex=False, sharey=False,
            twinx=False, twiny=False,
            plottername='A plot',
            do_cbars=False, **cbar_kwargs
            ):

        super(PlotWidget, self).__init__(parent)

        self.dpi = dpi
        self.subplotx = subplotx
        self.subploty = subploty
        self.twinx = twinx
        self.twiny = twiny
        self._plottername = plottername
        self.do_cbars = do_cbars
        self.cbar_kwargs = cbar_kwargs
        self.create()
        self.set_shared(sharex, sharey)

    @QtCore.pyqtProperty(str)
    def plottername(self):
        return self._plottername

    @plottername.setter
    def set_plottername(self, value):
        self._plottername = value

    def create(self):

        self._fig = Figure(dpi=self.dpi)
        self._canvas = FigureCanvas(self._fig)
        self._canvas.setParent(self)
        self._canvas.setFocusPolicy(QtCore.Qt.ClickFocus)
        self._canvas.setFocus()

        axes_kwargs = {}

        if self.subplotx != 1 or self.subploty != 1:

            self.numplots = self.subplotx * self.subploty
            self._axes = []
            self._cbaxes = []
            self._twinaxes_x = []
            self._twinaxes_y = []

            for a in range(self.numplots):

                self._axes.append(
                    self._fig.add_subplot(
                        self.subploty, self.subplotx, a + 1,
                        **axes_kwargs
                        )
                    )
                if self.do_cbars:
                    cax, kw = make_axes(self._axes[-1], **self.cbar_kwargs)
                    self._cbaxes.append(cax)

                if self.twinx:
                    self._twinaxes_x.append(self._axes[-1].twiny())  # not a typo!

                if self.twiny:
                    self._twinaxes_y.append(self._axes[-1].twinx())  # not a typo!

            # put main ax into foreground, otherwise the primary coords are
            # the second axes' ones
            for a in range(self.numplots):
                self._axes[a].set_zorder(100)

        else:

            self.numplots = 1
            self._axes = self._fig.add_subplot(1, 1, 1, **axes_kwargs)
            if self.do_cbars:
                cax, kw = make_axes(self._axes, **self.cbar_kwargs)
                self._cbaxes = cax

        self.mpl_toolbar = CustomToolbar(self._canvas, self)
        vbox = QtWidgets.QVBoxLayout()
        vbox.addWidget(self._canvas)
        vbox.addWidget(self.mpl_toolbar)
        self.setLayout(vbox)

    def set_shared(self, sharex, sharey):

        if self.sharex == sharex and self.sharey == sharey:

            return

        self.sharex = sharex
        self.sharey = sharey

        if self.numplots > 1:

            if self.sharex:
                self._axes[0].get_shared_x_axes().join(*self._axes)
                if self.twinx:
                    self._twinaxes_x[0].get_shared_x_axes().join(
                        *self._twinaxes_x
                        )

            if self.sharey:
                self._axes[0].get_shared_y_axes().join(*self._axes)
                if self.twiny:
                    self._twinaxes_y[0].get_shared_y_axes().join(
                        *self._twinaxes_y
                        )

    @QtCore.pyqtProperty(object)
    def axes(self):

        return self._axes

    @QtCore.pyqtProperty(object)
    def caxes(self):

        return self._cbaxes

    @QtCore.pyqtProperty(object)
    def twin_axes_x(self):

        return self._twinaxes_x

    @QtCore.pyqtProperty(object)
    def twin_axes_y(self):

        return self._twinaxes_y

    @QtCore.pyqtProperty(object)
    def canvas(self):

        return self._canvas

    @QtCore.pyqtProperty(object)
    def figure(self):

        return self._fig

    @QtCore.pyqtProperty(object)
    def toolbar(self):

        return self._canvas.toolbar

    def clear_history(self):
        self.mpl_toolbar.update()
        self.mpl_toolbar.push_current()

    def store_figure_data(self):

        import numpy as np

        print(self.plottername)

        try:
            enumiter = enumerate(self._axes)
        except TypeError:
            enumiter = enumerate([self._axes])

        datadict = {}
        for idx, ax in enumiter:

            xlabel = ax.xaxis.get_label().get_text()
            ylabel = ax.yaxis.get_label().get_text()
            axname = 'subplot_{}'.format(idx + 1)
            datadict[axname] = {}

            for line in ax.get_lines():

                llabel = line.get_label()
                if '_line' in llabel:
                    # don't use unlabeled lines
                    continue
                xdata = line.get_xdata()
                ydata = line.get_ydata()

                line_dtype = np.dtype([
                    (xlabel, np.float64), (ylabel, np.float64)
                    ])
                xyarray = np.array(list(zip(xdata, ydata)), dtype=line_dtype)

                datadict[axname][llabel] = xyarray

        if sys.version_info >= (3, 0):
            import pickle
        else:
            import cPickle as pickle
        pickle.dump(datadict, open('/tmp/plot.cpickle', 'wb'), protocol=2)
