#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib.transforms import Affine2D
import mpl_toolkits.axisartist.floating_axes as floating_axes
from matplotlib.projections import PolarAxes
from mpl_toolkits.axisartist.grid_finder import MaxNLocator


def setup_earth_axes(fig, rect, theta_lim, h_lim, a_e, theta_scale):
    """
    From https://matplotlib.org/examples/axes_grid/demo_floating_axes.html
    """

    # rotate a bit for better orientation
    tr_rotate = Affine2D().translate(
        np.pi / 2 - np.mean(theta_lim) * theta_scale, 0
        )

    # scale degree to radians
    tr_scale = Affine2D().scale(theta_scale, 1)

    # treat heights
    tr_htrans = Affine2D().translate(0, +a_e)

    tr = tr_htrans + tr_scale + tr_rotate + PolarAxes.PolarTransform()

    grid_locator1 = MaxNLocator(5)
    grid_locator2 = MaxNLocator(5)

    grid_helper = floating_axes.GridHelperCurveLinear(
        tr, extremes=(*theta_lim, *h_lim),
        grid_locator1=grid_locator1,
        grid_locator2=grid_locator2,
        tick_formatter1=None,
        tick_formatter2=None,
        )

    ax = floating_axes.FloatingSubplot(fig, rect, grid_helper=grid_helper)

    fig.add_subplot(ax)

    # adjust axis
    ax.axis["left"].set_axis_direction("top")
    ax.axis["left"].label.set_text("Height [m]")
    ax.axis["left"].toggle(ticklabels=True, label=True)
    ax.axis["right"].toggle(ticklabels=False, label=False)
    ax.axis["right"].set_axis_direction("right")
    ax.axis["bottom"].toggle(ticklabels=True, label=True)
    ax.axis["bottom"].set_axis_direction("bottom")
    ax.axis["bottom"].major_ticklabels.set_axis_direction("bottom")
    ax.axis["bottom"].label.set_axis_direction("bottom")
    ax.axis["bottom"].label.set_text("Distance [km]")

    # create a parasite axes whose transData in RA, cz
    aux_ax = ax.get_aux_axes(tr)

    aux_ax.patch = ax.patch  # for aux_ax to have a clip path as in ax
    ax.patch.zorder = 0.9  # but this has a side effect that the patch is
    # drawn twice, and possibly over some other
    # artists. So, we decrease the zorder a bit to
    # prevent this.

    return ax, aux_ax
