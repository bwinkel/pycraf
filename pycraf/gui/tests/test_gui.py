#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# from __future__ import unicode_literals

import pytest
import numpy as np
from numpy.testing import assert_equal, assert_allclose
from astropy.tests.helper import assert_quantity_allclose, remote_data
from astropy import units as apu
from astropy.units import Quantity
# from astropy.utils.misc import NumpyRNGContext
# from astropy.utils.data import get_pkg_data_filename
from PyQt5 import QtCore, QtWidgets
from .. import gui


@remote_data(source='any')
@pytest.mark.usefixtures('srtm_handler')
def test_gui_startup_shows_pathgeometry(qtbot):
    # change download option to missing and test, if the results label
    # in geometry pane has correct values (need to wait for startup-timer
    # to fire)

    # app = QtWidgets.QApplication([])
    myapp = gui.PycrafGui()
    myapp.show()
    qtbot.addWidget(myapp)
    myapp.ui.srtmDownloadComboBox.setCurrentIndex(
        gui.SRTM_DOWNLOAD_MAPPING.index('missing')
        )
    with qtbot.waitSignal(myapp.my_pp_worker.result_ready[object, object], timeout=5000, raising=False):
        myapp.on_any_param_changed()


    # click in the Greet button and make sure it updates the appropriate label
    # qtbot.mouseClick(pygui.button_greet, QtCore.Qt.LeftButton)

    assert myapp.ui.ppRichTextLabel.text() == "Hello!"
