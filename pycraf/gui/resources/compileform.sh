#!/bin/sh
ANACONDA3=${HOME}/local/anaconda3/
alias apyuic="PYTHONPATH= LDFLAGS= CFLAGS= LD_LIBRARY_PATH= QT5DIR= $ANACONDA3/bin/pyuic5"
alias apyrcc="PYTHONPATH= LDFLAGS= CFLAGS= LD_LIBRARY_PATH= QT5DIR= $ANACONDA3/bin/pyrcc5"

apyuic main_form.ui -o main_form.py
apyrcc resources.qrc -o resources_rc.py
sed -i 's/import resources_rc/from pycraf.gui.resources import resources_rc/g' *form.py
