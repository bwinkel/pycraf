#!/bin/sh
pyuic5 main_form.ui -o main_form.py
pyrcc5 resources.qrc -o resources_rc.py
sed -i 's/import resources_rc/from pycraf.gui.resources import resources_rc/g' *form.py
