#!/bin/bash
# from https://github.com/scikit-hep/boost-histogram/tree/v0.6.0/.ci

set -e -x

# Collect the pythons
pys=(/opt/python/*/bin)

# Print list of Python's available
echo "All Pythons: ${pys[@]}"

# Filter out Python 3.4
# pys=(${pys[@]//*34*/})
pys=( $( printf '%s\0' "${pys[@]}" | grep -z $py_whl ) )

# Print list of Python's being used
echo "Using Pythons: ${pys[@]}"

# Compile wheels
for PYBIN in "${pys[@]}"; do
    # "${PYBIN}/pip" install -r /io/pip-requirements-dev
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/$package_name-*.whl; do
    auditwheel repair --plat $PLAT "$whl" -w /io/wheelhouse/
done

# Install packages and test
for PYBIN in "${pys[@]}"; do
    "${PYBIN}/python" -m pip install $package_name --no-index -f /io/wheelhouse
    # "${PYBIN}/pytest" /io/tests
done
