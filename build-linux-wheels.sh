#!/bin/bash
for v in {9..11}; do
    /opt/python/cp3${v}-cp3${v}/bin/python -m pip install Cython
    /opt/python/cp3${v}-cp3${v}/bin/python setup.py bdist_wheel
    auditwheel repair dist/*cp3${v}-cp3${v}*.whl
done