# Not used by this project, but you can use this script to build a wheel
#!/bin/bash
set -e -x

export PYHOME=/home
cd ${PYHOME}

# /opt/python/cp37-cp37m/bin/pip install twine cmake
# ln -s /opt/python/cp37-cp37m/bin/cmake /usr/bin/cmake

# # Compile wheels
# for PYBIN in /opt/python/cp3*/bin; do
#     "${PYBIN}/pip" wheel /io/ -w wheelhouse/
#     "${PYBIN}/python" /io/setup.py sdist -d /io/wheelhouse/
# done

# Bundle external shared libraries into the wheels and fix naming
for whl in dist/*.whl; do
    auditwheel repair "$whl" -w dist/
done