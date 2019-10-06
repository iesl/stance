export SED_ROOT=`pwd`
export PYTHONPATH=$SED_ROOT/src:$PYTHONPATH
cd soft-dtw
export PYTHONPATH=$(pwd):$PYTHONPATH
make
cd ..
export PYTHON_EXEC=python
