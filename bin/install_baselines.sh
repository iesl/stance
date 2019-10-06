git clone https://github.com/mblondel/soft-dtw.git
cd soft-dtw 
export PYTHONPATH=$(pwd):$PYTHONPATH
make cython
python setup.py build 
python setup.py install
make
cd ..
