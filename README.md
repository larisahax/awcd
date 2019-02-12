

AWCD is a novel non-parametric community detection technique based on adaptive weights. Weights are recovered using an iterative procedure based on statistical test of "no gap". The procedure is fully adaptive, allows overlapping communities, it is numerically feasible and applicable for large graphs, demonstrates very good performance on artifical and real world networks.

Setup
-----------

Build the latest development version from source:

    git clone https://github.com/larisahax/awcd.git
    cd awcd
  Create, activate virtualenv and install the required packages.
  
    virtualenv -p python3 env && source env/bin/activate
    pip install -r awcd/requirements.txt
  
 Setup and build the C lib for sparce matrix multiplication.  
 
    cd awcd/src/cython_c_wrapper
    make
 
 Now all is set. Run the example file to see the AWCD result on a Stochastic Block Model.
 
    cd ../../
    python main.py

Requirements
-----
python 3.6 

see the awcd/requirements.txt for the packages.

