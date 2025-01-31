#!/bin/bash

conda create -n b_star python=3.10 -y
conda activate b_star

conda install conda-forge::pinocchio=3.3.1 -y
pip install "numpy<2" scipy cvxpy coptpy pyyaml loguru prettytable
pip install ./opt_term/cpp
