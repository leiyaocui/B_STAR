#!/bin/bash

conda create -n b_star python=3.10 -y
conda activate b_star

conda install conda-forge::pinocchio -y
pip install "numpy<2" scipy cvxpy coptpy pyyaml loguru prettytable
