#!/bin/bash

# attempt to load modules (for cluster computers)
if command -v module &> /dev/null
then
    module load GCC/10.3.0 OpenMPI/4.1.1 SciPy-bundle/2021.05 PyTorch/1.11.0-CUDA-11.3.1
fi

# create venv if necessary
if ! [[ -f ".env/bin/activate" ]]
then
    python -m venv .env
fi

# make sure pip is up to date
.env/bin/python -m pip install --upgrade pip -q

# install any missing dependencies
.env/bin/python -m pip install -r requirements.txt -q
if [[ -f "difuze-requirements.txt" ]]
then
    .env/bin/python -m pip install -r difuze-requirements.txt -q
fi

# activate the venv
source .env/bin/activate
