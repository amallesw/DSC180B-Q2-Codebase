#!/bin/bash

# Run preprocessing for the dartmouth dataset
python preprocessing/preprocessing.py --dataset_name=dartmouth

# Run preprocessing for the fer dataset
python preprocessing/preprocessing.py --dataset_name=fer