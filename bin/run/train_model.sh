#!/usr/bin/env bash

set -exu

config=$1
grid_search=${2-False}

$PYTHON_EXEC -m main.train.train_model -c $config -g $grid_search
