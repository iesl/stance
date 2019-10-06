#!/usr/bin/env bash

set -exu

exp_dir_one=$1
exp_dir_two=$2

$PYTHON_EXEC -m main.eval.compare_models -e1 $exp_dir_one -e2 $exp_dir_two
