#!/usr/bin/env bash

set -exu

exp_dir=$1
is_parallel=${2-False}
shard=${3-False}

$PYTHON_EXEC -m main.eval.test_model -e $exp_dir -p $is_parallel -s $shard
