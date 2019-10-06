#!/usr/bin/env bash

set -exu

testfile=$1
numshards=$2

$PYTHON_EXEC -m main.eval.shard_test_file -t $testfile -s $numshards
