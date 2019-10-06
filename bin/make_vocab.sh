#!/usr/bin/env bash

set -exu

inputfile=$1
outputfile=$2
tokenizer=$3
min_count=$4

$PYTHON_EXEC -m main.setup.make_vocab -i $inputfile -o $outputfile -t $tokenizer -m $min_count
