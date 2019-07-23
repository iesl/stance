"""
Copyright (C) 2019 University of Massachusetts Amherst.
This file is part of "stance"
http://github.com/iesl/stance
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import os
import torch
import argparse


from main.objects.Vocab import Vocab
from main.objects.Config import Config

'''
Setsup a bash script to evaluate the best model in parallel with multiple GPU's on a slurm manager
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", required=True)
    parser.add_argument("-n", "--num_shards", required=True)
    parser.add_argument("-g", "--gpu", required=True)
    args = parser.parse_args()


    test_output_dir = os.path.join(args.exp_dir, "test_shards_pred")
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir) 
    
    bash_script = os.path.join(args.exp_dir, "parallel_test.sh")

    error_dir = os.path.join(args.exp_dir, "error_dir")
    if(not os.path.exists(error_dir)):
        os.makedirs(error_dir)

    output_dir = os.path.join(args.exp_dir, "output_dir")
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    with open(bash_script, 'w+') as f:
        f.write("#!/usr/bin/env bash \n")

        for i in range(int(args.num_shards)):
            counter = i
            error_file = os.path.join(error_dir, "error" + str(counter))
            output_file = os.path.join(output_dir, "output" + str(counter)) 

            command = "sbatch --partition={} --gres=gpu:1 --error {} --output {} --mem=15GB bin/run/eval_model.sh \
                            {} True {} \n".format(args.gpu, error_file, output_file, args.exp_dir, int(counter))
            f.write(command + '\n')

