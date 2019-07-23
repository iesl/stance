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


'''
Combines all the test shards predictions into one file to score later 
'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", required=True)
    parser.add_argument("-n", "--num_partitions", required=True)
    args = parser.parse_args()

    partition_folder = os.path.join(args.exp_dir, "test_shards")
    test_prediction_filename = os.path.join(args.exp_dir, "test.predictions")


    with open(test_prediction_filename, 'w+') as f_out:
        total_lines = 0
        for i in range(int(args.num_partitions)):
            parititon_prediction_filename = os.path.join(partition_folder, "shard_{}.pred".format(str(i)))
            if(os.path.exists(parititon_prediction_filename)):
                with open(parititon_prediction_filename, 'r') as f_in:
                    all_lines =  f_in.readlines()
                total_lines += len(all_lines)
                for line in all_lines:
                    f_out.write(line)
        print(total_lines)