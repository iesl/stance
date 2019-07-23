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
import json
import os

import argparse

def find_best_config(experiment_dir):    
    '''
    Loops through configuration in experiment directory and finds the configuration with the highest dev score 
    '''
    best_config = 0
    best_score = 0
    config_counter = 0
    cwd = os.getcwd()

    while True:
        config_dir = os.path.join(experiment_dir, "config_" + str(config_counter))
        if(os.path.exists(config_dir)):

            dev_scores_json = os.path.join(config_dir, "dev_scores.json")
            if(os.path.exists(dev_scores_json)):
                with open(dev_scores_json) as f:

                    all_lines = f.readlines()
                    for line in all_lines:
                        score_json = json.loads(line)
                        score = score_json["map"]
        
                        if(float(score) > best_score):
                            best_score = float(score)
                            best_config = float(config_counter)
        else:
            break
        
        config_counter += 1 

    print("Best Config: ", best_config)
    print("Best Score: ", best_score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", required=True)
    args = parser.parse_args()

    find_best_config(args.exp_dir)