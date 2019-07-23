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

import datetime
import torch
import codecs
import subprocess
import json
import os


def __filter_json(the_dict):
    '''
    Filters the dictionary so that only the string, floats, ints, and lists are stored 

    param the_dict: dictionary to filter 
    return: filtered dictionary 
    '''
    res = {}
    for k in the_dict.keys():
        if type(the_dict[k]) is str or type(the_dict[k]) is float or type(the_dict[k]) is int or type(the_dict[k]) is list:
            res[k] = the_dict[k]
        elif type(the_dict[k]) is dict:
            res[k] = __filter_json(the_dict[k])
    return res

def save_dict_to_json(the_dict,the_file):
    '''
    Saves the dictionary to file 

    param the_dict: dictionary to save 
    param the_file: file to save dictionary 
    '''
    with open(the_file, 'a+') as fout:
        fout.write(json.dumps(__filter_json(the_dict)))
        fout.write("\n")

def make_directory(dir_name):
    '''
    Makes directory if it doesn't exist 

    param dir_name: directory name 
    '''
    if (not os.path.exists(dir_name)):
        os.makedirs(dir_name)


def make_exp_dir(dataset_name, model_name, tokenizer_name):
    '''
    Makes experiment directory which includes timestamp to ensure distinct 

    param dataset_name: name of dataset 
    param model_name: name of model 
    param tokenizer_name: name of tokenizer
    return: experiment directory name 
    '''
    now = datetime.datetime.now()
    ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                            now.second)

    exp_dir = os.path.join("exp_out", dataset_name, model_name, tokenizer_name, ts)
    make_directory(exp_dir)

    return exp_dir