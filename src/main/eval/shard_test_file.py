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

import random
import os
import sys
import shutil
import argparse


def split_test_set(filepath, num_shards):
    '''
    Shards the data in file to multiple files, ensuring that all the lines with the same query are grouped together

    param num_shards: number of shards
    '''
    map_query_2_lines = get_map_query_2_lines(filepath)
    
    data_shard_folder = os.path.join(os.path.split(filepath)[0], "test_shards")
    if(os.path.exists(data_shard_folder)):
        shutil.rmtree(data_shard_folder)
    os.makedirs(data_shard_folder)

    for counter, (query, list_of_entities) in enumerate(map_query_2_lines.items()):
        shard = counter % int(num_shards)

        shard_filename = os.path.join(data_shard_folder, "shard" + "_" + str(shard))

        with open(shard_filename, 'a+') as f:
            for entities in list_of_entities:
                f.write('\t'.join(entities) + '\n')


def get_map_query_2_lines(filepath):
    '''
    Gets a dictionary of query to all the lines that include the query

    param filepath: filepath to data file 
    return: dictionary of query to lines 
    '''
    map_query_2_lines = {}

    with open(filepath, 'r') as file:        
        for line in file.readlines():
            entities = line.strip('\n').split('\t')
            query = entities[0]
            if query in map_query_2_lines.keys():
                map_query_2_lines[query].append(entities)
            else:
                map_query_2_lines[query] = [entities]

    return map_query_2_lines


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_file", required=True)
    parser.add_argument("-s", "--num_shards", required=True)
    args = parser.parse_args()

    split_test_set(args.test_file, args.num_shards)



