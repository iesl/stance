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

import sys, os
import argparse
import numpy as np
from numpy import *


from sklearn.metrics import average_precision_score


def eval_map(list_of_list_of_labels,list_of_list_of_scores,randomize=True):
    """Compute Mean Average Precision
    Given a two lists with one element per test example compute the
    mean average precision score.
    The i^th element of each list is an array of scores or labels corresponding
    to the i^th training example.
    :param list_of_list_of_labels: Binary relevance labels. One list per example.
    :param list_of_list_of_scores: Predicted relevance scores. One list per example.
    :return: the mean average precision
    """
    np.random.seed(19)
    assert len(list_of_list_of_labels) == len(list_of_list_of_scores)
    aps = []
    for i in range(len(list_of_list_of_labels)):
        if randomize == True:
            perm = np.random.permutation(len(list_of_list_of_labels[i]))
            list_of_list_of_labels[i] = np.asarray(list_of_list_of_labels[i])[perm]
            list_of_list_of_scores[i] = np.asarray(list_of_list_of_scores[i])[perm]

        where_are_NaNs = isnan(list_of_list_of_scores[i])
        list_of_list_of_scores[i][where_are_NaNs] = -999999999999999
        if sum(list_of_list_of_labels[i]) > 0:
            aps.append(average_precision_score(list_of_list_of_labels[i],
                                               list_of_list_of_scores[i]))

    return sum(aps) / len(aps)


def eval_hits_at_k(list_of_list_of_labels,
                   list_of_list_of_scores,
                   k=10,
                   randomize=True,
                   oracle=False,
                   ):
    """Compute Hits at K
    Given a two lists with one element per test example compute the
    mean average precision score.
    The i^th element of each list is an array of scores or labels corresponding
    to the i^th training example.
    All scores are SIMILARITIES.
    :param list_of_list_of_labels: Binary relevance labels. One list per example.
    :param list_of_list_of_scores: Predicted relevance scores. One list per example.
    :param k: the number of elements to consider
    :param randomize: whether to randomize the ordering
    :param oracle: break ties using the labels
    :return: the mean average precision
    """
    np.random.seed(19)
    assert len(list_of_list_of_labels) == len(list_of_list_of_scores)
    aps = []
    for i in range(len(list_of_list_of_labels)):
        zpd = zip(list_of_list_of_scores[i],list_of_list_of_labels[i])
        sorted_zpd =sorted(zpd, key=lambda x: x[0], reverse=True)
        list_of_list_of_labels[i] = [x[1] for x in sorted_zpd]
        list_of_list_of_scores[i] = [x[0] for x in sorted_zpd]
        labels_topk = list_of_list_of_labels[i][0:k]
        if sum(list_of_list_of_labels[i]) > 0:
            hits_at_k = sum(labels_topk) * 1.0 / min(k, sum(list_of_list_of_labels[i]))
            aps.append(hits_at_k)

    return sum(aps) / len(aps)

def load(filename):
    """Load the labels and scores for Hits at K evaluation.
    Loads labels and model predictions from files of the format:
    Query \t Example \t Label \t Score
    :param filename: Filename to load.
    :return: list_of_list_of_labels, list_of_list_of_scores
    """
    result_labels = []
    result_scores = []
    current_block_name = ""
    current_block_scores = []
    current_block_labels = []
    with open(filename,'r') as fin:
        for line in fin:
            splt = line.strip().split("\t")
            block_name = splt[0]
            block_example = splt[1]
            example_label = int(splt[2])
            example_score = float(splt[3])
            if block_name != current_block_name and current_block_name != "":
                result_labels.append(current_block_labels)
                result_scores.append(current_block_scores)
                current_block_labels = []
                current_block_scores = []
            current_block_labels.append(example_label)
            current_block_scores.append(example_score)
            current_block_name = block_name
    result_labels.append(current_block_labels)
    result_scores.append(current_block_scores)
    return result_labels,result_scores

all_list_of_list_of_labels = []
all_list_of_list_of_scores = []

def add_shard_file(filename):
    list_of_list_of_labels,list_of_list_of_scores = load(filename)
    all_list_of_list_of_labels.extend(list_of_list_of_labels)
    all_list_of_list_of_scores.extend(list_of_list_of_scores)

def score():
    """ Given a file of predictions, compute all metrics
    
    :param prediction_filename: TSV file of predictions
    :param model_name: Name of the model
    :param dataset_name: Name of the dataset
    :return: 
    """
    counter = 0
    scores = ""
    map_score = eval_map(all_list_of_list_of_labels, all_list_of_list_of_scores)
    scores +="MAP\t{}\n".format(map_score)
    scores += "Hits@1\t{}\n".format(eval_hits_at_k(all_list_of_list_of_labels, all_list_of_list_of_scores,k=1,oracle=False))
    scores += "Hits@10\t{}\n".format(eval_hits_at_k(all_list_of_list_of_labels, all_list_of_list_of_scores,k=10,oracle=False))
    scores += "Hits@50\t{}\n".format(eval_hits_at_k(all_list_of_list_of_labels, all_list_of_list_of_scores,k=50,oracle=False))
    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", required=True)
    args = parser.parse_args()

    test_shards_folder = os.path.join(args.exp_dir, "test_shards_pred")

    shard_num = 0
    while True:
        shard_file_name = os.path.join(test_shards_folder, "shard_{}.pred".format(str(shard_num)))

        if os.path.exists(shard_file_name):
            add_shard_file(shard_file_name)
            shard_num += 1
        else:
            break

    test_scores_file = os.path.join(args.exp_dir, "test_scores_shards.json")
    with open(test_scores_file, 'w+') as fout:
        s = score()
        fout.write(s)