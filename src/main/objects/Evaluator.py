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

import numpy as np
import time
import string
import torch
import os

from main.objects.Config import Config
from main.objects.Batcher import Batcher
from main.objects.Vocab import Vocab
from main.objects.Tokenizer import Char
from main.objects.Scorer import Scorer
from main.objects.Writer import Writer

'''
Evaluator object which scores the model and write the model predictions 
'''
class Evaluator(object):
    def __init__(self, config, vocab, tokenizer, input_type, exp_dir, list_k, labeled_file=None, output_file=None):
        '''
        param config: configuration to use for evaluation 
        param vocab: vocabulary to use 
        param tokenizer: tokenizer to use 
        param input_type: input type of either dev/test
        param exp_dir: experiment directory to save output
        param list_k: list of k to evaluate hits@k
        param labeled_file: labeled file to use for labels (default is specified in config )
        param output_file: output file to use for writing prediction 
        '''
        self.batcher = Batcher(config, input_type, tokenizer, labeled_file)
        self.input_type = input_type
        self.list_k = list_k
        
        if self.input_type == "dev":
            self.best_dev_score = 0
            self.score_filename = os.path.join(exp_dir, "dev_scores.json")
            self.best_model_filename = os.path.join(exp_dir, "best_model")

        elif self.input_type == "test":
            if output_file is not None:
                self.test_file = output_file
            else:
                self.test_file = os.path.join(exp_dir, "test.predictions")
            self.score_filename = os.path.join(exp_dir, "test_scores.json")
            self.writer = Writer(self.test_file)

        self.output_file = None
        if output_file:
            self.output_file = output_file

        self.score = True
        if self.output_file is not None and "shard" in self.output_file:
            self.score = False


    def evaluate(self, model, train_num_batches):
        '''
        Evaluates the model by scoring it and writing its predictions 

        param train_num_batches: number of batches the model has trained on 
        '''
        if self.score == True:
            scorer = Scorer(self.list_k, self.score_filename, train_num_batches)

        # Score the model batch by batch 
        for qry_tk, qry, cnd_tk, cnd, labels, end_block in self.batcher.get_dev_test_batches():
            scores = model.score_dev_test_batch(qry_tk, cnd_tk)
            scores = list(scores.cpu().data.numpy().squeeze(1))

            # Adds the batch of scores to Scorer
            if self.score == True:
                scorer.add_batch_pred_scores(qry_tk, scores, labels, end_block)

            # Adds the batch of predictions to Writer
            if self.input_type == "test":
                self.writer.add_batch_pred_lab(qry, cnd, labels, scores)

        # Calculate the scores and save if best so far
        if self.score == True:
            map_score = scorer.calc_scores()
            if self.input_type == "dev":
                if map_score > self.best_dev_score:
                    torch.save(model, self.best_model_filename)
                self.best_dev_score = map_score



