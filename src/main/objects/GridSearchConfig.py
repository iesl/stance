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

import itertools
import json
import os
import random
import string

from main.objects.Config import Config

'''
GridSearchConfig objects that stores parameters for doing grid search over hyperparameters
'''
class GridSearchConfig(Config):
    def __init__(self,filename=None):
        '''
        Initializes the hyperparameters to default values 

        param filename: filename containing the hyperparameters to use for training 
        '''
        super(Config, self).__init__()
        self.grid_search_fields = []
        self.learning_rate = 0.0001
        self.l2penalty = 10.0
        self.vocab_file = None
        self.train_file = None
        self.dev_file = None
        self.test_file = None

        self.num_minibatches = 40000
        self.eval_every_minibatch = 100
        self.train_batch_size = 32
        self.dev_test_batch_size = 16

        self.max_num_unigram = 40
        self.max_num_char = 200
        self.max_num_unigramuc = 40

        self.embedding_dim = 100
        self.rnn_hidden_dim = 100
        self.random_seed = 2524
        self.bidirectional = True
        self.cnn_num_layers = 3
        self.filter_counts = [25, 25, 25, 25]
        self.increasing = False

        self.dropout_rate = 0.2
        self.clip = 0.25

        self.dataset_name = "dataset"
        self.model_name = "model"
        self.tokenizer_name = "tokenizer"
        self.random = random.Random(self.random_seed)

        if filename:
            self.__dict__.update(json.load(open(filename)))


    def to_json(self):
        '''
        Stores all the parameters into a json 
        '''
        res = {}
        for k in self.__dict__.keys():
            if type(self.__dict__[k]) is str or type(self.__dict__[k]) is float or type(self.__dict__[k]) is int:
                res[k] = self.__dict__[k]
        return json.dumps(res)

    def save_config(self,exp_dir):
        '''
        Saves the parameters used for training in experiment directory 

        param exp_dir: experiment directory to save configuration 
        '''
        with open(os.path.join(exp_dir,"config.json"), 'w') as fout:
            fout.write(self.to_json())
            fout.write("\n")

    def copy_config(self):
        '''
        Copies the config to use for different grid search configurations
        '''
        c = Config()
        c.__dict__ = self.__dict__.copy()
        return c

    def configs_iter(self):
        '''
        Iterates through the grid search fields and creates a configuration for each possible configuration of grid search paramter values 

        return: grid search configuration 
        '''
        gs_vals = [self.__dict__[k] for k in self.grid_search_fields]
        gs_params = itertools.product(*gs_vals)
        for param_setting in gs_params:
            new_config = self.copy_config()
            for param,setting in zip(self.grid_search_fields,param_setting):
                new_config.__dict__[param] = setting
            yield new_config

    def update_dataset(self):
        '''
        Updates the dataset appropriately by looking at the training filename 
        '''
        self.dataset_name = '/'.join(str.split(self.train_file, '/')[1:2])
