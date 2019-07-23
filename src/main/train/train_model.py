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
import os
import sys
from shutil import copyfile, copytree
import argparse

import torch
import torch.optim as optim


from main.objects.Config import Config
from main.objects.Batcher import Batcher
from main.objects.Evaluator import Evaluator
from main.objects.Scorer import Scorer


from main.eval.test_model import test_model


from main.utils.util import make_directory, save_dict_to_json, make_exp_dir
from main.utils.model_helper import get_tokenizer, get_vocab, get_model



def train_model(config, exp_dir):
    """ Train based on the given config, model / dataset
    
    :param config: config object
    :param dataset_name: name of dataset
    :param model_name: name of model
    :return: 
    """
    torch.manual_seed(config.random_seed)

    tokenizer, max_len_token = get_tokenizer(config)
    vocab = get_vocab(config, tokenizer, max_len_token)
    model = get_model(config, vocab, max_len_token)
    model = model.cuda()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate,
                           weight_decay=config.l2penalty)   

    train_batcher = Batcher(config, 'train', tokenizer)
    dev_evaluator = Evaluator(config, vocab, tokenizer, 'dev', exp_dir, list_k=[5])

    for train_num_batches, (qry_tk, pos_tk, neg_tk) in enumerate(train_batcher.get_train_batches()):
        optimizer.zero_grad()
        model.train()
        loss = model.compute_loss(qry_tk, pos_tk, neg_tk)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()

        if train_num_batches == config.num_minibatches:
            break
        if train_num_batches > 0 and train_num_batches % config.eval_every_minibatch == 0:
            model.eval()
            dev_evaluator.evaluate(model, train_num_batches)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file", required=True)
    parser.add_argument("-g", "--grid_search", default="False")
    args = parser.parse_args()

    # Set up the config
    config = Config(args.config_file)
    config.update_dataset()

    # For non grid search, must set up exp dir
    if args.grid_search == "False":
        exp_dir = make_exp_dir(config.dataset_name, config.model_name, config.tokenizer_name)  
        copytree(os.path.join(os.environ['SED_ROOT'], 'src'), os.path.join(exp_dir, 'src'))  
        config.save_config(exp_dir)
    else:
        exp_dir = os.path.split(args.config_file)[0]

    train_model(config, exp_dir)
