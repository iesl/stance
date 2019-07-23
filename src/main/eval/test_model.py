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

import torch
import os
import argparse


from main.objects.Config import Config
from main.objects.Batcher import Batcher
from main.objects.Evaluator import Evaluator
from main.objects.Scorer import Scorer

from main.utils.model_helper import get_tokenizer, get_vocab, get_model


def test_model(config, vocab, tokenizer, exp_dir, test_label_file=None, test_output_file=None):
    '''
    Test the model on test set 

    param config: configuration for training 
    param vocab: vocabulary for training
    param tokenizer: tokenizer to training
    param exp_dir: experiment directory that contains trained model to evaluate 
    param test_label_file: file with test data (default used will be the one specified in config)
    param test_output_file: file to write test predictions (default will be the one specified in config )  
    '''
    test_evaluator = Evaluator(config, vocab, tokenizer, 'test', exp_dir, list_k=[1, 10, 50], \
        labeled_file=test_label_file, output_file=test_output_file)
    model = torch.load(os.path.join(exp_dir, "best_model"))

    model.eval()
    test_evaluator.evaluate(model, train_num_batches=-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_dir", required=True)
    parser.add_argument("-p", "--is_parallel", required=True)
    parser.add_argument("-s", "--shard")
    parser.add_argument("-t", "--test_file")
    args = parser.parse_args()

    config_file = os.path.join(args.exp_dir, "config.json")
    config = Config(config_file)

    tokenizer, max_len_token = get_tokenizer(config)
    vocab = get_vocab(config, tokenizer, max_len_token)

    data_dir = os.path.split(config.test_file)[0]

    if args.is_parallel == "True": 
        test_output_file = os.path.join(args.exp_dir, "test_shards_pred", "shard_" + args.shard + ".pred")
        test_label_file = os.path.join(data_dir, "test_shards", "shard_" + args.shard)
    elif args.test_file is not None:
        test_output_file = os.path.join("output.txt")
        test_label_file = args.test_file   
    else:
        test_output_file = os.path.join(args.exp_dir, "test.pred")
        test_label_file = config.test_file

    test_model(config, vocab, tokenizer, args.exp_dir, test_label_file, test_output_file)
    