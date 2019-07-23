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

import os
import sys


from main.objects.Tokenizer import Char
from main.objects.Config import Config
from main.objects.Vocab import Vocab


from main.models.AlignBinary import AlignBinary
from main.models.AlignDot import AlignDot
from main.models.AlignLinear import AlignLinear
from main.models.AlignCNN import AlignCNN
from main.models.Stance import Stance
from main.models.LDTW import LDTW


def get_tokenizer(config):
    '''
    Returns the tokenizer specified in config 

    param config: configuration for training 
    return: tokenizer object 
    return: maximum number of tokens 
    '''
    tokenizer = ""
    max_len_token = 0

    if(config.tokenizer_name == "Char"):
        tokenizer = Char()
        max_len_token = config.max_num_char

    return tokenizer, max_len_token

def get_vocab(config, tokenizer, max_len_token):
    '''
    Returns the vocabulary object 

    param config: configuratin for training 
    param tokenizer: tokenizer for training 
    param max_len_token: maximumum number of tokens 
    return: vocab object 
    '''
    vocab_file = config.vocab_file + "_" + config.tokenizer_name.lower()
    if(not os.path.exists(vocab_file)):
        print("Make Vocab for " + config.tokenizer_name)

    vocab = Vocab(vocab_file, tokenizer, max_len_token)
    return vocab

def get_model(config, vocab, max_len_token):
    '''
    Returns an object of the model 

    param config: configuration of the model 
    param vocab: vocab for model 
    param max_len_token: maximum number of tokens 
    return: model 
    '''
    model = None

    # Set up Model
    if config.model_name == "AlignBinary":
        model = AlignBinary(config, vocab, max_len_token)
    elif config.model_name == "AlignDot":
        model = AlignDot(config, vocab, max_len_token)
    elif config.model_name == "AlignLinear":
        model = AlignLinear(config, vocab, max_len_token)
    elif config.model_name == "AlignCNN":
        model = AlignCNN(config, vocab, max_len_token)
    elif config.model_name == "Stance":
        model = Stance(config, vocab, max_len_token)
    elif config.model_name == "LDTW":
        model = LDTW(config, vocab, max_len_token)
    else:
        raise ValueError("Model Unknown: ", config.model_name)

    return model
