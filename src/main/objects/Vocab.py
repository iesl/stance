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

import codecs
import numpy as np

'''
Vocab object that represents the vocabulary 
'''
class Vocab(object):
    def __init__(self, filename, tokenizer, max_len_token):
        '''
        param filename: filename of vocab 
        param tokenizer: tokenizer used to generate vocab 
        param max_len_token: maximum number of tokens 
        '''
        self.filename = filename
        self.OOV = "<OOV>"
        self.OOV_INDEX = 1
        self.tokenizer = tokenizer
        (self.token2id, self.id2token) = self.load(self.filename)
        self.PADDING_INDEX = 0
        self.max_len_token = int(max_len_token)
        self.size = len(self.token2id)

    def __len__(self):
        '''

        return: vocab size
        '''
        return self.size

    def load(self, filename):
        '''
        Loads the vocab from file 

        param filename: file name of vocab 
        return: dictionary of token to id of token 
        return: dictionary of id to token 
        '''
        token2id = dict()
        id2token = dict()

        token2id[self.OOV] = self.OOV_INDEX
        id2token[self.OOV_INDEX] = self.OOV

        with codecs.open(filename,'r','UTF-8') as fin:
            for line in fin:
                splt = line.split("\t")
                item = splt[0]
                id = int(splt[1].strip())
                token2id[item] = id
                id2token[id] = item

        return token2id, id2token 

    def to_ints(self, list_tokens):
        '''
        Converts a list of tokens to list of token indices

        param list_tokens: list of tokens in string 
        return: list of token indices 
        '''
        list_ints = []

        for token in list_tokens:
            list_ints.append(self.token2id.get(token, self.OOV_INDEX))

        if len(list_ints) > self.max_len_token:
            return np.asarray(list_ints[0:self.max_len_token])
        
        # Pad the list of ints if less than max_len
        while len(list_ints) < self.max_len_token:
            list_ints += [self.PADDING_INDEX]

        return np.asarray(list_ints)

    def to_string(self, list_idx):
        '''
        Converts a list of indices to a list of tokens 

        param list_idx: list of indices of tokens in string 
        return: list of tokens 
        '''
        list_tokens = []

        for idx in list_idx:
            print(idx)
            list_tokens.append(self.id2token.get(int(idx), self.OOV))

        return np.asarray(list_tokens)


