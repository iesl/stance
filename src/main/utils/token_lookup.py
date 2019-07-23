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

def lookup_qry_cnd(vocab, btc_qry_tk, btc_cnd_tk):
    '''
    Returns the list of token indices given list of tokens passed in 

    param vocab: vocab object
    param btc_qry_tk: batch of query tokens 
    param btc_cnd_tk: batch of candidate tokens 
    return: batch of query token indices 
    return: batch of candidate token indices 
    '''
    btc_qry_lkup = []
    btc_cnd_lkup = []

    for qry_tk, cnd_tk in zip(btc_qry_tk, btc_cnd_tk):
        btc_qry_lkup.append(vocab.to_ints(qry_tk))
        btc_cnd_lkup.append(vocab.to_ints(cnd_tk))

    return np.asarray(btc_qry_lkup), np.asarray(btc_cnd_lkup)

def lookup_qry_pos_neg(vocab, btc_qry_tk, btc_pos_tk, btc_neg_tk):
    '''
    Returns the list of token indices given list of tokens passed in 

    param vocab: vocab object
    param btc_qry_tk: batch of query tokens 
    param btc_pos_tk: batch of positive tokens 
    param btc_neg_tk: batch of negative tokens 
    return: batch of query token indices 
    return: batch of positive token indices
    return: batch of negative token indices  
    '''
    btc_qry_lkup = []
    btc_pos_lkup = []
    btc_neg_lkup = []

    for qry_tk, pos_tk, neg_tk in zip(btc_qry_tk, btc_pos_tk, btc_neg_tk):
        btc_qry_lkup.append(vocab.to_ints(qry_tk))
        btc_pos_lkup.append(vocab.to_ints(pos_tk))
        btc_neg_lkup.append(vocab.to_ints(neg_tk))

    return np.asarray(btc_qry_lkup), np.asarray(btc_pos_lkup), np.asarray(btc_neg_lkup)

def get_qry_pos_neg_tok_lookup(vocab, qry_tk, pos_tk, neg_tk):
    '''
    Returns the list of token indices given list of tokens passed in 

    param vocab: vocab object
    param btc_qry_tk: batch of query tokens 
    param btc_cnd_tk: batch of candidate tokens 
    return: batch of query token indices 
    return: batch of candidate token indices 
    '''
    return lookup_qry_pos_neg(vocab, qry_tk, pos_tk, neg_tk)

def get_qry_cnd_tok_lookup(vocab, qry_tk, cnd_tk):
    '''
    Returns the list of token indices given list of tokens passed in 

    param vocab: vocab object
    param btc_qry_tk: batch of query tokens 
    param btc_pos_tk: batch of positive tokens 
    param btc_neg_tk: batch of negative tokens 
    return: batch of query token indices 
    return: batch of positive token indices
    return: batch of negative token indices  
    '''
    return lookup_qry_cnd(vocab, qry_tk, cnd_tk)
