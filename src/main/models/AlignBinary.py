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

import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss

from main.base_models.CNN import CNN
from main.utils.token_lookup import get_qry_pos_neg_tok_lookup, get_qry_cnd_tok_lookup


'''
AlignBinary converts characters to one-hot vectors. Then, binary similarity
matrix created: 1 when chars are the same, 0 otherwise. Finally, CNN detects 
features in that matrix and outputs similarity score
'''
class AlignBinary(torch.nn.Module):
    def __init__(self, config, vocab, max_len_token):
        """
        param config: config object
        param vocab: vocab object
        param max_len_token: max number of tokens 
        """
        super(AlignBinary, self).__init__()
        self.config = config
        self.vocab = vocab

        self.need_flatten = True

        # Embeddings hack to create one-hot vectors for everything in our vocab
        self.embedding = nn.Embedding(vocab.size+1, vocab.size+1, padding_idx=0)
        self.embedding.weight.data = torch.eye(vocab.size+1)
        self.embedding.weight.requires_grad = False
        self.max_len_token = max_len_token
        self.CNN = CNN(config.increasing, config.num_layers, config.filter_counts, max_len_token)

        # Vector of ones (used for loss)
        self.ones = Variable(torch.ones(config.train_batch_size, 1))
        self.loss = BCEWithLogitsLoss()

        self.has_hidden = False

    def compute_loss(self, qry_tk, pos_tk, neg_tk):
        """ 
        Computes loss for batch of query positive negative triplets

        param qry: query mention tokens (batch_size of list of token)
        param pos: positive mention tokens (batch_size of list of token)
        param neg: negative mention tokens (batch_size of list of token)
        return: loss (batch_size)
        """
        qry_lkup, pos_lkup, neg_lkup = get_qry_pos_neg_tok_lookup(self.vocab, qry_tk, pos_tk, neg_tk)

        qry_emb, qry_mask = self.embed(qry_lkup)
        pos_emb, pos_mask = self.embed(pos_lkup)
        neg_emb, neg_mask = self.embed(neg_lkup)

        scores = self.score_pair(qry_emb , pos_emb, qry_mask, pos_mask) - self.score_pair(qry_emb , neg_emb, qry_mask, neg_mask)
        loss = self.loss(scores, self.ones)

        return loss

    def score_pair(self, qry_emb, cnd_emb, qry_msk, cnd_msk):
        """ 
        Scores the batch of query candidate pair by taking the dot produc tof all pairs of embeddings 
        which are 1 hot vectors 

        param qry_emb: query mention embedding (batch_size * max_len_token * hidden_state_output_size)
        param cnd_emb: candidate mention embedding (batch_size * max_len_token * hidden_state_output_size)
        param qry_msk: query mention mask (batch_size * max_len_token)
        param cnd_mask: candidate mention mask (batch_size * max_len_token)
        return: score for query candidate pairs (batch_size * 1)
        """
        qry_cnd_sim = torch.bmm(qry_emb, torch.transpose(cnd_emb, 2, 1))
        qry_mask = qry_msk.unsqueeze(dim=2)
        cnd_mask = cnd_msk.unsqueeze(dim=1)
        qry_cnd_mask = torch.bmm(qry_mask, cnd_mask)
        qry_cnd_sim = torch.mul(qry_cnd_sim, qry_cnd_mask)
        return self.CNN(qry_cnd_sim)

    def embed(self, mnt_lkp):
        """
        Look up embeddings for tokens - which are actually 1 hot vectors

        param mnt_lkp: batch mention lookup (batch_size * max_len_token)
        return: batch mention embedding (batch_size * max_len_token embedding dim
        """
        mnt_lkp = torch.from_numpy(mnt_lkp).cuda()
        mnt_mask = Variable(torch.cuda.ByteTensor((mnt_lkp > 0)).float())
        mnt_emb = self.embedding(Variable(mnt_lkp))
        return mnt_emb, mnt_mask

    def score_dev_test_batch(self, qry_tk, cnd_tk):
        """ 
        Returns the score for query candidate pair 

        param qry: query mention tokens (batch_size of list of token)
        param cnd: candidate mention tokens (batch_size of list of token)
        return: score (batch_size)
        """
        qry_lkup, cnd_lkup = get_qry_cnd_tok_lookup(self.vocab, qry_tk, cnd_tk)

        qry_emb, qry_mask = self.embed(qry_lkup)
        cnd_emb, cnd_mask = self.embed(cnd_lkup)

        return self.score_pair(qry_emb, cnd_emb, qry_mask, cnd_mask)
  
