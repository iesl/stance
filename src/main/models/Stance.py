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
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
from torch.autograd import Variable
import math

from main.base_models.EMB import EMB
from main.base_models.LSTM import LSTM
from main.base_models.CNN import CNN

from main.objects.Sinkhorn import batch_sinkhorn_loss
from main.utils.token_lookup import get_qry_pos_neg_tok_lookup, get_qry_cnd_tok_lookup


'''
STANCE first gets character embeddings. Next, LSTM runs over char embeddings 
to get char representations. Then, similarity matrix created where all LSTM
embeddings are scored for similarity using dot product. Optimal Transport is then run over the 
similarity matrix to align weights. Finally, CNN detects features in aligned matrix and outputs similarity score
'''
class Stance(torch.nn.Module):
    def __init__(self, config, vocab, max_len_token):
        """
        param config: config object
        param vocab: vocab object
        param max_len_token: max number of tokens 
        """
        super(Stance, self).__init__()
        self.config = config
        self.vocab = vocab
        self.max_len_token = max_len_token

        self.need_flatten = True

        self.EMB = EMB(vocab.size+1, config.embedding_dim)
        self.LSTM = LSTM(self.EMB, config.embedding_dim, config.rnn_hidden_dim, config.bidirectional)
        self.CNN = CNN(config.increasing, config.cnn_num_layers, config.filter_counts, max_len_token)

        # Vector of ones (used for loss)
        self.ones = Variable(torch.ones(config.train_batch_size, 1)).cuda()
        self.loss = BCEWithLogitsLoss()

    def compute_loss(self, qry_tok, pos_tok, neg_tok):
        """
        Computes loss for batch of query positive negative triplets

        param qry: query tokens (batch size of list of tokens)
        param pos: positive mention lookup (batch size of list of tokens)
        param neg: negative mention lookup (batch size of list of tokens)
        return: loss (batch_size)
        """
        qry_lkup, pos_lkup, neg_lkup = get_qry_pos_neg_tok_lookup(self.vocab, qry_tok, pos_tok, neg_tok)

        qry_emb, qry_mask = self.LSTM(torch.from_numpy(qry_lkup).cuda())
        pos_emb, pos_mask = self.LSTM(torch.from_numpy(pos_lkup).cuda())
        neg_emb, neg_mask = self.LSTM(torch.from_numpy(neg_lkup).cuda())

        loss = self.loss(self.score_pair_train(qry_emb, pos_emb, qry_mask, pos_mask)
                            - self.score_pair_train(qry_emb, neg_emb, qry_mask, neg_mask),  self.ones)

        return loss


    def score_pair_train(self, qry_emb, cnd_emb, qry_msk, cnd_msk):
        """ 
        Scores the batch of query candidate pair
        Take the dot product of all pairs of embeddings (with bmm) to get similarity matrix
        Uses optimal transport to align the weights
        Then runs CNN over the similarity matrix

        param qry: query mention embedding (batch_size * max_len_token * hidden_state_output_size)
        param cnd: candidate mention embedding (batch_size * max_len_token * hidden_state_output_size)
        param qry_msk: query mention mask (batch_size * max_len_token)
        param cnd_mask: candidate mention mask (batch_size * max_len_token)
        return: score for query candidate pairs (batch_size * 1)
        """
        qry_cnd_sim = torch.bmm(qry_emb, torch.transpose(cnd_emb, 2, 1))

        qry_mask = qry_msk.unsqueeze(dim=2)
        cnd_msk = cnd_msk.unsqueeze(dim=1)
        qry_cnd_mask = torch.bmm(qry_mask, cnd_msk)

        qry_cnd_dist = torch.cuda.FloatTensor(qry_cnd_sim.size()).fill_(torch.max(qry_cnd_sim)) - qry_cnd_sim + 1e-6
        qry_cnd_pi = batch_sinkhorn_loss(qry_cnd_dist, qry_cnd_mask)
        qry_cnd_sim_aligned = torch.mul(qry_cnd_sim, qry_cnd_pi)
        qry_cnd_sim_aligned = torch.mul(qry_cnd_sim_aligned, qry_cnd_mask)

        return self.CNN(qry_cnd_sim_aligned)

    def score_dev_test_batch(self, qry_tk, cnd_tk):
        """ 
        Returns the score for query candidate pair 

        param qry: query mention lookup (batch size of list of tokens)
        param cnd: candidate mention lookup (batch size of list of tokens)
        return: score (batch_size)
        """
        qry_lkup, cnd_lkup = get_qry_cnd_tok_lookup(self.vocab, qry_tk, cnd_tk)

        qry_emb, qry_mask = self.LSTM(torch.from_numpy(qry_lkup).cuda())
        cnd_embed, cnd_mask = self.LSTM(torch.from_numpy(cnd_lkup).cuda())

        scores = self.score_pair_train(qry_emb, cnd_embed, qry_mask, cnd_mask)
        return scores

    def flatten_parameters(self):
        self.LSTM.flatten_parameters()
