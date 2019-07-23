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

from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean
from main.utils.token_lookup import get_qry_pos_neg_tok_lookup, get_qry_cnd_tok_lookup


class MySoftDTW(torch.autograd.Function):
    calc_grad = 0
    batch_size = 0

    def forward(self, dist_matrix):
        batch_size = dist_matrix.shape[0]
        output = torch.ones([batch_size, 1]).cuda()   

        self.calc_grad = torch.ones(dist_matrix.size())

        for i in range(batch_size):
            sdtw = SoftDTW(dist_matrix[i].cpu().detach().numpy(), .5)
            value = sdtw.compute()
            output[i] = - value
            self.calc_grad[i] = torch.from_numpy(sdtw.grad())

        return output

    def backward(self, grad_out):
        grad_input = torch.ones(self.calc_grad.size()).float()
        batch_size = grad_input.shape[0]

        for i in range(batch_size):
            grad_input[i] = - self.calc_grad[i] * grad_out[i][0].item()
        return grad_input.cuda()


#This model corresponds to the baseline LDTW in our paper and incorporates code from https://github.com/mblondel/soft-dtw
class LDTW(torch.nn.Module):
    def __init__(self, config, vocab, max_len_token):
        super(LDTW, self).__init__()
        self.config = config
        self.vocab = vocab
        self.max_len_token = max_len_token

        self.need_flatten = True

        self.EMB = EMB(vocab.size+1, config.embedding_dim)
        self.LSTM = LSTM(self.EMB, config.embedding_dim, config.rnn_hidden_dim, config.bidirectional)

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

        loss = self.loss(self.score_pair_train(qry_emb, pos_emb, qry_mask, pos_mask) - \
                            self.score_pair_train(qry_emb, neg_emb, qry_mask, neg_mask),  self.ones)

        return loss


    def score_pair_train(self, qry_emb, cnd_emb, qry_msk, cnd_msk):
        """ 
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

        qry_cnd_sim = torch.mul(qry_cnd_sim, qry_cnd_mask)
        qry_cnd_dist = - qry_cnd_sim

        return MySoftDTW()(qry_cnd_dist)

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
