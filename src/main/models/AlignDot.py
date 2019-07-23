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
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss

from main.base_models.LSTM import LSTM
from main.base_models.EMB import EMB
from main.utils.token_lookup import get_qry_pos_neg_tok_lookup, get_qry_cnd_tok_lookup

'''
AlignDot first gets character embeddings. Next, LSTM runs over char embeddings 
to get char representations. Then, similarity matrix created where all LSTM
embeddings are scored for similarity using dot product. Finally, similarity matrix 
summed to get score. 
'''
class AlignDot(torch.nn.Module):
    def __init__(self, config, vocab, max_len_token):
        """
        param config: config object
        param vocab: vocab object
        param max_len_token: max number of tokens 
        """
        super(AlignDot, self).__init__()
        self.config = config
        self.vocab = vocab
        self.max_len_token = max_len_token

        self.need_flatten = True

        self.EMB = EMB(vocab.size+1, config.embedding_dim)
        self.LSTM = LSTM(self.EMB, config.embedding_dim, config.rnn_hidden_dim, config.bidirectional)

        # Vector of ones (used for loss)
        self.ones = Variable(torch.ones(config.train_batch_size, 1))
        self.loss = BCEWithLogitsLoss()

    def compute_loss(self, qry_tk, pos_tk, neg_tk):
        """
        Computes loss for batch of query positive negative triplets

        param qry: query mention lookup (batch_size of list of token)
        param pos: positive mention lookup (batch_size of list of token)
        param neg: negative mention lookup (batch_size of list of token)
        return: loss (batch_size)
        """
        qry_lkup, pos_lkup, neg_lkup = get_qry_pos_neg_tok_lookup(self.vocab, qry_tk, pos_tk, neg_tk)

        qry_emb, qry_mask = self.LSTM(torch.from_numpy(qry_lkup).cuda())
        pos_emb, pos_mask = self.LSTM(torch.from_numpy(pos_lkup).cuda())
        neg_emb, neg_mask = self.LSTM(torch.from_numpy(neg_lkup).cuda())

        output_dim = qry_emb.shape[2]

        qry_len = torch.sum(qry_mask, dim=1).view(-1, 1).unsqueeze(2).repeat(1, 1, output_dim).long()
        pos_len = torch.sum(pos_mask, dim=1).view(-1, 1).unsqueeze(2).repeat(1, 1, output_dim).long()
        neg_len = torch.sum(neg_mask, dim=1).view(-1, 1).unsqueeze(2).repeat(1, 1, output_dim).long()

        qry_emb = torch.gather(input=qry_emb, dim=1, index=qry_len)
        pos_emb = torch.gather(input=pos_emb, dim=1, index=pos_len)
        neg_emb = torch.gather(input=neg_emb, dim=1, index=neg_len)

        loss = self.loss(
            self.score_pair_train(qry_emb, pos_emb, qry_mask, pos_mask)
            - self.score_pair_train(qry_emb, neg_emb, qry_mask, neg_mask),
            self.ones)

        return loss

    def score_pair_train(self, qry_emb, cnd_emb, qry_mask, cnd_mask):
        """ 
        Scores the batch of query candidate pair
        Take the dot product of all pairs of embeddings (with bmm) to get similarity matrix
        Then multiply by weight matrix and sum across row and column of 

        param qry: query mention embedding (batch_size * max_len_token * hidden_state_output_size)
        param cnd: candidate mention embedding (batch_size * max_len_token * hidden_state_output_size)
        param qry_msk: query mention mask (batch_size * max_len_token)
        param cnd_mask: candidate mention mask (batch_size * max_len_token)
        return: score for query candidate pairs (batch_size)
        """

        return torch.sum(qry_emb * cnd_emb, dim=2)

    def score_dev_test_batch(self, qry_tk, cnd_tk):
        """ 
        Returns the score for query candidate pair 

        param qry: query mention lookup (batch_size of list of tokens)
        param cnd: candidate mention lookup (batch_size of list of tokens)
        return: scores (batch_size)
        """
        qry_lkup, cnd_lkup = get_qry_cnd_tok_lookup(self.vocab, qry_tk, cnd_tk)

        qry_emb, qry_mask = self.LSTM(torch.from_numpy(qry_lkup).cuda())
        cnd_emb, cnd_mask = self.LSTM(torch.from_numpy(cnd_lkup).cuda())

        output_dim = qry_emb.shape[2]

        qry_len = torch.sum(qry_mask, dim=1).view(-1, 1).unsqueeze(2).repeat(1, 1, output_dim).long()
        cnd_len = torch.sum(cnd_mask, dim=1).view(-1, 1).unsqueeze(2).repeat(1, 1, output_dim).long()

        qry_emb = torch.gather(input=qry_emb, dim=1, index=qry_len)
        cnd_emb = torch.gather(input=cnd_emb, dim=1, index=cnd_len)

        scores = self.score_pair_train(qry_emb, cnd_emb, qry_mask, cnd_mask)

        return scores

    def flatten_parameters(self):
        self.LSTM.flatten_parameters()
