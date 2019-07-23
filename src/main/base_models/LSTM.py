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

"""
LSTM building block class
"""
class LSTM(torch.nn.Module):

    def __init__(self, emb, embedding_dim, rnn_hidden_size, is_bidirectional):
        '''
        param emb: embedding layer to use 
        param embedding_dim: embedding dimension 
        param rnn_hidden_size: dimension of rnn hidden unit 
        param is_bidirectional: whether the LSTM is bidirectional or not 
        '''
        super(LSTM, self).__init__()
        self.num_directions = 1
        if is_bidirectional:
            self.num_directions = 2
        self.rnn_hidden_size = rnn_hidden_size

        self.EMB = emb
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=rnn_hidden_size, num_layers=1, bidirectional=is_bidirectional, batch_first=True)


    def forward(self, str_lkp):
        """
        Returns the hidden units after running LSTM and EMB over string lookup 

        :param str_lkp: batch_size * max_len_token
        :return emb: batch_size * max_len_token * embedding dim
        :return mask: batch_size * max_len_token * embedding dim
        """
        emb, mask = self.EMB(str_lkp)
        emb, final_hn_cn = self.lstm(emb)
        return emb, mask

    def flatten_parameters(self):
        self.lstm.flatten_parameters()
