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
Embedding building block class
"""
class EMB(torch.nn.Module):

    def __init__(self, vocab_size, embedding_dim):
        '''
        param  vocab_size: size of vocabulary
        param embedding_dim: embedding dimension 
        '''
        super(EMB, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)

    def forward(self, string_lkup):
        """
        Looks up the embedding for the string lookup integers 

        param string_lkup: string lookup integers 
        return:  embeddings for string lookup integers 
        return: mask for embeddings 
        """
        mask = Variable(torch.cuda.ByteTensor((string_lkup > 0)).float())
        emb = self.embedding(Variable(string_lkup))
        return emb, mask
