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
import numpy as np
import math

"""
CNN building block class
"""
class CNN(torch.nn.Module):

    def __init__(self, is_increasing, num_layers, filter_counts, max_len_token):
        """
        params is_increasing: whether the filter size is increasing or decreasing 
        params num_layers: number of layers in the CNN
        params filter_counts: dictionary of filter index to filter size 
        params max_len_token: maximum number of tokens in sentence 
        """
        super(CNN, self).__init__()
        decreasing = 0
        if(is_increasing != True):
            decreasing = 1
        self.num_layers = num_layers

        map_conv_layer_to_filter_size = {4: [[3, 5, 5, 7], [7, 5, 5, 3]], 3: [[5, 5, 7], [7, 5, 5]], 2: [[5, 3],[5, 3]], 1: [[7],[7]]}
        pool_output_height = int(np.floor(max_len_token/2.0))


        for i in range(1, self.num_layers+1):
            filter_size = map_conv_layer_to_filter_size[self.num_layers][decreasing][i-1]
            padding_size = math.floor(filter_size / 2)
            prev_filter_count = 1
            if(i > 1):
                prev_filter_count = filter_counts[i-2]
            convlyr = nn.Conv2d(prev_filter_count, filter_counts[i-1], filter_size, padding = padding_size, stride=1)
            if(i == 1):
                self.add_module("cnn_1", convlyr)
            elif(i == 2):
                self.add_module("cnn_2", convlyr)
            elif(i == 3):
                self.add_module("cnn_3", convlyr)
            elif(i == 4):
                self.add_module("cnn_4", convlyr)

        self.align_weights = nn.Parameter(torch.randn(filter_counts[num_layers - 1], pool_output_height, pool_output_height).cuda(),requires_grad=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d((2, 2), stride=2)

    def forward(self, src_tgt_sim):
        """
        Runns CNN over input 

        :params src_tgt_sim: tensor representing similarity between source and target 
        :return: scores for similarity
        """

        # Needs num channels
        convd = self.cnn_1(src_tgt_sim.unsqueeze(1)) 
        if self.num_layers > 1:
            convd = self.relu(convd)
            convd = self.cnn_2(convd)
        if self.num_layers > 2:
            convd = self.relu(convd)
            convd = self.cnn_3(convd)
        if self.num_layers > 3:
            convd = self.relu(convd)
            convd = self.cnn_4(convd)

        convd_after_pooling = self.pool(convd)

        output = torch.sum(self.align_weights.expand_as(convd_after_pooling) * convd_after_pooling, dim=3,keepdim=True)
        output = torch.sum(output, dim=2,keepdim=True)
        output = torch.squeeze(output, dim=3)
        output = torch.squeeze(output, dim=2)
        output = torch.sum(output, dim=1,keepdim=True)

        return output
