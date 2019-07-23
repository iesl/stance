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
import codecs 
import time
import string 
import torch 

'''
Batcher objects to represent getting batches from data 
'''
class Batcher(object):
	def __init__(self, config, input_type, tokenizer, labeled_file=None):
		'''
		param config: configuration object 
		param input_type: whether the batcher is for train/dev/test
		param tokenizer: tokenizer object 
		param labeled_file: labeled file to use for labels (default is specified in config )
		'''
		self.config = config
		self.tokenizer = tokenizer

		if input_type == 'train':
			self.train_batch_size = config.train_batch_size
		else:
			self.dev_test_batch_size = config.dev_test_batch_size

		self.all_qry_tk = []
		self.all_pos_tk = []
		self.all_neg_tk = []
		self.all_cnd_tk = []

		self.all_qry = []        
		self.all_cnd = []
		self.all_lbl = []

		if input_type == 'train':
			self.input_file = config.train_file
			self.train_load_data()
			self.train_shuffle_data()
		else:
			if input_type == 'dev':
				self.input_file = config.dev_file
			else: 
				# For partitioning test file to parallelize, so test_file won't be config but will be passed in
				if labeled_file is not None:
					self.input_file = labeled_file
				else:
					self.input_file = config.test_file
			self.dev_test_load_data()

		self.input_type = input_type
		self.start_idx = 0


	def get_next_dev_test_end_idx(self, cur_qry):
		'''
		Get the next batch end index such that every batch has the same query to calculate MAP without writing predictions

		param cur_qry: current query of the current batch 
		'''
		end_idx = self.start_idx

		# After breaking out of for loop, end_idx will point to first qry_tk that doesn't match cur_qry_tk
		while(end_idx < self.num_examples and self.all_qry[end_idx] == cur_qry):
			end_idx += 1

		return end_idx

	def get_dev_test_batches(self):
		'''
		Returns all the dev or test batches
		'''
		self.start_idx = 0
		cur_qry = self.all_qry[self.start_idx]

		while True:
			if self.start_idx >= self.num_examples:
				return
			else:
				end_idx = self.get_next_dev_test_end_idx(cur_qry)

				if end_idx > self.start_idx + self.dev_test_batch_size:
					end_idx = self.start_idx + self.dev_test_batch_size

				if end_idx < self.num_examples:
					cur_qry = self.all_qry[end_idx]
				end_block = (end_idx >= self.num_examples or self.all_qry[end_idx-1] != self.all_qry[end_idx])

				yield self.all_qry_tk[self.start_idx:end_idx], \
					  self.all_qry[self.start_idx:end_idx], \
					  self.all_cnd_tk[self.start_idx:end_idx], \
					  self.all_cnd[self.start_idx:end_idx], \
					  self.all_lbl[self.start_idx:end_idx], \
					  end_block
				self.start_idx = end_idx

	def get_train_batches(self):
		'''
		Returns all the train batches, where each batch includes examples with the same query 
		'''
		while True:
			if self.start_idx > self.num_examples - self.train_batch_size: 
				self.start_idx = 0
				self.train_shuffle_data()
			else:
				end_idx = self.start_idx + self.train_batch_size
				yield self.all_qry_tk[self.start_idx:end_idx], \
					  self.all_pos_tk[self.start_idx:end_idx], \
					  self.all_neg_tk[self.start_idx:end_idx]
				self.start_idx = end_idx

	def train_shuffle_data(self):
		'''
		Shuffles the training data, maintining the permutation across query, positive, and negative tokens 
		'''
		perm = np.random.permutation(self.num_examples)  # perm of index in range(0, num_questions)
		assert len(perm) == self.num_examples
		self.all_qry_tk, self.all_pos_tk, self.all_neg_tk = self.all_qry_tk[perm], self.all_pos_tk[perm], self.all_neg_tk[perm]

	def dev_test_load_data(self):
		'''
		Loads and stores the tokens for dev/test data 
		'''
		with open(self.input_file, "r") as f:
			for line in f.readlines():
				split = line.strip('\n').split("\t") 

				self.all_qry_tk.append(self.tokenizer.tokenize(split[0]))
				self.all_qry.append(split[0])
				self.all_cnd_tk.append(self.tokenizer.tokenize(split[1]))
				self.all_cnd.append(split[1])
				self.all_lbl.append(int(split[2]))

		self.all_qry_tk = np.asarray(self.all_qry_tk)
		self.all_cnd_tk = np.asarray(self.all_cnd_tk)
		self.all_lbl = np.asarray(self.all_lbl, dtype=np.int32)
		self.num_examples = len(self.all_qry_tk)

	def train_load_data(self):
		'''
		Loads and stores the tokens for test data 
		'''
		with codecs.open(self.input_file, "r", "UTF-8", errors="ignore") as inp:
			for line in inp:
				split = line.encode("UTF-8").strip().decode("UTF-8").split("\t") 
				
				if(len(self.tokenizer.tokenize(split[0])) <= 0):
					print(line)
					raise ValueError

				self.all_qry_tk.append(self.tokenizer.tokenize(split[0]))
				self.all_pos_tk.append(self.tokenizer.tokenize(split[1]))
				self.all_neg_tk.append(self.tokenizer.tokenize(split[2]))

		self.all_qry_tk = np.asarray(self.all_qry_tk)
		self.all_pos_tk = np.asarray(self.all_pos_tk)
		self.all_neg_tk = np.asarray(self.all_neg_tk)
		self.num_examples = len(self.all_qry_tk)
