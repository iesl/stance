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

'''
Writer object to write the scores 
'''
class Writer(object):
    def __init__(self, test_file):
        '''
        param test_file: test file to write the scores to 
        '''
        self.test_file = open(test_file, 'w+')

    def add_batch_pred_lab(self, qry, cnd, lbl, score):
        '''
        Writes batch of prediction 

        param qry: batch of query tokens 
        param cnd: batch of candidate tokens 
        param lbl: batch of labels 
        param score: batch of scores 
        '''
        for i in range(len(qry)):
            tab_splits = [qry[i], cnd[i], str(lbl[i]), ("{:.3f}".format(score[i]))]
            line = '\t'.join(tab_splits) + '\n'
            self.test_file.write(line)
