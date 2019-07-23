"""
Copyright (C) 2017-2018 University of Massachusetts Amherst.
This file is part of "learned-string-alignments"
http://github.com/iesl/learned-string-alignments
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

import codecs
import sys
import string
from main.objects.Tokenizer import Char, Unigram, UnigramUC
import argparse


def count_tokens(read_file, write_file, tokenizer_name, min_count):
    '''
    Counts the number of each token in the file and then writes the tokens that occur more than a threshold to form the vocab

    param read_file: file to read tokens 
    param write_file: file to write vocab
    param tokenizer_name: tokenizer to use 
    param min_count: threshold of minimum number of occurences of tokens for vocab
    '''
    tokenizer = None
    if(tokenizer_name == "Char"):
        print("Made Char")
        tokenizer = Char()
    elif(tokenizer_name == "Unigram"):
        tokenizer = Unigram()
    elif(tokenizer_name == "UnigramUC"):
        tokenizer = UnigramUC()


    token_dict = {}

    counter = 0

    with open(read_file, 'r+') as rf:
        for line in rf:
            splt = line.strip().split("\t")
            if counter % 1000 == 0:
                sys.stdout.write("\rProcessed {} lines".format(counter))
            for s in splt:
                s_tokens = tokenizer.tokenize(s)
                for token in s_tokens:
                    if token not in token_dict:
                        token_dict[token] = 1
                    else:
                        token_dict[token] += 1

                counter += 1
    
    sys.stdout.write("\nDone....Now Writing Vocab.")
    
    with codecs.open(outputfile, "w+", "UTF-8") as wf:
        
        token_id = 2
        for token in token_dict.keys():
            if token_dict[token] >= min_count:
                wf.write("{}\t{}\n".format(token, token_id))
                token_id += 1

        wf.flush()
        wf.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_file", required=True)
    parser.add_argument("-o", "--output_file", default="False")
    parser.add_argument("-t", "--tokenizer_name", required=True)
    parser.add_argument("-m", "--min_count", default="False")
    args = parser.parse_args()

    outputfile = args.output_file + "_" + args.tokenizer_name.lower()
    count_tokens(args.input_file, args.output_file, args.tokenizer_name, int(args.min_count))