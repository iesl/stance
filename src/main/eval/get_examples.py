import argparse
import os
from collections import defaultdict
import collections


def read_test_file(test_file, is_dcm=False):
    '''
    Reads the test file shard and returns dictionary mapping query to list of [candiate, label, prediction]

    Args:
        test_file: test_file containing predictions

    Returns:
        dictionary mapping to query to list of [candiate, label, prediction]
    '''

    # Different paths for Deep Conflation Models and everything else 

    dict_qry_2_cnd = defaultdict(dict)

    with open(test_file) as f_in:
        for line in f_in:
            tab_split_line = line.strip('\n').split('\t')
            qry = tab_split_line[0] 
            cnd = tab_split_line[1]
            if is_dcm == False:
                lbl = int(tab_split_line[2])
                pred = float(tab_split_line[3])
            else:
                pred = float(tab_split_line[2])  
                lbl = int(2)
            lbl_pred = [lbl, pred]

            dict_qry_2_cnd[qry][cnd] = lbl_pred


    return dict_qry_2_cnd

def form_line(tab_split_list):
    '''
    Forms line from tab split list 

    Args:
        tab_split_list: tab split list of strings 

    Returns:
        string of tab split list joined by tab 
    '''
    str_tab_split_list = map(lambda x: str(x), tab_split_list)
    return '\t'.join(str_tab_split_list)

def compare_two_dictionary_queries(dict_one_qry_2_cnd, dict_two_qry_2_cnd, dict_three_qry_2_cnd):
    '''
    Finds candidates which model one performs better than model 2

    Args:
        dict_one_qry_2_cnd: dictionary of query to list of dictionaries of candidates to [label, pred]
        dict_two_qry_2_cnd: dictionary of query to list of dictionaries of candidates to [label, pred]
    '''

    for qry in dict_one_qry_2_cnd.keys():
        cnd_lbl_pred_one = dict_one_qry_2_cnd[qry]
        cnd_lbl_pred_two = dict_two_qry_2_cnd[qry]

        cnd_lbl_pred_three = dict_three_qry_2_cnd[qry]

        assert(len(cnd_lbl_pred_one) == len(cnd_lbl_pred_two))
        assert(len(cnd_lbl_pred_one) == len(cnd_lbl_pred_three))


        # Sort according to prediction score which is in index 0 of value 
        cnd_lbl_pred_one = collections.OrderedDict(sorted(cnd_lbl_pred_one.items(), key=lambda kv: kv[1][1], reverse=True))
        cnd_lbl_pred_two = collections.OrderedDict(sorted(cnd_lbl_pred_two.items(), key=lambda kv: kv[1][1], reverse=True))
        cnd_lbl_pred_three = collections.OrderedDict(sorted(cnd_lbl_pred_three.items(), key=lambda kv: kv[1][1], reverse=True))

        counter = 0
        print_qry = False

        for ([cnd_one, [lbl_one, pred_one]], [cnd_two, [lbl_two, pred_two]], [cnd_three, [lbl_three, pred_three]]) in \
                        zip(cnd_lbl_pred_one.items(), cnd_lbl_pred_two.items(), cnd_lbl_pred_three.items()):
            
            lbl_two = cnd_lbl_pred_one[cnd_two][0]
            lbl_three = cnd_lbl_pred_three[cnd_two][0]

            # Consider only candidates which model 1 got correct (and candidates ranked higher must also be correct)
            if int(lbl_one) == 1:
                # Try to find candidates which model 2 got wrong 
                if int(lbl_two) == 0 or int(lbl_three) == 0:
                    print_qry = True

                    print(form_line([qry, cnd_one, lbl_one, pred_one]))
                    [model_one_cnd_two_lbl, model_one_cnd_two_pred] = cnd_lbl_pred_one[cnd_two]
                    print(form_line([qry, cnd_two, lbl_two, model_one_cnd_two_pred]))

                    print(form_line([qry, cnd_two, lbl_two, pred_two]))
                    [model_two_cnd_one_lbl, model_two_cnd_one_pred] = cnd_lbl_pred_two[cnd_one]
                    print(form_line([qry, cnd_one, model_two_cnd_one_lbl, model_two_cnd_one_pred]))

            else:
                break

def compare_three_dictionary_queries(dict_one_qry_2_cnd, dict_two_qry_2_cnd, dict_three_qry_2_cnd):
    '''
    Finds candidates which model one performs better than model 2

    Args:
        dict_one_qry_2_cnd: dictionary of query to list of dictionaries of candidates to [label, pred]
        dict_two_qry_2_cnd: dictionary of query to list of dictionaries of candidates to [label, pred]
    '''

    for qry in dict_one_qry_2_cnd.keys():
        cnd_lbl_pred_one = dict_one_qry_2_cnd[qry]
        cnd_lbl_pred_two = dict_two_qry_2_cnd[qry]
        cnd_lbl_pred_three = dict_three_qry_2_cnd[qry]

        assert(len(cnd_lbl_pred_one) == len(cnd_lbl_pred_two))
        assert(len(cnd_lbl_pred_one) == len(cnd_lbl_pred_three))


        # Sort according to prediction score which is in index 0 of value 
        cnd_lbl_pred_one = collections.OrderedDict(sorted(cnd_lbl_pred_one.items(), key=lambda kv: kv[1][1], reverse=True))
        cnd_lbl_pred_two = collections.OrderedDict(sorted(cnd_lbl_pred_two.items(), key=lambda kv: kv[1][1], reverse=True))
        cnd_lbl_pred_three = collections.OrderedDict(sorted(cnd_lbl_pred_three.items(), key=lambda kv: kv[1][1], reverse=True))

        counter = 0
        print_qry = False

        for ([cnd_one, [lbl_one, pred_one]], [cnd_two, [lbl_two, pred_two]], [cnd_three, [lbl_three, pred_three]]) in \
                        zip(cnd_lbl_pred_one.items(), cnd_lbl_pred_two.items(), cnd_lbl_pred_three.items()):
            
            lbl_two = cnd_lbl_pred_one[cnd_two][0]
            lbl_three = cnd_lbl_pred_three[cnd_two][0]

            # Consider only candidates which model 1 got correct (and candidates ranked higher must also be correct)
            if int(lbl_one) == 1:
                # Try to find candidates which model 2 got wrong 
                if int(lbl_two) == 0:
                    print_qry = True
                    print("Rank: ", counter)
                    print("STANCE: ", form_line([qry, cnd_one, lbl_one, pred_one]))
                    print("LDTW: ", form_line([qry, cnd_two, lbl_two, pred_two]))
                    print("DCM: ", form_line([qry, cnd_three, lbl_three, pred_three]))
                    counter += 1

                elif print_qry == True and counter < 5:
                    print("Rank: ", counter)
                    print("STANCE: ", form_line([qry, cnd_one, lbl_one, pred_one]))
                    print("LDTW: ", form_line([qry, cnd_two, lbl_two, pred_two]))
                    print("DCM: ", form_line([qry, cnd_three, lbl_three, pred_three]))
                    counter += 1

            elif print_qry == True and counter < 5:
                print("Rank: ", counter)
                print("STANCE: ", form_line([qry, cnd_one, lbl_one, pred_one]))
                print("LDTW: ", form_line([qry, cnd_two, lbl_two, pred_two]))
                print("DCM: ", form_line([qry, cnd_three, lbl_three, pred_three]))
                counter += 1

            else:
                break

def get_correct_queries(dict_one_qry_2_cnd):
    '''
    Finds candidates which model one performs better than model 2

    Args:
        dict_one_qry_2_cnd: dictionary of query to list of dictionaries of candidates to [label, pred]
    '''
    for qry in dict_one_qry_2_cnd.keys():
        cnd_lbl_pred_one = dict_one_qry_2_cnd[qry]

        # Sort according to prediction score which is in index 0 of value 
        cnd_lbl_pred_one = collections.OrderedDict(sorted(cnd_lbl_pred_one.items(), key=lambda kv: kv[1][1], reverse=True))

        for [cnd_one, [lbl_one, pred_one]] in cnd_lbl_pred_one.items():
            # Consider only candidates which model 1 got correct (and candidates ranked higher must also be correct)
            if int(lbl_one) == 1:
            # Try to find candidates which model 2 got wrong 
                print(form_line([qry, cnd_one, lbl_one, pred_one]))
                break
            else:
                break

def model_get_correct(exp_dir_one):
    '''
    Gets the queries which the model gets correct (i.e. the highest ranked candidate is a true positive)

    Args:
        exp_dir_one: directory of model (assumes that test predictions are already scored)
    '''
    dict_one_qry_2_cnd = {}
    for i in range(10):
        test_shard_pred_file = os.path.join(exp_dir_one, 'test_shards_pred', 'shard_%d.pred' % i)
        shard_dict_one_qry_2_cnd = read_test_file(test_shard_pred_file)
        dict_one_qry_2_cnd = {**dict_one_qry_2_cnd, **shard_dict_one_qry_2_cnd}

    get_correct_queries(dict_one_qry_2_cnd)

def compare_models(exp_dir_one, exp_dir_two):
    '''
    Compares two models and determines which model_one got correct that model_two got wrong 
    where correct is defined as highest ranked canddiate is a true positive 

    Args:
        exp_dir_one: directory of model (assumes that test predictions are already scored)
        exp_dir_two: directory of model (assumes that test predictions are already scored)
    '''
    dict_one_qry_2_cnd = {}
    for i in range(10):
        test_shard_pred_file = os.path.join(exp_dir_one, 'test_shards_pred', 'shard_%d.pred' % i)

        shard_dict_one_qry_2_cnd = read_test_file(test_shard_pred_file)
        dict_one_qry_2_cnd = {**dict_one_qry_2_cnd, **shard_dict_one_qry_2_cnd}

    exp_dir_two_num_shards = 10
    if "Deep_Conflation_Model" in exp_dir_two:
        exp_dir_two_num_shards = 5

    dict_two_qry_2_cnd = {}
    for i in range(exp_dir_two_num_shards):
        if 'Deep_Conflation_Model' in exp_dir:
            test_shard_pred_file = os.path.join(exp_dir, 'test_predictions', 'prediction_%d' % i)
        else:
            test_shard_pred_file = os.path.join(exp_dir_one, 'test_shards_pred', 'shard_%d.pred' % i)

        shard_dict_two_qry_2_cnd = read_test_file(test_shard_pred_file)
        dict_two_qry_2_cnd = {**dict_two_qry_2_cnd, **shard_dict_two_qry_2_cnd}

    compare_dictionary_queries(dict_one_qry_2_cnd, dict_two_qry_2_cnd)

def compare_three_models(exp_dir_one, exp_dir_two, exp_dir_three):
    '''
    Compares two models and determines which model_one got correct that model_two got wrong 
    where correct is defined as highest ranked canddiate is a true positive 

    Args:
        exp_dir_one: directory of model (assumes that test predictions are already scored)
        exp_dir_two: directory of model (assumes that test predictions are already scored)
    '''
    dict_one_qry_2_cnd = {}
    for i in range(10):
        test_shard_pred_file = os.path.join(exp_dir_one, 'test_shards_pred', 'shard_%d.pred' % i)

        shard_dict_one_qry_2_cnd = read_test_file(test_shard_pred_file)
        dict_one_qry_2_cnd = {**dict_one_qry_2_cnd, **shard_dict_one_qry_2_cnd}

    dict_two_qry_2_cnd = {}
    for i in range(10):
        test_shard_pred_file = os.path.join(exp_dir_two, 'test_shards_pred', 'shard_%d.pred' % i)

        shard_dict_two_qry_2_cnd = read_test_file(test_shard_pred_file)
        dict_two_qry_2_cnd = {**dict_two_qry_2_cnd, **shard_dict_two_qry_2_cnd}

    dict_three_qry_2_cnd = {}
    for i in range(7):
        test_shard_pred_file = os.path.join(exp_dir_three, 'test_predictions', 'prediction_%d' % i)

        shard_dict_three_qry_2_cnd = read_test_file(test_shard_pred_file, is_dcm=True)
        dict_three_qry_2_cnd = {**dict_three_qry_2_cnd, **shard_dict_three_qry_2_cnd}

    compare_three_dictionary_queries(dict_one_qry_2_cnd, dict_two_qry_2_cnd, dict_three_qry_2_cnd)

def sort_queries(output_file):
    '''
    Sorts the lines according to predicted score per query in output_file

    '''
    dict_qry_2_cnd = read_test_file(output_file)

    for qry in dict_qry_2_cnd.keys():
        cnd_lbl_pred = dict_qry_2_cnd[qry]

        # Sort according to prediction score which is in index 0 of value 
        cnd_lbl_pred = collections.OrderedDict(sorted(cnd_lbl_pred.items(), key=lambda kv: kv[1][1], reverse=True))

        for (candidate, [label, pred]) in cnd_lbl_pred.items():
            print(form_line([qry, candidate, label, pred]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e1", "--exp_dir_one", required=True)
    parser.add_argument("-e2", "--exp_dir_two", required=True)
    parser.add_argument("-e3", "--exp_dir_three", required=True)

    # parser.add_argument("-p", "--pred_file", required=True)
    args = parser.parse_args()

    compare_three_models(args.exp_dir_one, args.exp_dir_two, args.exp_dir_three)
    # model_get_correct(args.exp_dir_one)
    # sort_queries(args.pred_file)