# Stance #
**S**imiliarity of 
**T**ransport 
**A**ligned 
**N**eural 
**C**haracter 
**E**ncodings

[Optimal Transport-based Alignment of Learned Character Representations for String Similarity](https://www.aclweb.org/anthology/P19-1592)
Derek Tam, Nicholas Monath, Ari Kobren, Aaron Traylor, Rajarshi Das, Andrew McCallum.
Association for Computational Linguistics (ACL). 2019.


## Dependencies ##
Python 3.6\
Pytorch 0.4\
numpy 1.13.3\
scikit-learn 0.21.1 \
cython \
nose 

## Dataset ## 

The datasets are at this google drive [link](https://drive.google.com/drive/folders/1LeGQWdXOwkDxVJ2UXieqbh0-ZyqraAyj) \[Updated 9/7/19\]  and the `data` directory should be put under the top directory `stance`

Training files are of the form `query \t positive \t negative`. For example, 
```
William Paget, 1st Baron Paget \t William Lord Paget \t William George Stevens 
William Paget, 1st Baron Paget \t William Lord Paget \t William Tighe  
William Paget, 1st Baron Paget \t William Lord Paget \t Edward Paget    
```

Dev and Test files are of the form `query \t candidate \t label ` where label is 1 (if candidate is alias of query) or 0 (if candidate is not alias of query). For example, 

```
peace agreement peace negotiation       1      
peace agreement interim peace treaty    1      
peace agreement Peace Accord    1  
```


## Setup ##

First, install the baselines by running `source bin/install_baseline.sh`  (from https://github.com/mblondel/soft-dtw)

For each session, run `source bin/setup.sh` to set environment variables.

If running on your own dataset, create the vocab for a dataset by running `bin/make_vocab.sh` with the training file, vocab file name, tokenizer, and miniumum count as arguments. For example, `sh bin/make_vocab.sh data/artist/artist.train data/artist/artist.vocab Char 5`. Vocab files are provided for the datasets we released.

\* Note creating the vocab only has to be done once per dataset.

## Training Models ##

First create a config JSON file (sample file at `config/artist/STANCE.json`).

Then, train the model by running `bin/run/train_model.sh` with the config JSON file as an argument. For example, `sh bin/run/train_mode.sh config/artist/stance.json`

See below for how to grid search train models 

## Evaluating Models ##

There are two options: 
1) evaluating the model on the entire test file (can take a long time to run)

    * For the first option, run `bin/run/eval_model.sh`, passing in the experiment directory as the argument. For example, `sh bin/run/eval_model.sh exp_out/artist/Stance/Char/2019-05-30-10-36-55/`. 

2) sharding the test file and evaluate the model in parallel  


    * For the second option, first shard the test file by running `bin/shard_test.sh` and passing in the test file and number of shards as arguments. For example, `sh bin/shard_test.sh data/disease/disease.test 10 0`.

        \* Note this only has to be done once per dataset

    * Then, setup a script by running `src/main/eval/setup_parallel_test.py` that will evaluate the model on each shard in parallel, passing in the experiment directory, number of shards, and gpu type as arguments. Note that the experiment directory has to be the configuration directory with the best model when using grid search. For example, `python src/main/eval/setup_parallel_test.py -e exp_out/artist/Stance/Char/2019-05-30-10-36-55 -n 10 -g 1080ti-short`

    * Finally, run the script which will be at `exp_out/{dataset}/{model}/{tokenizer}/{timestamp}/parallel_test.sh`. For example, `sh exp_out/artist/Stance/Char/2019-05-30-10-36-55/parallel_test.sh`. 

## Grid Search Train Models ##

First, create a grid search config JSON file (sample file at `config/artist/grid_search_STANCE.json`)

Then, create a script to train each model configuration in parallel by running `src/main/setup/setup_grid_search_train.py` with the grid search config file and gpu type as arguments. For example, `python src/main/setup/setup_grid_search_train.py -c config/artist/grid_search_STANCE.json -g gpu`.

\* Note the script assumes a slurm manager 

Finally, run the script, which wil be at `exp_out/{dataset}/{model}/{tokenizer}/{timestamp}/grid_search_config.sh`. For example, `sh exp_out/artist/Stance/Char/2019-05-30-15-08-47/grid_search_config.sh`.

## Citing ##

Please cite: 

```
@inproceedings{tam2019optimal,
    title = "Optimal Transport-based Alignment of Learned Character Representations for String Similarity",
    author = "Tam, Derek  and
      Monath, Nicholas  and
      Kobren, Ari  and
      Traylor, Aaron  and
      Das, Rajarshi  and
      McCallum, Andrew",
    booktitle = "Association for Computational Linguistics (ACL)",
    year = "2019"
}
```


