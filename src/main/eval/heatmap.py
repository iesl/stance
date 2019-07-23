import argparse

import matplotlib
import numpy as np
import torch
from torch.autograd import Variable

import os

from main.objects.Config import Config
from main.objects.Batcher import Batcher

from main.utils.model_helper import get_tokenizer, get_vocab, get_model
from main.utils.token_lookup import get_qry_cnd_tok_lookup
from main.objects.Sinkhorn import batch_sinkhorn_loss

from sdtw import SoftDTW
from sdtw.distance import SquaredEuclidean


matplotlib.use('Agg')
import matplotlib.pyplot as plt
import codecs

def load_data(input_file, vocab, config, tokenizer, max_len_token):
    with open(input_file, "r") as f:
        qry_tk = []
        qry_len = []
        cnd_tk = []
        cnd_len = []

        for line in f.readlines():
            split = line.strip('\n').split("\t") 

            qry_tk.append(tokenizer.tokenize(split[0]))
            qry_len.append(len(tokenizer.tokenize(split[0])))
            cnd_tk.append(tokenizer.tokenize(split[1]))
            cnd_len.append(len((tokenizer.tokenize(split[1]))))

    qry_tk = np.asarray(qry_tk)
    qry_len = np.asarray(qry_len)
    cnd_tk = np.asarray(cnd_tk)
    cnd_len = np.asarray(cnd_len)

    return qry_tk, cnd_tk, qry_len, cnd_len

def print_stance_mm(exp_dir, qry_tk, cnd_tk):
    """ Prints the matrix multiplication of the two embeddings.
    This function is useful for creating a heatmap for figures.
    :param src: Entity mentions
    :param tgt: Entity mentions
    :param src_len: lengths of src mentions
    :param neg_len: lengths of tgt mentions
    :return:
    """
    model = torch.load(os.path.join(exp_dir, "best_model"))

    qry_lkup, cnd_lkup = get_qry_cnd_tok_lookup(vocab, qry_tk, cnd_tk)

    qry_emb, qry_mask = model.LSTM(torch.from_numpy(qry_lkup).cuda())
    cnd_emb, cnd_mask = model.LSTM(torch.from_numpy(cnd_lkup).cuda())

    sim = torch.bmm(qry_emb, torch.transpose(cnd_emb, 2, 1))
    dist = torch.cuda.FloatTensor(sim.size()).fill_(torch.max(sim)) - sim + 1e-6

    mat_mask = torch.bmm(qry_mask.unsqueeze(2), cnd_mask.unsqueeze(1))
    pi = batch_sinkhorn_loss(dist, mat_mask)

    stance_multpld = torch.mul(sim, pi)
    stance_multpld = torch.mul(stance_multpld, mat_mask)

    return sim, dist, pi, stance_multpld

def print_dtw_mm(exp_dir, qry_tk, cnd_tk):
    """ Prints the matrix multiplication of the two embeddings.
    This function is useful for creating a heatmap for figures.
    :param src: Entity mentions
    :param tgt: Entity mentions
    :param src_len: lengths of src mentions
    :param neg_len: lengths of tgt mentions
    :return:
    """
    model = torch.load(os.path.join(exp_dir, "best_model"))

    qry_lkup, cnd_lkup = get_qry_cnd_tok_lookup(vocab, qry_tk, cnd_tk)

    qry_emb, qry_mask = model.LSTM(torch.from_numpy(qry_lkup).cuda())
    cnd_emb, cnd_mask = model.LSTM(torch.from_numpy(cnd_lkup).cuda())

    sim = torch.bmm(qry_emb, torch.transpose(cnd_emb, 2, 1))
    dist = torch.cuda.FloatTensor(sim.size()).fill_(torch.max(sim)) - sim + 1e-6

    dtw_mat = torch.ones(dist.size())
    batch_size = dist.shape[0]

    for i in range(batch_size):
        sdtw = SoftDTW(dist[i].cpu().detach().numpy(), .5)
        value = sdtw.compute()
        dtw_mat[i] = torch.from_numpy(sdtw.grad())

    return dtw_mat

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--test_file", required=True)
    parser.add_argument("-s", "--stance_exp_dir", required=True)
    parser.add_argument("-d", "--dtw_exp_dir", required=True)    
    parser.add_argument("-o", "--output_folder", required = True)

    args = parser.parse_args()
    output_folder = args.output_folder

    config = Config(os.path.join(args.stance_exp_dir, "config.json"))
    tokenizer, max_len_token = get_tokenizer(config)
    vocab = get_vocab(config, tokenizer, max_len_token)
    qry_tk, cnd_tk, qry_len, cnd_len = load_data(args.test_file, vocab, config, tokenizer, max_len_token)

    stance_sim, stance_dist, stance_pi, stance_mulpld = print_stance_mm(args.stance_exp_dir, qry_tk, cnd_tk)
    dtw_mat = print_dtw_mm(args.dtw_exp_dir, qry_tk, cnd_tk)

    stance_sim, stance_dist, stance_pi, stance_mulpld, dtw_mat = stance_sim.cpu().data.numpy(),  stance_dist.cpu().data.numpy(),\
                                                             stance_pi.cpu().data.numpy(), stance_mulpld.cpu().data.numpy(), \
                                                                    dtw_mat.cpu().data.numpy()
    min_sim = np.min(stance_sim)
    max_sim = np.max(stance_sim)
    min_pi = np.min(stance_pi)
    max_pi = np.max(stance_pi)
    min_dist = np.min(stance_dist)
    max_dist = np.max(stance_dist)
    min_multpld = np.min(stance_mulpld)
    max_multpld = np.max(stance_mulpld)
    min_dtw = np.min(dtw_mat)
    max_dtw = np.max(dtw_mat)

    # for idx in range(0, len(stance_sim)):
    #     stance_sim[idx][qry_len[idx]] = max_sim
    #     stance_sim[idx][qry_len[idx] + 1] = min_sim
    #     stance_pi[idx][qry_len[idx]] = max_pi
    #     stance_pi[idx][qry_len[idx] + 1] = min_pi
    #     stance_mulpld[idx][qry_len[idx]] = max_multpld
    #     stance_mulpld[idx][qry_len[idx] + 1] = min_multpld
    #     dtw_mat[idx][qry_len[idx]] = max_dtw
    #     dtw_mat[idx][qry_len[idx] + 1] = min_dtw

    fontsize=64

    for idx in range(0, len(stance_sim)):
        max_interesting = int(max(qry_len[idx], cnd_len[idx]))
        my_xticks = list(qry_tk[idx])
        my_yticks = list(cnd_tk[idx])

        sim_fig = plt.figure(figsize=(25, 25))
        plt.yticks(range(cnd_len[idx]), my_yticks, size = fontsize, rotation = 'horizontal')
        plt.xticks(range(qry_len[idx]), my_xticks, size = fontsize, rotation = 'horizontal')
        plt.imshow(stance_sim[idx][:max_interesting, :max_interesting].transpose(), cmap='hot', interpolation='nearest')
        plt.gca().invert_yaxis()
        sim_fig.savefig(os.path.join(output_folder, "sim_{}.png".format(idx)))

        dist_fig = plt.figure(figsize = (25, 25))
        plt.yticks(range(cnd_len[idx]), my_yticks, size = fontsize, rotation = 'horizontal')
        plt.xticks(range(qry_len[idx]), my_xticks, size = fontsize, rotation = 'horizontal')
        plt.imshow(stance_dist[idx][:max_interesting, :max_interesting].transpose(), cmap='hot', interpolation='nearest')
        plt.gca().invert_yaxis()
        dist_fig.savefig(os.path.join(output_folder, "dist_{}.png".format(idx)))

        pi_fig = plt.figure(figsize = (25, 25))
        plt.yticks(range(cnd_len[idx]), my_yticks, size = fontsize, rotation = 'horizontal')
        plt.xticks(range(qry_len[idx]), my_xticks, size = fontsize, rotation = 'horizontal')
        plt.imshow(stance_pi[idx][:max_interesting, :max_interesting].transpose(), cmap='hot', interpolation='nearest')
        plt.gca().invert_yaxis()
        pi_fig.savefig(os.path.join(output_folder, "pi_{}.png".format(idx)))

        multpld_fig = plt.figure(figsize = (25, 25))
        plt.yticks(range(cnd_len[idx]), my_yticks, size = fontsize, rotation = 'horizontal')
        plt.xticks(range(qry_len[idx]), my_xticks, size = fontsize, rotation = 'horizontal')
        plt.imshow(stance_mulpld[idx][:max_interesting, :max_interesting].transpose(), cmap='hot', interpolation='nearest')
        plt.gca().invert_yaxis()
        multpld_fig.savefig(os.path.join(output_folder, "multpld_{}.png".format(idx)))

        dtw_fig = plt.figure(figsize = (25, 25))
        plt.yticks(range(cnd_len[idx]), my_yticks, size = fontsize, rotation = 'horizontal')
        plt.xticks(range(qry_len[idx]), my_xticks, size = fontsize, rotation = 'horizontal')
        plt.imshow(dtw_mat[idx][:max_interesting, :max_interesting].transpose(), cmap='hot', interpolation='nearest')
        plt.gca().invert_yaxis()
        dtw_fig.savefig(os.path.join(output_folder, "dtw_{}.png".format(idx)))

