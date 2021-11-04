# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import re
import shutil
import warnings
import pandas as pd
import numpy as np
import sys
import socket

if 'kwai' not in socket.gethostname():
    sys.path.append('/home/zhengyu/workspace/kmcuda/src/')
    from libKMCUDA import kmeans_cuda
import gzip
import random
import logging
import json
from datetime import datetime
from tqdm import tqdm
import _pickle as cPickle
from reco_utils.dataset.download_utils import maybe_download, download_path

from reco_utils.recommender.deeprec.deeprec_utils import load_dict

logger = logging.getLogger()


def strong_data_preprocessing(
    raw_data,
    test_file,
    strong_file,
    user_vocab,
    item_vocab,
    test_num_ngs=9,
    dataset='taobao',
    category_file=None,
):
    """Create data for counterfactual evaluation using stronger behaviors.
    """
    if dataset == 'taobao' or dataset == 'taobao_global':
        (strong_last_test_file, strong_first_test_file, weak_test_file, strong_last_vocab, strong_first_vocab) = raw_data
        taobao_strong_main(strong_last_vocab, strong_first_vocab, strong_file, user_vocab, item_vocab)
        _strong_data_processing(strong_last_test_file, strong_first_test_file, weak_test_file, strong_last_vocab, strong_first_vocab, test_file, test_num_ngs)
    elif dataset == 'kuaishou_open':
        (strong_like_last_test_file, strong_like_first_test_file, strong_follow_last_test_file, strong_follow_first_test_file, weak_test_file, 
                             strong_like_last_vocab, strong_like_first_vocab, strong_follow_last_vocab, strong_follow_first_vocab) = raw_data
        kuaishou_open_strong_main(strong_like_last_vocab, strong_like_first_vocab, strong_follow_last_vocab, strong_follow_first_vocab, strong_file, user_vocab, item_vocab, category_file)
        _strong_data_processing(strong_like_last_test_file, strong_like_first_test_file, weak_test_file, strong_like_last_vocab, strong_like_first_vocab, test_file, test_num_ngs)
        _strong_data_processing(strong_follow_last_test_file, strong_follow_first_test_file, weak_test_file, strong_follow_last_vocab, strong_follow_first_vocab, test_file, test_num_ngs)


def _strong_data_processing(strong_last_test_file, strong_first_test_file, weak_test_file, strong_last_vocab, strong_first_vocab, test_file, test_num_ngs):

    strong_last_dict = load_dict(strong_last_vocab)
    strong_first_dict = load_dict(strong_first_vocab)
    with open(test_file, "r") as f:
        test_lines = f.readlines()
    write_strong_last_test = open(strong_last_test_file, "w")
    write_strong_first_test = open(strong_first_test_file, "w")
    write_weak_test = open(weak_test_file, "w")

    strong_last_position = []
    strong_first_position = []

    for count in tqdm(range(len(test_lines)//(1 + test_num_ngs))):
        cursor = count*(1 + test_num_ngs)
        words = test_lines[cursor].strip().split("\t")
        user_id = int(words[1])

        next_cursor = (count + 1)*(1 + test_num_ngs)

        if count + 1 >= len(test_lines)//(1 + test_num_ngs) or int(test_lines[next_cursor].strip().split("\t")[1]) != user_id:

            if user_id in strong_last_dict:
                neg_data = test_lines[cursor+1:cursor+1+test_num_ngs]
                neg_items = [neg_sentence.strip().split("\t")[2] for neg_sentence in neg_data]
                strong_last_data = strong_last_dict[user_id]
                strong_first_data = strong_first_dict[user_id]
                strong_last_item = strong_last_data[0]
                strong_first_item = strong_first_data[0]
                if strong_last_item not in neg_items and strong_first_item not in neg_items:
                    write_weak_test.write(test_lines[cursor])
                    words[2] = str(strong_last_item)
                    strong_last_category = strong_last_data[1]
                    words[3] = str(strong_last_category)
                    write_strong_last_test.write("\t".join(words) + "\n")
                    words[2] = str(strong_first_item)
                    strong_first_category = strong_first_data[1]
                    words[3] = str(strong_first_category)
                    write_strong_first_test.write("\t".join(words) + "\n")
                    for neg_sentence in neg_data:
                        write_weak_test.write(neg_sentence)
                        write_strong_last_test.write(neg_sentence)
                        write_strong_first_test.write(neg_sentence)

                    time_seq = [int(t) for t in words[-1].strip().split(",")]
                    strong_last_time = strong_last_data[-1]
                    strong_first_time = strong_first_data[-1]
                    strong_last_position.append((strong_last_time - time_seq[0])/(time_seq[-1] - time_seq[0]))
                    strong_first_position.append((strong_first_time - time_seq[0])/(time_seq[-1] - time_seq[0]))

    write_strong_last_test.close()
    write_strong_first_test.close()
    write_weak_test.close()

    strong_last_position_avg = sum(strong_last_position)/len(strong_last_position)
    strong_first_position_avg = sum(strong_first_position)/len(strong_first_position)

    print('strong last average position: {:.4f}'.format(strong_last_position_avg))
    print('strong first average position: {:.4f}'.format(strong_first_position_avg))


def data_preprocessing(
    reviews_file,
    meta_file,
    train_file,
    valid_file,
    test_file,
    user_vocab,
    item_vocab,
    cate_vocab,
    sample_rate=0.01,
    valid_num_ngs=4,
    test_num_ngs=9,
    dataset='yelp',
    is_history_expanding=True,
):
    """Create data for training, validation and testing from original dataset

    Args:
        reviews_file (str): Reviews dataset downloaded from former operations.
        meta_file (str): Meta dataset downloaded from former operations.
    """
    if dataset == 'amazon':
        reviews_output = _reviews_preprocessing(reviews_file)
        meta_output = _meta_preprocessing(meta_file)
    elif dataset == 'yelp':
        reviews_output, meta_output = yelp_main(reviews_file, meta_file)
    elif dataset == 'taobao':
        reviews_output, meta_output = taobao_main(reviews_file)
    elif dataset == 'yelp_global':
        reviews_output, meta_output = yelp_main(reviews_file, meta_file)
    elif dataset == 'taobao_global':
        reviews_output, meta_output = taobao_main(reviews_file)
    elif dataset == 'kuaishou':
        reviews_output, meta_output = kuaishou_main(reviews_file)
    elif dataset == 'kuaishou_open':
        reviews_output, meta_output = kuaishou_open_main(reviews_file, meta_file)

    instance_output = _create_instance(reviews_output, meta_output)
    _create_item2cate(instance_output)
    sampled_instance_file = _get_sampled_data(instance_output, sample_rate=sample_rate)

    if dataset == 'kuaishou':
        preprocessed_output = _data_processing_ks(sampled_instance_file)
    elif dataset == 'kuaishou_open':
        preprocessed_output = _data_processing_ratio_global(sampled_instance_file, 0.9, 0.8)
    elif dataset == 'taobao_global':
        preprocessed_output = _data_processing_taobao_global(sampled_instance_file)
    elif dataset == 'yelp_global':
        preprocessed_output = _data_processing_ratio_global(sampled_instance_file, 0.8, 0.7)
    else:
        preprocessed_output = _data_processing(sampled_instance_file)

    if is_history_expanding:
        if dataset == 'kuaishou':
            _data_generating_ks(preprocessed_output, train_file, valid_file, test_file)
        elif dataset == 'kuaishou_open':
            _data_generating_global(preprocessed_output, train_file, valid_file, test_file)
        elif dataset == 'taobao_global':
            _data_generating_global(preprocessed_output, train_file, valid_file, test_file)
        elif dataset == 'yelp_global':
            _data_generating_global(preprocessed_output, train_file, valid_file, test_file)
        else:
            _data_generating(preprocessed_output, train_file, valid_file, test_file)
    else:
        _data_generating_no_history_expanding(
            preprocessed_output, train_file, valid_file, test_file
        )
    _create_vocab(dataset, train_file, user_vocab, item_vocab, cate_vocab)
    _negative_sampling_offline(
        sampled_instance_file, valid_file, test_file, valid_num_ngs, test_num_ngs
    )


def _create_vocab(dataset, train_file, user_vocab, item_vocab, cate_vocab):

    f_train = open(train_file, "r")

    user_dict = {}
    item_dict = {}
    cat_dict = {}

    logger.info("vocab generating...")
    for line in f_train:
        arr = line.strip("\n").split("\t")
        uid = arr[1]
        mid = arr[2]
        cat = arr[3]
        mid_list = arr[5]
        cat_list = arr[6]

        if uid not in user_dict:
            user_dict[uid] = 0
        user_dict[uid] += 1
        if mid not in item_dict:
            item_dict[mid] = 0
        item_dict[mid] += 1
        if cat not in cat_dict:
            cat_dict[cat] = 0
        cat_dict[cat] += 1
        if len(mid_list) == 0:
            continue
        for m in mid_list.split(","):
            if m not in item_dict:
                item_dict[m] = 0
            item_dict[m] += 1
        for c in cat_list.split(","):
            if c not in cat_dict:
                cat_dict[c] = 0
            cat_dict[c] += 1

    sorted_user_dict = sorted(user_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_item_dict = sorted(item_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_cat_dict = sorted(cat_dict.items(), key=lambda x: x[1], reverse=True)

    uid_voc = {}
    if dataset in ['kuaishou', 'kuaishou_open', 'taobao_global', 'yelp_global']: # global time split, there are some unseen users
        uid_voc["default_uid"] = 0
        index = 1
    else:
        index = 0
    for key, value in sorted_user_dict:
        uid_voc[key] = index
        index += 1

    mid_voc = {}
    mid_voc["default_mid"] = 0
    index = 1
    for key, value in sorted_item_dict:
        mid_voc[key] = index
        index += 1

    cat_voc = {}
    cat_voc["default_cat"] = 0
    index = 1
    for key, value in sorted_cat_dict:
        cat_voc[key] = index
        index += 1

    # add for gcn
    #  i2c_voc = {}
    #  cat_voc["default_cat"] = 0
    #  index = 1
    #  for key, value in sorted_cat_dict:
        #  cat_voc[key] = index
        #  index += 1

    cPickle.dump(uid_voc, open(user_vocab, "wb"))
    cPickle.dump(mid_voc, open(item_vocab, "wb"))
    cPickle.dump(cat_voc, open(cate_vocab, "wb"))
    # add for gcn
    #  cPickle.dump(i2c_voc, open(i2c_vocab, "wb"))


def _negative_sampling_offline(
    instance_input_file, valid_file, test_file, valid_neg_nums=4, test_neg_nums=49
):

    columns = ["label", "user_id", "item_id", "timestamp", "cate_id"]
    ns_df = pd.read_csv(instance_input_file, sep="\t", names=columns)
    items_with_popular = list(ns_df["item_id"])

    global item2cate

    # valid negative sampling
    logger.info("start valid negative sampling")
    with open(valid_file, "r") as f:
        valid_lines = f.readlines()
    write_valid = open(valid_file, "w")
    for line in valid_lines:
        write_valid.write(line)
        words = line.strip().split("\t")
        positive_item = words[2]
        count = 0
        neg_items = set()
        while count < valid_neg_nums:
            neg_item = random.choice(items_with_popular)
            if neg_item == positive_item or neg_item in neg_items:
                continue
            count += 1
            neg_items.add(neg_item)
            words[0] = "0"
            words[2] = str(neg_item)
            words[3] = str(item2cate[neg_item])
            write_valid.write("\t".join(words) + "\n")

    # test negative sampling
    logger.info("start test negative sampling")
    with open(test_file, "r") as f:
        test_lines = f.readlines()
    write_test = open(test_file, "w")
    for line in test_lines:
        write_test.write(line)
        words = line.strip().split("\t")
        positive_item = words[2]
        count = 0
        neg_items = set()
        while count < test_neg_nums:
            neg_item = random.choice(items_with_popular)
            if neg_item == positive_item or neg_item in neg_items:
                continue
            count += 1
            neg_items.add(neg_item)
            words[0] = "0"
            words[2] = str(neg_item)
            words[3] = str(item2cate[neg_item])
            write_test.write("\t".join(words) + "\n")


def _data_generating(input_file, train_file, valid_file, test_file, min_sequence=1):
    """produce train, valid and test file from processed_output file
    Each user's behavior sequence will be unfolded and produce multiple lines in trian file.
    Like, user's behavior sequence: 12345, and this function will write into train file:
    1, 12, 123, 1234, 12345
    """
    f_input = open(input_file, "r")
    f_train = open(train_file, "w")
    f_valid = open(valid_file, "w")
    f_test = open(test_file, "w")
    logger.info("data generating...")
    last_user_id = None
    for line in f_input:
        line_split = line.strip().split("\t")
        tfile = line_split[0]
        label = int(line_split[1])
        user_id = line_split[2]
        movie_id = line_split[3]
        date_time = line_split[4]
        category = line_split[5]

        if tfile == "train":
            fo = f_train
        elif tfile == "valid":
            fo = f_valid
        elif tfile == "test":
            fo = f_test
        if user_id != last_user_id:
            movie_id_list = []
            cate_list = []
            dt_list = []
        else:
            history_clk_num = len(movie_id_list)
            cat_str = ""
            mid_str = ""
            dt_str = ""
            for c1 in cate_list:
                cat_str += c1 + ","
            for mid in movie_id_list:
                mid_str += mid + ","
            for dt_time in dt_list:
                dt_str += dt_time + ","
            if len(cat_str) > 0:
                cat_str = cat_str[:-1]
            if len(mid_str) > 0:
                mid_str = mid_str[:-1]
            if len(dt_str) > 0:
                dt_str = dt_str[:-1]
            if history_clk_num >= min_sequence:
                fo.write(
                    line_split[1]
                    + "\t"
                    + user_id
                    + "\t"
                    + movie_id
                    + "\t"
                    + category
                    + "\t"
                    + date_time
                    + "\t"
                    + mid_str
                    + "\t"
                    + cat_str
                    + "\t"
                    + dt_str
                    + "\n"
                )
        last_user_id = user_id
        if label:
            movie_id_list.append(movie_id)
            cate_list.append(category)
            dt_list.append(date_time)

def _data_generating_ks(input_file, train_file, valid_file, test_file, min_sequence=1):
    """produce train, valid and test file from processed_output file
    Each user's behavior sequence will be unfolded and produce multiple lines in trian file.
    Like, user's behavior sequence: 12345, and this function will write into train file:
    1, 12, 123, 1234, 12345
    Add sampling with 1/10 train instances for long-range sequence dataset(kuaishou)
    """
    f_input = open(input_file, "r")
    f_train = open(train_file, "w")
    f_valid = open(valid_file, "w")
    f_test = open(test_file, "w")
    logger.info("data generating...")
    last_user_id = None
    for line in f_input:
        line_split = line.strip().split("\t")
        tfile = line_split[0]
        label = int(line_split[1])
        user_id = line_split[2]
        movie_id = line_split[3]
        date_time = line_split[4]
        category = line_split[5]

        if tfile == "train":
            fo = f_train
            sample_probability = round(np.random.uniform(0, 1), 1) # add 
        elif tfile == "valid":
            fo = f_valid
            #  sample_probability = 0 # add
            sample_probability = round(np.random.uniform(0, 1), 1) # add 
        elif tfile == "test":
            fo = f_test
            #  sample_probability = 0 # add
            sample_probability = round(np.random.uniform(0, 1), 1) # add 
        if user_id != last_user_id:
            movie_id_list = []
            cate_list = []
            dt_list = []
        else:
            if 0 <= sample_probability < 0.1: # add: 1/10 probability  
                history_clk_num = len(movie_id_list)
                cat_str = ""
                mid_str = ""
                dt_str = ""
                for c1 in cate_list:
                    cat_str += c1 + ","
                for mid in movie_id_list:
                    mid_str += mid + ","
                for dt_time in dt_list:
                    dt_str += dt_time + ","
                if len(cat_str) > 0:
                    cat_str = cat_str[:-1]
                if len(mid_str) > 0:
                    mid_str = mid_str[:-1]
                if len(dt_str) > 0:
                    dt_str = dt_str[:-1]
                if history_clk_num >= min_sequence:
                    fo.write(
                        line_split[1]
                        + "\t"
                        + user_id
                        + "\t"
                        + movie_id
                        + "\t"
                        + category
                        + "\t"
                        + date_time
                        + "\t"
                        + mid_str
                        + "\t"
                        + cat_str
                        + "\t"
                        + dt_str
                        + "\n"
                    )
            else: 
                pass
        last_user_id = user_id
        if label:
            movie_id_list.append(movie_id)
            cate_list.append(category)
            dt_list.append(date_time)

def group_sequence(test_file, split_length):
    """produce train, valid and test file from processed_output file
    Each user's behavior sequence will be unfolded and produce multiple lines in trian file.
    Like, user's behavior sequence: 12345, and this function will write into train file:
    1, 12, 123, 1234, 12345
    Add sampling with 1/10 train instances for long-range sequence dataset(kuaishou)
    """
    logger.info("data spliting for sparsity study...")
    f_test = open(test_file, "r")
    f_test_group1 = open(test_file+'_group1', "w")
    f_test_group2 = open(test_file+'_group2', "w")
    f_test_group3 = open(test_file+'_group3', "w")
    f_test_group4 = open(test_file+'_group4', "w")
    f_test_group5 = open(test_file+'_group5', "w")
    last_user_id = None
    for line in f_test:
        line_split = line.strip().split("\t")
        item_hist_list = line_split[5].split(",")
        if len(item_hist_list) <= split_length[0]:
            f_test_group1.write(line)
        elif split_length[0] < len(item_hist_list) <= split_length[1]:
            f_test_group2.write(line)
        elif split_length[1] < len(item_hist_list) <= split_length[2]:
            f_test_group3.write(line)
        elif split_length[2] < len(item_hist_list) <= split_length[3]:
            f_test_group4.write(line)
        else:
            f_test_group5.write(line)


def _data_generating_global(input_file, train_file, valid_file, test_file, min_sequence=1):
    """produce train, valid and test file from processed_output file
    Each user's behavior sequence will be unfolded and produce multiple lines in trian file.
    Like, user's behavior sequence: 12345, and this function will write into train file:
    1, 12, 123, 1234, 12345
    Add sampling with 1/10 train instances for long-range sequence dataset(kuaishou)
    """
    f_input = open(input_file, "r")
    f_train = open(train_file, "w")
    f_valid = open(valid_file, "w")
    f_test = open(test_file, "w")
    logger.info("data generating...")
    last_user_id = None
    for line in f_input:
        line_split = line.strip().split("\t")
        tfile = line_split[0]
        label = int(line_split[1])
        user_id = line_split[2]
        movie_id = line_split[3]
        date_time = line_split[4]
        category = line_split[5]

        if tfile == "train":
            fo = f_train
            sample_probability = 0
        elif tfile == "valid":
            fo = f_valid
            #  sample_probability = 0 # add
            sample_probability = round(np.random.uniform(0, 1), 1) # add 
        elif tfile == "test":
            fo = f_test
            #  sample_probability = 0 # add
            sample_probability = round(np.random.uniform(0, 1), 1) # add 
        if user_id != last_user_id:
            movie_id_list = []
            cate_list = []
            dt_list = []
        else:
            if 0 <= sample_probability < 0.2: # add: 1/5 probability  
                history_clk_num = len(movie_id_list)
                cat_str = ""
                mid_str = ""
                dt_str = ""
                for c1 in cate_list:
                    cat_str += c1 + ","
                for mid in movie_id_list:
                    mid_str += mid + ","
                for dt_time in dt_list:
                    dt_str += dt_time + ","
                if len(cat_str) > 0:
                    cat_str = cat_str[:-1]
                if len(mid_str) > 0:
                    mid_str = mid_str[:-1]
                if len(dt_str) > 0:
                    dt_str = dt_str[:-1]
                if history_clk_num >= min_sequence:
                    fo.write(
                        line_split[1]
                        + "\t"
                        + user_id
                        + "\t"
                        + movie_id
                        + "\t"
                        + category
                        + "\t"
                        + date_time
                        + "\t"
                        + mid_str
                        + "\t"
                        + cat_str
                        + "\t"
                        + dt_str
                        + "\n"
                    )
            else: 
                pass
        last_user_id = user_id
        if label:
            movie_id_list.append(movie_id)
            cate_list.append(category)
            dt_list.append(date_time)


def _data_generating_no_history_expanding(
    input_file, train_file, valid_file, test_file, min_sequence=1
):
    """produce train, valid and test file from processed_output file
    Each user's behavior sequence will only produce one line in trian file.
    Like, user's behavior sequence: 12345, and this function will write into train file: 12345
    """
    f_input = open(input_file, "r")
    f_train = open(train_file, "w")
    f_valid = open(valid_file, "w")
    f_test = open(test_file, "w")
    logger.info("data generating...")

    last_user_id = None
    last_movie_id = None
    last_category = None
    last_datetime = None
    last_tfile = None
    for line in f_input:
        line_split = line.strip().split("\t")
        tfile = line_split[0]
        label = int(line_split[1])
        user_id = line_split[2]
        movie_id = line_split[3]
        date_time = line_split[4]
        category = line_split[5]

        if last_tfile == "train":
            fo = f_train
        elif last_tfile == "valid":
            fo = f_valid
        elif last_tfile == "test":
            fo = f_test
        if user_id != last_user_id or tfile == "valid" or tfile == "test":
            if last_user_id is not None:
                history_clk_num = len(movie_id_list)
                cat_str = ""
                mid_str = ""
                dt_str = ""
                for c1 in cate_list[:-1]:
                    cat_str += c1 + ","
                for mid in movie_id_list[:-1]:
                    mid_str += mid + ","
                for dt_time in dt_list[:-1]:
                    dt_str += dt_time + ","
                if len(cat_str) > 0:
                    cat_str = cat_str[:-1]
                if len(mid_str) > 0:
                    mid_str = mid_str[:-1]
                if len(dt_str) > 0:
                    dt_str = dt_str[:-1]
                if history_clk_num > min_sequence:
                    fo.write(
                        line_split[1]
                        + "\t"
                        + last_user_id
                        + "\t"
                        + last_movie_id
                        + "\t"
                        + last_category
                        + "\t"
                        + last_datetime
                        + "\t"
                        + mid_str
                        + "\t"
                        + cat_str
                        + "\t"
                        + dt_str
                        + "\n"
                    )
            if tfile == "train" or last_user_id == None:
                movie_id_list = []
                cate_list = []
                dt_list = []
        last_user_id = user_id
        last_movie_id = movie_id
        last_category = category
        last_datetime = date_time
        last_tfile = tfile
        if label:
            movie_id_list.append(movie_id)
            cate_list.append(category)
            dt_list.append(date_time)


def _create_item2cate(instance_file):
    logger.info("creating item2cate dict")
    global item2cate
    instance_df = pd.read_csv(
        instance_file,
        sep="\t",
        names=["label", "user_id", "item_id", "timestamp", "cate_id"],
    )
    item2cate = instance_df.set_index("item_id")["cate_id"].to_dict()


def _get_sampled_data(instance_file, sample_rate):
    logger.info("getting sampled data...")
    global item2cate
    output_file = instance_file + "_" + str(sample_rate)
    columns = ["label", "user_id", "item_id", "timestamp", "cate_id"]
    ns_df = pd.read_csv(instance_file, sep="\t", names=columns)
    if sample_rate < 1:
        items_num = ns_df["item_id"].nunique()
        items_with_popular = list(ns_df["item_id"])
        items_sample, count = set(), 0
        while count < int(items_num * sample_rate):
            random_item = random.choice(items_with_popular)
            if random_item not in items_sample:
                items_sample.add(random_item)
                count += 1
        ns_df_sample = ns_df[ns_df["item_id"].isin(items_sample)]
    else:
        ns_df_sample = ns_df
    ns_df_sample.to_csv(output_file, sep="\t", index=None, header=None)
    return output_file


def _meta_preprocessing(meta_readfile):
    logger.info("start meta preprocessing...")
    meta_writefile = meta_readfile + "_output"
    meta_r = open(meta_readfile, "r")
    meta_w = open(meta_writefile, "w")
    for line in meta_r:
        line_new = eval(line)
        meta_w.write(line_new["asin"] + "\t" + line_new["categories"][0][-1] + "\n")
    meta_r.close()
    meta_w.close()
    return meta_writefile


def _reviews_preprocessing(reviews_readfile):
    logger.info("start reviews preprocessing...")
    reviews_writefile = reviews_readfile + "_output"
    reviews_r = open(reviews_readfile, "r")
    reviews_w = open(reviews_writefile, "w")
    for line in reviews_r:
        line_new = eval(line.strip())
        reviews_w.write(
            str(line_new["reviewerID"])
            + "\t"
            + str(line_new["asin"])
            + "\t"
            + str(line_new["unixReviewTime"])
            + "\n"
        )
    reviews_r.close()
    reviews_w.close()
    return reviews_writefile


def _create_instance(reviews_file, meta_file):
    logger.info("start create instances...")
    dirs, _ = os.path.split(reviews_file)
    output_file = os.path.join(dirs, "instance_output")

    f_reviews = open(reviews_file, "r")
    user_dict = {}
    item_list = []
    for line in f_reviews:
        line = line.strip()
        reviews_things = line.split("\t")
        if reviews_things[0] not in user_dict:
            user_dict[reviews_things[0]] = []
        user_dict[reviews_things[0]].append((line, float(reviews_things[-1])))
        item_list.append(reviews_things[1])

    f_meta = open(meta_file, "r")
    meta_dict = {}
    for line in f_meta:
        line = line.strip()
        meta_things = line.split("\t")
        if meta_things[0] not in meta_dict:
            meta_dict[meta_things[0]] = meta_things[1]

    f_output = open(output_file, "w")
    for user_behavior in user_dict:
        sorted_user_behavior = sorted(user_dict[user_behavior], key=lambda x: x[1])
        for line, _ in sorted_user_behavior:
            user_things = line.split("\t")
            asin = user_things[1]
            if asin in meta_dict:
                f_output.write("1" + "\t" + line + "\t" + meta_dict[asin] + "\n")
            else:
                f_output.write("1" + "\t" + line + "\t" + "default_cat" + "\n")

    f_reviews.close()
    f_meta.close()
    f_output.close()
    return output_file


def _data_processing(input_file):
    logger.info("start data processing...")
    dirs, _ = os.path.split(input_file)
    output_file = os.path.join(dirs, "preprocessed_output")

    f_input = open(input_file, "r")
    f_output = open(output_file, "w")
    user_count = {}
    for line in f_input:
        line = line.strip()
        user = line.split("\t")[1]
        if user not in user_count:
            user_count[user] = 0
        user_count[user] += 1
    f_input.seek(0)
    i = 0
    last_user = None
    for line in f_input:
        line = line.strip()
        user = line.split("\t")[1]
        if user == last_user:
            if i < user_count[user] - 2:
                f_output.write("train" + "\t" + line + "\n")
            elif i < user_count[user] - 1:
                f_output.write("valid" + "\t" + line + "\n")
            else:
                f_output.write("test" + "\t" + line + "\n")
        else:
            last_user = user
            i = 0
            if i < user_count[user] - 2:
                f_output.write("train" + "\t" + line + "\n")
            elif i < user_count[user] - 1:
                f_output.write("valid" + "\t" + line + "\n")
            else:
                f_output.write("test" + "\t" + line + "\n")
        i += 1
    return output_file

def _data_processing_ks(input_file):
    logger.info("start data processing...")
    dirs, _ = os.path.split(input_file)
    output_file = os.path.join(dirs, "preprocessed_output")

    f_input = open(input_file, "r")
    f_output = open(output_file, "w")

    ## global time division: last 12h
    test_interval = 12*60*60*1000
    user_touch_time = []
    for line in f_input:
        line = line.strip()
        time = int(line.split("\t")[3]) 
        user_touch_time.append(time)
    print("get user touch time completed") #
    user_touch_time_sorted = sorted(user_touch_time)
    test_split_time = user_touch_time_sorted[-1] - test_interval
    valid_split_time = user_touch_time_sorted[-1] - 2*test_interval

    f_input.seek(0)
    for line in f_input:
        line = line.strip()
        time = int(line.split("\t")[3]) # add
        if time < valid_split_time:
            f_output.write("train" + "\t" + line + "\n")
        elif valid_split_time <= time < test_split_time:
            f_output.write("valid" + "\t" + line + "\n")
        else:
            f_output.write("test" + "\t" + line + "\n")
    return output_file


def _data_processing_taobao_global(input_file):
    logger.info("start data processing...")
    dirs, _ = os.path.split(input_file)
    output_file = os.path.join(dirs, "preprocessed_output")

    f_input = open(input_file, "r")
    f_output = open(output_file, "w")

    ## global time division: last day
    test_interval = 24*60*60
    user_touch_time = []
    for line in f_input:
        line = line.strip()
        time = int(line.split("\t")[3]) 
        user_touch_time.append(time)
    print("get user touch time completed") #
    user_touch_time_sorted = sorted(user_touch_time)
    test_split_time = user_touch_time_sorted[-1] - test_interval
    valid_split_time = user_touch_time_sorted[-1] - 2*test_interval

    f_input.seek(0)
    for line in f_input:
        line = line.strip()
        time = int(line.split("\t")[3]) # add
        if time < valid_split_time:
            f_output.write("train" + "\t" + line + "\n")
        elif valid_split_time <= time < test_split_time:
            f_output.write("valid" + "\t" + line + "\n")
        else:
            f_output.write("test" + "\t" + line + "\n")
    return output_file


def _data_processing_ratio_global(input_file, test_split, valid_split):
    logger.info("start data processing...")
    dirs, _ = os.path.split(input_file)
    output_file = os.path.join(dirs, "preprocessed_output")

    f_input = open(input_file, "r")
    f_output = open(output_file, "w")

    ## global time division: last 20%
    user_touch_time = []
    for line in f_input:
        line = line.strip()
        time = int(line.split("\t")[3]) 
        user_touch_time.append(time)
    print("get user touch time completed") #
    user_touch_time_sorted = sorted(user_touch_time)
    test_split_time = user_touch_time_sorted[int(test_split*len(user_touch_time_sorted))]
    valid_split_time = user_touch_time_sorted[int(valid_split*len(user_touch_time_sorted))]

    f_input.seek(0)
    for line in f_input:
        line = line.strip()
        time = int(line.split("\t")[3]) # add
        if time < valid_split_time:
            f_output.write("train" + "\t" + line + "\n")
        elif valid_split_time <= time < test_split_time:
            f_output.write("valid" + "\t" + line + "\n")
        else:
            f_output.write("test" + "\t" + line + "\n")
    return output_file


def download_and_extract(name, dest_path):
    """Downloads and extracts Amazon reviews and meta datafiles if they don’t already exist"""
    dirs, _ = os.path.split(dest_path)
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    file_path = os.path.join(dirs, name)
    if not os.path.exists(file_path):
        _download_reviews(name, dest_path)
        _extract_reviews(file_path, dest_path)

    return file_path


def _download_reviews(name, dest_path):
    """Downloads Amazon reviews datafile.

    Args:
        dest_path (str): File path for the downloaded file
    """

    url = (
        "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/"
        + name
        + ".gz"
    )

    dirs, file = os.path.split(dest_path)
    maybe_download(url, file + ".gz", work_directory=dirs)


def _extract_reviews(file_path, zip_path):
    """Extract Amazon reviews and meta datafiles from the raw zip files.

    To extract all files,
    use ZipFile's extractall(path) instead.

    Args:
        file_path (str): Destination path for datafile
        zip_path (str): zipfile path
    """
    with gzip.open(zip_path + ".gz", "rb") as zf, open(file_path, "wb") as f:
        shutil.copyfileobj(zf, f)


def filter_k_core(record, k_core, filtered_column, count_column):

    stat = record[[filtered_column, count_column]] \
            .groupby(filtered_column) \
            .count() \
            .reset_index() \
            .rename(index=str, columns={count_column: 'count'})
    
    stat = stat[stat['count'] >= k_core]

    record = record.merge(stat, on=filtered_column)
    record = record.drop(columns=['count'])

    return record

def filter_k_core_consider_neg(record, k_core, filtered_column, count_column, pos_neg_column):

    stat = record[record[pos_neg_column]==1][[filtered_column, count_column]] \
            .groupby(filtered_column) \
            .count() \
            .reset_index() \
            .rename(index=str, columns={count_column: 'count'})
    
    stat = stat[stat['count'] >= k_core]

    record = record.merge(stat, on=filtered_column)
    record = record.drop(columns=['count'])

    return record

def load_data(reviews_file, business_file, dirs):

    with open(reviews_file, 'r') as f:
        review_json = f.readlines()
        review_json = [json.loads(review) for review in tqdm(review_json)]

    df_review = pd.DataFrame(review_json)
    df_review = df_review[['review_id', 'user_id', 'business_id', 'stars', 'date']]
    df_review.to_csv(os.path.join(dirs, 'yelp_review.csv'))

    with open(business_file, 'r') as f:
        business_json = f.readlines()
        business_json = [json.loads(business) for business in tqdm(business_json)]

    df_business = pd.DataFrame(business_json)
    df_business = df_business[['business_id', 'name', 'city', 'state', 'latitude', 'longitude', 'stars', 'review_count', 'attributes', 'categories']]

    df_business.to_csv(os.path.join(dirs, 'yelp_business.csv'))

    with open(os.path.join(dirs, 'categories.json'), 'r') as f:
        category = json.load(f)

    category_level_1 = [c['title'] for c in category if len(c['parents']) == 0]

    return df_review, df_business, category_level_1


def filter(review, business, category_level_1, k_core, dirs):

    business = get_business_with_category(business, category_level_1, dirs)
    review = filter_category(review, business, dirs)

    review, business = filter_cf(review, business, k_core, dirs)

    return review, business


def get_business_with_category(business, category_level_1, dirs):

    def transform(x):
        x = str(x).split(', ')
        for c in x:
            if c in category_level_1:
                return c
    business['categories'] = business['categories'].apply(transform)
    business = business.dropna(subset=['categories']).reset_index(drop=True)
    business.to_csv(os.path.join(dirs, 'yelp_business_with_category.csv'))

    return business


def filter_category(review, business, dirs):

    interacted_business = review['business_id'].drop_duplicates().reset_index(drop=True)
    interacted_business_with_category = pd.merge(interacted_business, business['business_id'], on='business_id')

    review = pd.merge(review, interacted_business_with_category, on='business_id')
    review.to_csv(os.path.join(dirs, 'yelp_review_with_category.csv'))

    return review


def filter_cf(review, business, k_core, dirs):

    review = filter_k_core(review, k_core, 'user_id', 'business_id')
    review.to_csv(os.path.join(dirs, 'yelp_review_k10.csv'))

    interacted_business = review['business_id'].drop_duplicates().reset_index(drop=True)
    business = pd.merge(business, interacted_business, on='business_id')
    business.to_csv(os.path.join(dirs, 'yelp_business_k10.csv'))

    return review, business


def transform_recommenders(review, business, dirs):

    from datetime import datetime
    def date2timestamp(x):
        x = str(x).split('-')
        day = datetime(int(x[0]), int(x[1]), int(x[2]))
        timestamp = int(datetime.timestamp(day))
        return timestamp
    review['timestamp'] = review['date'].apply(date2timestamp)

    review_slirec = review[['user_id', 'business_id', 'timestamp']]
    review_slirec.to_csv(os.path.join(dirs, 'yelp_review_recommenders.csv'), sep='\t', header=False, index=False)

    business_slirec = business[['business_id', 'categories']]
    business_slirec.to_csv(os.path.join(dirs, 'yelp_business_recommenders.csv'), sep='\t', header=False, index=False)


def yelp_main(reviews_file, meta_file):

    dirs, _ = os.path.split(reviews_file)
    review, business, category_level_1 = load_data(reviews_file, meta_file, dirs)


    k_core = 10
    review, business = filter(review, business, category_level_1, k_core, dirs)

    transform_recommenders(review, business, dirs)

    reviews_output = os.path.join(dirs, 'yelp_review_recommenders.csv')
    meta_output = os.path.join(dirs, 'yelp_business_recommenders.csv')

    return reviews_output, meta_output


def filter_items_with_multiple_cids(record):

    item_cate = record[['iid', 'category']].drop_duplicates().groupby('iid').count().reset_index().rename(columns={'category': 'count'})
    items_with_single_cid = item_cate[item_cate['count'] == 1]['iid']

    record = pd.merge(record, items_with_single_cid, on='iid')

    return record


def downsample(record, col, frac):

    sample_col = record[col].drop_duplicates().sample(frac=frac)

    record = record.merge(sample_col, on=col).reset_index(drop=True)

    return record


def taobao_main(reviews_file):

    reviews = pd.read_csv(reviews_file, header=None, names=['uid', 'iid', 'category', 'behavior', 'ts'])
    reviews = reviews[reviews['behavior'] == 'pv']
    reviews = reviews.drop_duplicates(subset=['uid', 'iid'])
    reviews = filter_items_with_multiple_cids(reviews)
    start_ts = int(datetime.timestamp(datetime(2017, 11, 25, 0, 0, 0)))
    end_ts = int(datetime.timestamp(datetime(2017, 12, 3, 23, 59, 59)))
    reviews = reviews[reviews['ts'] >= start_ts]
    reviews = reviews[reviews['ts'] <= end_ts]
    reviews = downsample(reviews, 'uid', 0.05)

    k_core = 10
    reviews = filter_k_core(reviews, k_core, 'iid', 'uid')
    reviews = filter_k_core(reviews, k_core, 'uid', 'iid')

    business = reviews[['iid', 'category']].drop_duplicates()
    reviews = reviews[['uid', 'iid', 'ts']]

    dirs, _ = os.path.split(reviews_file)

    reviews_output = os.path.join(dirs, 'taobao_review_recommenders.csv')
    meta_output = os.path.join(dirs, 'taobao_business_recommenders.csv')

    reviews.to_csv(reviews_output, sep='\t', header=False, index=False)
    business.to_csv(meta_output, sep='\t', header=False, index=False)

    return reviews_output, meta_output


def taobao_strong_main(strong_last_vocab, strong_first_vocab, strong_behavior_file, user_vocab, item_vocab):

    strong_behavior = pd.read_csv(strong_behavior_file, index_col=0)
    user_dict = load_dict(user_vocab)
    item_dict = load_dict(item_vocab)
    uids = pd.Series([int(uid) for uid in user_dict.keys() if uid != 'default_uid'], name='uid', dtype='int64')
    iids = pd.Series([int(iid) for iid in item_dict.keys() if iid != 'default_mid'], name='iid', dtype='int64')

    strong_behavior = strong_behavior.merge(uids, on='uid')
    strong_behavior = strong_behavior.merge(iids, on='iid')

    dirs, _ = os.path.split(strong_behavior_file)
    strong_behavior_output = os.path.join(dirs, 'taobao_strong_behavior.csv')
    strong_behavior.to_csv(strong_behavior_output, sep='\t', header=None, index=False)

    strong_last_behavior = strong_behavior.sort_values('ts').groupby('uid').tail(1).reset_index()
    strong_first_behavior = strong_behavior.sort_values('ts').groupby('uid').head(1).reset_index()

    strong_last_behavior_vocab = dict(zip(strong_last_behavior['uid'].to_numpy(), strong_last_behavior[['iid', 'category', 'ts']].to_numpy()))
    cPickle.dump(strong_last_behavior_vocab, open(strong_last_vocab, "wb"))
    strong_first_behavior_vocab = dict(zip(strong_first_behavior['uid'].to_numpy(), strong_first_behavior[['iid', 'category', 'ts']].to_numpy()))
    cPickle.dump(strong_first_behavior_vocab, open(strong_first_vocab, "wb"))


def kuaishou_open_strong_main(strong_like_last_vocab, strong_like_first_vocab, strong_follow_last_vocab, strong_follow_first_vocab, strong_behavior_file, user_vocab, item_vocab, category_file):

    with open(strong_behavior_file, 'rb') as f:
        train_interaction_data = cPickle.load(f) 
        test_interaction_data = cPickle.load(f) 
        _ = cPickle.load(f) 
        _ = cPickle.load(f) 
        _ = cPickle.load(f) 
        _ = cPickle.load(f)

    data = train_interaction_data + test_interaction_data
    like_data = [d[0] for d in data if d[0][3] == 1]
    follow_data = [d[0] for d in data if d[0][4] == 1]

    reviews_like = pd.DataFrame(like_data, columns=['uid', 'iid', 'click', 'like', 'follow', 'ts', 'playing_time', 'duration_time'], dtype='int64')
    reviews_like = reviews_like[['uid', 'iid', 'ts']]
    reviews_like = reviews_like.drop_duplicates(subset=['uid', 'iid'])

    reviews_follow = pd.DataFrame(follow_data, columns=['uid', 'iid', 'click', 'like', 'follow', 'ts', 'playing_time', 'duration_time'], dtype='int64')
    reviews_follow = reviews_follow[['uid', 'iid', 'ts']]
    reviews_follow = reviews_follow.drop_duplicates(subset=['uid', 'iid'])

    user_dict = load_dict(user_vocab)
    item_dict = load_dict(item_vocab)
    uids = pd.Series([int(uid) for uid in user_dict.keys() if uid != 'default_uid'], name='uid', dtype='int64')
    iids = pd.Series([int(iid) for iid in item_dict.keys() if iid != 'default_mid'], name='iid', dtype='int64')
    categories = pd.read_csv(category_file, sep='\t', header=None, names=['iid', 'category'])

    reviews_like = reviews_like.merge(uids, on='uid')
    reviews_like = reviews_like.merge(iids, on='iid')
    reviews_like = reviews_like.merge(categories, on='iid')

    reviews_follow = reviews_follow.merge(uids, on='uid')
    reviews_follow = reviews_follow.merge(iids, on='iid')
    reviews_follow = reviews_follow.merge(categories, on='iid')

    dirs, _ = os.path.split(strong_behavior_file)
    reviews_like_output = os.path.join(dirs, 'kuaishou_open_like.csv')
    reviews_like.to_csv(reviews_like_output, sep='\t', header=None, index=False)
    reviews_follow_output = os.path.join(dirs, 'kuaishou_open_follow.csv')
    reviews_follow.to_csv(reviews_follow_output, sep='\t', header=None, index=False)

    like_last_behavior = reviews_like.sort_values('ts').groupby('uid').tail(1).reset_index()
    like_first_behavior = reviews_like.sort_values('ts').groupby('uid').head(1).reset_index()
    follow_last_behavior = reviews_follow.sort_values('ts').groupby('uid').tail(1).reset_index()
    follow_first_behavior = reviews_follow.sort_values('ts').groupby('uid').head(1).reset_index()

    like_last_behavior_vocab = dict(zip(like_last_behavior['uid'].to_numpy(), like_last_behavior[['iid', 'category', 'ts']].to_numpy()))
    cPickle.dump(like_last_behavior_vocab, open(strong_like_last_vocab, "wb"))
    like_first_behavior_vocab = dict(zip(like_first_behavior['uid'].to_numpy(), like_first_behavior[['iid', 'category', 'ts']].to_numpy()))
    cPickle.dump(like_first_behavior_vocab, open(strong_like_first_vocab, "wb"))
    follow_last_behavior_vocab = dict(zip(follow_last_behavior['uid'].to_numpy(), follow_last_behavior[['iid', 'category', 'ts']].to_numpy()))
    cPickle.dump(follow_last_behavior_vocab, open(strong_follow_last_vocab, "wb"))
    follow_first_behavior_vocab = dict(zip(follow_first_behavior['uid'].to_numpy(), follow_first_behavior[['iid', 'category', 'ts']].to_numpy()))
    cPickle.dump(follow_first_behavior_vocab, open(strong_follow_first_vocab, "wb"))


def statistics_ks(df):
    print('length:', len(df))
    print('num of users:', df['uid'].nunique())
    print('num of items:', df['iid'].nunique())
    print('num of positives:', len(df[df['effective_view']==1]))
    print('num of negtives:', len(df[df['effective_view']==0]))
    his = df[['uid', 'iid']].groupby('uid').count().reset_index()
    his_l = his['iid'].to_numpy()
    print('mean of his', his_l.mean())
    print('max of his', his_l.max())
    print('min of his:', his_l.min())
    print('median of his:', np.median(his_l))


def kuaishou_main(reviews_file):

    #  reviews = pd.read_csv(reviews_file, header=None, names=['uid', 'iid', 'category', 'behavior', 'ts'])
    reviews = pd.read_csv(reviews_file, header=0)
    reviews = reviews.rename(columns={
        'time_ms':'ts',
        'user_id': 'uid',
        'photo_id': 'iid',
        #  'effective_view': 'behavior', # all=1
        'photo_kmeans_cluster_id': 'category'
    })

    #  reviews = reviews[reviews['behavior'] == 'pv']
    # add new column
    reviews['behavior'] = 'pv'
    reviews = reviews.drop_duplicates(subset=['uid', 'iid'])
    #  reviews = filter_items_with_multiple_cids(reviews)
    #  reviews = downsample(reviews, 'uid', 0.05)

    # need raw data rather than filtered data
    k_core = 10
    reviews = filter_k_core(reviews, k_core, 'iid', 'uid')
    # DONE filter users according to only pos items.
    reviews = filter_k_core_consider_neg(reviews, k_core, 'uid', 'iid', 'effective_view')

    # removes negtives for now
    reviews = reviews[reviews['effective_view']==1]

    # DONE: add data statistics
    statistics_ks(reviews)

    business = reviews[['iid', 'category']].drop_duplicates()
    reviews = reviews[['uid', 'iid', 'ts']]

    dirs, _ = os.path.split(reviews_file)

    reviews_output = os.path.join(dirs, 'kuaishou_review_recommenders.csv')
    meta_output = os.path.join(dirs, 'kuaishou_business_recommenders.csv')

    reviews.to_csv(reviews_output, sep='\t', header=False, index=False)
    business.to_csv(meta_output, sep='\t', header=False, index=False)

    return reviews_output, meta_output


def get_categories_by_clustering(meta_file, num_centroids, items):

    visual_feature = np.load(meta_file)
    item_embed = visual_feature[items].astype('float32')

    _, assignments = kmeans_cuda(item_embed, num_centroids, verbosity=1, seed=43)

    return assignments


def kuaishou_open_main(reviews_file, meta_file):

    with open(reviews_file, 'rb') as f:
        train_interaction_data = cPickle.load(f) 
        test_interaction_data = cPickle.load(f) 
        _ = cPickle.load(f) 
        _ = cPickle.load(f) 
        _ = cPickle.load(f) 
        _ = cPickle.load(f)

    data = train_interaction_data + test_interaction_data
    data = [d[0] for d in data if d[0][2] == 1]

    reviews = pd.DataFrame(data, columns=['uid', 'iid', 'click', 'like', 'follow', 'ts', 'playing_time', 'duration_time'], dtype='int64')
    reviews = reviews[['uid', 'iid', 'ts']]
    reviews = reviews.drop_duplicates(subset=['uid', 'iid'])

    k_core = 10
    reviews = filter_k_core(reviews, k_core, 'iid', 'uid')
    reviews = filter_k_core(reviews, k_core, 'uid', 'iid')

    items = reviews['iid'].unique()
    num_centroids = 500
    categories = get_categories_by_clustering(meta_file, num_centroids, items)
    business = pd.DataFrame({'iid': items, 'category': categories})

    dirs, _ = os.path.split(reviews_file)

    reviews_output = os.path.join(dirs, 'kuaishou_open_review_recommenders.csv')
    meta_output = os.path.join(dirs, 'kuaishou_open_business_recommenders.csv')

    reviews.to_csv(reviews_output, sep='\t', header=False, index=False)
    business.to_csv(meta_output, sep='\t', header=False, index=False)

    return reviews_output, meta_output
