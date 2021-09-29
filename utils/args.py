# -*- coding: utf-8 -*-
# author:SHAN
# datetime:2021/9/29 16:48

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--bert_dir', default='./bert/torch_roberta_wwm')
    parser.add_argument('--max_seq_len', default=512)
    args = parser.parse_args()

    return args
