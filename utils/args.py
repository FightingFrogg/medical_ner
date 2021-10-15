# -*- coding: utf-8 -*-
# author:SHAN
# datetime:2021/9/29 16:48

import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default=False)
    parser.add_argument('--eval', default=True)
    parser.add_argument('--data_dir', default='./data')
    parser.add_argument('--bert_dir', default='./bert/torch_roberta_wwm')
    parser.add_argument('--output_dir', default='./save')
    parser.add_argument('--log_output_dir', default='./log')
    parser.add_argument('--max_seq_len', default=512)
    parser.add_argument('--batch_size', default=8)
    parser.add_argument('--eval_batch_size', default=64)
    parser.add_argument('--epoch', default=10)
    parser.add_argument('--weight_decay', default=0.01)
    parser.add_argument('--lr', default=2e-5)
    parser.add_argument('--crf_lr', default=2e-3)       # crf 的 lr 较大
    parser.add_argument('--adamw_epsilon', default=1e-8)
    parser.add_argument('--warmup_proportion', default=0.1)
    parser.add_argument('--max_grad_norm', default=1.0)
    args = parser.parse_args()

    return args
