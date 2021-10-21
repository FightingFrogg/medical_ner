# -*— coding: utf-8 -*-
# Name:main.py
# Author:SHAN
# Datetime:2021/9/29 上午9:36

import json
import os
import logging
import torch
import time

from processer import *
from utils.args import get_args
from transformers import BertTokenizer, BertPreTrainedModel, BertConfig
from model import CRFModel, SpanModel
from train import train, eval, span_eval


def init_log(args, info='log'):
    log_dir = args.log_output_dir
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    _time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    log_file = log_dir + f'/{info}_{_time}.log'

    log_format = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(name)s : %(message)s ', datefmt='%m/%d/%Y %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)   # 设置日志器将会处理的日志消息的最低严重级别
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def main():
    args = get_args()
    init_log(args)      # info='log'
    logging.info('-------Start-------')

    if args.task_type == 'crf':
        with open(os.path.join(args.data_dir, 'crf_ent2id.json')) as f:
            ent2id = json.load(f)
    elif args.task_type == 'span':
        with open(os.path.join(args.data_dir, 'span_ent2id.json')) as f:
            ent2id = json.load(f)

    # hfl/chinese-roberta-wwm-ext
    # tokenizer = BertTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext')
    # config = BertConfig.from_pretrained('hfl/chinese-roberta-wwm-ext')
    # model = CRFModel.from_pretrained('hfl/chinese-roberta-wwm-ext', config=config)
    config = BertConfig.from_pretrained(args.bert_dir)
    config.num_labels = len(ent2id)
    tokenizer = BertTokenizer(os.path.join(args.bert_dir, 'vocab.txt'))

    model = None
    if args.task_type == 'crf':
        model = CRFModel.from_pretrained(args.bert_dir, config=config)
    elif args.task_type == 'span':
        model = SpanModel.from_pretrained(args.bert_dir, config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    if args.train:
        logging.info('Train...')
        train(args, model, tokenizer, ent2id)

        # 保存模型
        model.save_pretrained(args.output_dir)
        # 保存 tokenizer
        tokenizer.save_vocabulary(args.output_dir)

    if args.eval:
        logging.info('Evaluation...')
        # 加载保存的 tokenizer
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=True)
        if args.task_type == 'crf':
            model = CRFModel.from_pretrained(args.output_dir, config=config)
        elif args.task_type == 'span':
            model = SpanModel.from_pretrained(args.output_dir, config=config)
        model.to(args.device)

        if args.task_type == 'crf':
            all_metric = eval(args, model, tokenizer, ent2id)
        elif args.task_type == 'span':
            all_metric = span_eval(args, model, tokenizer, ent2id)

        logging.info('Metric')
        logging.info('Precision: {} Recall: {} F1: {}\n'
                     .format(round(all_metric['precision'], 4), round(all_metric['recall'], 4), round(all_metric['f1'], 4)))


if __name__ == '__main__':
    main()
