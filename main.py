# -*— coding: utf-8 -*-
# Name:main.py
# Author:SHAN
# Datetime:2021/9/29 上午9:36

import json
import os

import torch.types

from processer import *
from utils.args import get_args
from transformers import BertTokenizer, BertPreTrainedModel, BertConfig
from model import CRFModel
from train import train, eval



def main():
    args = get_args()

    with open(os.path.join(args.data_dir, 'crf_ent2id.json')) as f:
        ent2id = json.load(f)

    tokenizer = BertTokenizer(os.path.join(args.bert_dir, 'vocab.txt'))
    config = BertConfig(os.path.join(args.bert_dir, 'config.json'), num_labels=len(ent2id))
    model = CRFModel.from_pretrained(os.path.join(args.bert_dir, 'pytorch_model.bin'), config=config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    if args.train:
        train(args, model, tokenizer, ent2id)

        # 保存模型
        model.save_pretrain(args.output_dir)
        # 保存 tokenizer
        tokenizer.save_vocabulary(args.output_dir)

    if args.eval:
        # 加载保存的 tokenizer
        tokenizer = BertTokenizer.from_pretrained(args.output_dir, do_lower_case=True)
        model = CRFModel.from_pretrained(args.output_dir, config=config)
        model.to(args.device)

        eval(args, model, tokenizer, ent2id)


if __name__ == '__main__':
    main()
