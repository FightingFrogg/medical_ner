# -*— coding: utf-8 -*-
# Name:main.py
# Author:SHAN
# Datetime:2021/9/29 上午9:36

import json
import os
from processer import *
from utils.args import get_args
from transformers import BertTokenizer, BertPreTrainedModel, BertConfig
from model import CRFModel
from train import train



def main():
    args = get_args()

    with open(os.path.join(args.data_dir, 'crf_ent2id.json')) as f:
        ent2id = json.load(f)

    tokenizer = BertTokenizer(os.path.join(args.bert_dir, 'vocab.txt'))
    config = BertConfig(os.path.join(args.bert_dir, 'config.json'), num_labels=len(ent2id))
    model = CRFModel.from_pretrained(os.path.join(args.bert_dir, 'pytorch_model.bin'), config=config)

    train(args, model, tokenizer, ent2id)




if __name__ == '__main__':
    main()
