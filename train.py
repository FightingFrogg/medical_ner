# -*- coding: utf-8 -*-
# author:SHAN
# datetime:2021/9/30 15:19

from processer import *
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup


def train(args, model, tokenizer, ent2id):

    features = convert2features(args, 'train', tokenizer, ent2id)

    all_token_idx = torch.tensor([f.token_idx for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.labels for f in features], dtype=torch.long)
    all_token_type_idx = torch.tensor([f.token_type_idx for f in features], dtype=torch.long)

    data_set = TensorDataset(all_token_idx, all_token_type_idx, all_attention_mask, all_labels)

    data_sampler = RandomSampler(data_set)
    loader = DataLoader(dataset=data_set, sampler=data_sampler, batch_size=args.batch_size)

    t_total = len(loader) * args.epoch
    optimizer, scheduler = build_optimizer_scheduler(args, model, t_total)

    model.to(args.device)

    for epoch in range(args.epoch):
        for step, batch_data in enumerate(loader):
            model.train()

            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(args.device)

            loss = model(**batch_data)
            loss.backward()
            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            torch.cuda.empty_cache()


def eval(args, model, tokenizer, ent2id):
    features = convert2features(args, 'dev', tokenizer, ent2id)

    all_token_idx = torch.tensor([f.token_idx for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.labels for f in features], dtype=torch.long)
    all_token_type_idx = torch.tensor([f.token_type_idx for f in features], dtype=torch.long)

    data_set = TensorDataset(all_token_idx, all_token_type_idx, all_attention_mask, all_labels)

    data_sampler = RandomSampler(data_set)
    loader = DataLoader(dataset=data_set, sampler=data_sampler, batch_size=args.batch_size)

    for step,batch_data in enumerate(loader):
        model.eval()
        batch_data = tuple(t.to(args.device) for t in batch_data)

        with torch.no_grad():
            inputs = {'emission': batch_data[0], 'mask': batch_data[2]}
            out = model(**inputs)

        print(out.shape())
        print(batch_data[3].shape())



def build_optimizer_scheduler(args, model, t_total):

    # 差分学习率，不同的模块使用不同的学习率
    no_decay = ["bias", "LayerNorm.weight"]
    bert_param_optim = list(model.bert.named_parameters())
    crf_param_optim = list(model.crf.named_parameters())
    classifier_param_optim = list(model.classifier.named_parameters())

    optimizer_grouped_parameters = [
        {"param": [param for name, param in bert_param_optim if not any(nd in no_decay for nd in name)],
         "weight_decay": args.weight_decay, "lr": args.lr},
        {"param": [param for name, param in bert_param_optim if any(nd in no_decay for nd in name)],
         "weight_decay": 0.0, "lr": args.lr},

        {"param": [param for name, param in crf_param_optim if not any(nd in no_decay for nd in name)],
         "weight_decay": args.weight_decay, "lr": args.crf_lr},
        {"param": [param for name, param in crf_param_optim if any(nd in no_decay for nd in name)],
         "weight_decay": 0.0, "lr": args.crf_lr},

        {"param": [param for name, param in classifier_param_optim if not any(nd in no_decay for nd in name)],
         "weight_decay": args.weight_decay, "lr": args.crf_lr},
        {"param": [param for name, param in classifier_param_optim if any(nd in no_decay for nd in name)],
         "weight_decay": 0.0, "lr": args.crf_lr},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adamw_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * t_total),
                                                num_training_steps=t_total)

    return optimizer, scheduler
