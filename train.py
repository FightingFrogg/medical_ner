# -*- coding: utf-8 -*-
# author:SHAN
# datetime:2021/9/30 15:19
import json
import os.path

from processer import *
import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset
from tqdm import trange, tqdm
from utils.metric import Metric
from model import FGM, PGD


class NERDataset(Dataset):
    def __init__(self, task_type, features, mode):

        self.nums = len(features)

        self.all_token_idx = torch.tensor([f.token_idx for f in features], dtype=torch.long)
        self.all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        self.all_token_type_idx = torch.tensor([f.token_type_idx for f in features], dtype=torch.long)

        self.all_labels = None
        self.all_start_idx, self.all_end_idx = None, None
        self.all_entity_type = None

        if mode == 'train':
            if task_type == 'crf':
                self.all_labels = torch.tensor([f.labels for f in features], dtype=torch.long)
            elif task_type == 'span':
                self.all_start_idx = torch.tensor([f.start_idx for f in features], dtype=torch.long)
                self.all_end_idx = torch.tensor([f.end_idx for f in features], dtype=torch.long)
            elif task_type == 'mrc':
                self.all_start_idx = torch.tensor([f.start_idx for f in features], dtype=torch.long)
                self.all_end_idx = torch.tensor([f.end_idx for f in features], dtype=torch.long)
                self.all_entity_type = torch.tensor([f.entity_type for f in features])

        elif mode == 'eval':
            if task_type == 'crf':
                self.all_labels = torch.tensor([f.labels for f in features], dtype=torch.long)
            elif task_type == 'span':
                self.all_start_idx = torch.tensor([f.start_idx for f in features], dtype=torch.long)
                self.all_end_idx = torch.tensor([f.end_idx for f in features], dtype=torch.long)
            elif task_type == 'mrc':
                self.all_start_idx = torch.tensor([f.start_idx for f in features], dtype=torch.long)
                self.all_end_idx = torch.tensor([f.end_idx for f in features], dtype=torch.long)
                self.all_entity_type = torch.tensor([f.entity_type for f in features])

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {'token_idx': self.all_token_idx[index],
                'attention_mask': self.all_attention_mask[index],
                'token_type_idx': self.all_token_type_idx[index]}

        if self.all_labels is not None:
            data['labels'] = self.all_labels[index]

        if self.all_start_idx is not None:
            data['start_idx'] = self.all_start_idx[index]
            data['end_idx'] = self.all_end_idx[index]

        if self.all_entity_type is not None:
            data['entity_type'] = self.all_entity_type[index]

        return data

class MRCDataset(Dataset):
    def __init__(self, data_dict, mode):
        self.nums = len(data_dict)

        self.all_token_idx = torch.tensor([data_dict[i]['token_idx'] for i in data_dict], dtype=torch.long)
        self.all_attention_mask = torch.tensor([data_dict[i]['attention_mask'] for i in data_dict], dtype=torch.long)
        self.all_token_type_idx = torch.tensor([data_dict[i]['token_type_idx'] for i in data_dict], dtype=torch.long)

        self.all_start_idx, self.all_end_idx = None, None
        self.all_entity_type = None

        if mode == 'train':
            self.all_start_idx = torch.tensor([data_dict[i]['start_idx'] for i in data_dict], dtype=torch.long)
            self.all_end_idx = torch.tensor([data_dict[i]['end_idx'] for i in data_dict], dtype=torch.long)
            self.all_entity_type = torch.tensor([data_dict[i]['entity_type'] for i in data_dict])

        elif mode == 'eval':
            self.all_start_idx = torch.tensor([data_dict[i]['start_idx'] for i in data_dict], dtype=torch.long)
            self.all_end_idx = torch.tensor([data_dict[i]['end_idx'] for i in data_dict], dtype=torch.long)
            self.all_entity_type = torch.tensor([data_dict[i]['entity_type'] for i in data_dict])

    def __len__(self):
        return self.nums

    def __getitem__(self, index):
        data = {'token_idx': self.all_token_idx[index],
                'attention_mask': self.all_attention_mask[index],
                'token_type_idx': self.all_token_type_idx[index],
                'start_idx': self.all_start_idx[index],
                'end_idx': self.all_end_idx[index],
                'entity_type': self.all_entity_type[index]}

        return data


def train(args, model, tokenizer, ent2id):
    data_set = None
    if args.task_type == 'crf':
        features = convert_crf_features(args, 'train', tokenizer, ent2id)
        data_set = NERDataset('crf', features, 'train')
    elif args.task_type == 'span':
        features = convert_span_features(args, 'train', tokenizer, ent2id)
        data_set = NERDataset('span', features, 'train')
    elif args.task_type == 'mrc':
        if os.path.exists(args.data_dir + '/mrc_dataSet.json'):
            with open(args.data_dir + '/mrc_dataSet.json', 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            data_set = MRCDataset(data_dict, 'train')
        else:
            features = convert_mrc_features(args, 'train', tokenizer, ent2id)
            data_set = NERDataset('mrc', features, 'train')
            data_dict = {}
            for i, data in enumerate(data_set):
                for key in data.keys():
                    data[key] = data[key].tolist()
                data_dict[i] = data
            with open(args.data_dir + '/mrc_dataSet.json', 'w', encoding='utf-8') as f:
                json.dump(data_dict, f)
                logging.info('MRC dataset 写入 json 文件')

    data_sampler = RandomSampler(data_set)
    loader = DataLoader(dataset=data_set, sampler=data_sampler, batch_size=args.batch_size)

    t_total = len(loader) * args.epoch
    optimizer, scheduler = build_optimizer_scheduler(args, model, t_total)

    model.to(args.device)

    fgm = None
    pgd = None
    pgd_k = None
    if args.attack == 'FGM':
        fgm = FGM(model)
    elif args.attack == 'PGD':
        pgd = PGD(model)
        pgd_k = 3

    for epoch in trange(args.epoch):
        for step, batch_data in enumerate(tqdm(loader)):
            model.train()

            for key in batch_data.keys():
                batch_data[key] = batch_data[key].to(args.device)

            loss = model(**batch_data)[0]
            loss.backward()

            # 加入对抗
            if fgm is not None:
                fgm.attack()
                loss_adv = model(**batch_data)[0]
                loss_adv.backward()
                fgm.restore()
            elif pgd is not None:
                pgd.backup_grad()

                for pgd_i in range(pgd_k):      # pgd 循环 3 次
                    pgd.attack(is_first_attack=(pgd_i==0))
                    if pgd_i != pgd_k - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    loss_adv = model(**batch_data)[0]
                    loss_adv.backward()
                pgd.restore()

            print('loss: ', loss.item())
            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            torch.cuda.empty_cache()


def eval(args, model, tokenizer, ent2id):

    features = convert_crf_features(args, 'eval', tokenizer, ent2id)

    id2ent = {ent2id[key]: key for key in ent2id.keys()}        # {1:'B-DRUG_GROUP',  ...}
    metric = Metric(id2ent=id2ent)

    all_token_idx = torch.tensor([f.token_idx for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_labels = torch.tensor([f.labels for f in features], dtype=torch.long)
    all_token_type_idx = torch.tensor([f.token_type_idx for f in features], dtype=torch.long)

    data_set = TensorDataset(all_token_idx, all_token_type_idx, all_attention_mask, all_labels)

    data_sampler = RandomSampler(data_set)
    loader = DataLoader(dataset=data_set, sampler=data_sampler, batch_size=args.batch_size)

    pred_outputs = []
    for step, batch_data in enumerate(tqdm(loader)):
        model.eval()
        batch_data = tuple(t.to(args.device) for t in batch_data)

        with torch.no_grad():
            inputs = {'token_idx': batch_data[0], 'token_type_idx': batch_data[1], 'attention_mask': batch_data[2]}
            out = model(**inputs)
            pred_outputs.append(out[0])

        labels = batch_data[3].cpu().numpy().tolist()
        tags = out[0]       # 包括了 cls 和 sep
        for i, tag in enumerate(tags):
            temp_tags = []
            temp_labels = []
            for j, t in enumerate(tag):
                if j == 0:
                    continue        # 跳过 cls
                elif j == len(tag) - 2:       # 跳过 sep
                    metric.update([temp_labels], [temp_tags], 'crf')
                    break
                else:
                    temp_labels.append(id2ent[labels[i][j]])
                    temp_tags.append(id2ent[tag[j]])

    # 计算 p, r, f
    all_metric, entity_metric = metric.result()

    return all_metric


def span_eval(args, model, tokenizer, ent2id):
    id2ent = {ent2id[key]: key for key in ent2id.keys()}        # {1:'B-DRUG_GROUP',  ...}
    metric = Metric(id2ent=id2ent)

    features = convert_span_features(args, 'eval', tokenizer, ent2id)
    data_set = NERDataset('span', features, 'train')
    data_sampler = RandomSampler(data_set)

    loader = DataLoader(dataset=data_set, sampler=data_sampler, batch_size=args.batch_size)
    for step, batch_data in enumerate(tqdm(loader)):
        model.eval()
        for key in batch_data.keys():
            batch_data[key] = batch_data[key].to(args.device)

        with torch.no_grad():
            inputs = {'token_idx': batch_data['token_idx'], 'token_type_idx': batch_data['token_type_idx'],
                      'attention_mask': batch_data['attention_mask']}
            out = model(**inputs)

        pre = bert_extract_item(out[0], out[1], 'pre')       # (11, 8, 9)
        labels = bert_extract_item(batch_data['start_idx'], batch_data['end_idx'], 'labels')
        for i in range(len(pre)):
            metric.update(labels[i], pre[i], 'span')

    # 计算 p, r, f
    all_metric, entity_metric = metric.result()

    return all_metric


def mrc_eval(args, model, tokenizer, ent2id):
    id2ent = {ent2id[key]: key for key in ent2id.keys()}  # {1:'B-DRUG_GROUP',  ...}
    metric = Metric(id2ent=id2ent)

    if os.path.exists(args.data_dir + '/mrc_dataSet.json'):
        with open(args.data_dir + '/mrc_dataSet.json', 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        data_set = MRCDataset(data_dict, 'eval')
    else:
        features = convert_mrc_features(args, 'eval', tokenizer, ent2id)
        data_set = NERDataset('mrc', features, 'eval')
        data_dict = {}
        for i, data in enumerate(data_set):
            for key in data.keys():
                data[key] = data[key].tolist()
            data_dict[i] = data
        with open(args.data_dir + '/mrc_dataSet.json', 'w', encoding='utf-8') as f:
            json.dump(data_dict, f)
            logging.info('MRC dataset 写入 json 文件')

    data_sampler = RandomSampler(data_set)
    loader = DataLoader(dataset=data_set, sampler=data_sampler, batch_size=args.batch_size)
    for step, batch_data in enumerate(tqdm(loader)):
        model.eval()
        for key in batch_data.keys():
            batch_data[key] = batch_data[key].to(args.device)

        with torch.no_grad():
            inputs = {'token_idx': batch_data['token_idx'], 'token_type_idx': batch_data['token_type_idx'],
                      'attention_mask': batch_data['attention_mask']}
            out = model(**inputs)

        pre = bert_extract_item(out[0], out[1], 'pre')
        labels = bert_extract_item(batch_data['start_idx'], batch_data['end_idx'], 'labels')
        for i in range(len(pre)):
            metric.update(labels[i], pre[i], 'span')

    # 计算 p, r, f
    all_metric, entity_metric = metric.result()

    return all_metric


def bert_extract_item(start_logits, end_logits, flag):
    SS = []
    for i in range(len(start_logits)):
        S = []
        if flag == 'pre':
            start = torch.argmax(start_logits, -1).cpu().numpy()[i][1:-1]
            end = torch.argmax(end_logits, -1).cpu().numpy()[i][1:-1]
        elif flag == 'labels':
            start = start_logits.cpu().numpy()[i][1:-1]
            end = end_logits.cpu().numpy()[i][1:-1]
        for i, s_l in enumerate(start):
            if s_l == 0:
                continue
            for j, e_l in enumerate(end[i:]):
                if s_l == e_l:
                    S.append((s_l, i, i + j))
                    break
        SS.append(S)
    return SS


def build_optimizer_scheduler(args, model, t_total):
    model = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率，不同的模块使用不同的学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(model.named_parameters())

    bert_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        _name = name.split('.')
        if _name[0] == 'bert':
            bert_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    # bert_param_optim = list(model.bert.named_parameters())
    # crf_param_optim = list(model.crf.named_parameters())
    # classifier_param_optim = list(model.classifier.named_parameters())
    #
    # optimizer_grouped_parameters = [
    #     {"params": [param for name, param in bert_param_optim if not any(nd in no_decay for nd in name)],
    #      "weight_decay": args.weight_decay, "lr": args.lr},
    #     {"params": [param for name, param in bert_param_optim if any(nd in no_decay for nd in name)],
    #      "weight_decay": 0.0, "lr": args.lr},
    #
    #     {"params": [param for name, param in crf_param_optim if not any(nd in no_decay for nd in name)],
    #      "weight_decay": args.weight_decay, "lr": args.crf_lr},
    #     {"params": [param for name, param in crf_param_optim if any(nd in no_decay for nd in name)],
    #      "weight_decay": 0.0, "lr": args.crf_lr},
    #
    #     {"params": [param for name, param in classifier_param_optim if not any(nd in no_decay for nd in name)],
    #      "weight_decay": args.weight_decay, "lr": args.crf_lr},
    #     {"params": [param for name, param in classifier_param_optim if any(nd in no_decay for nd in name)],
    #      "weight_decay": 0.0, "lr": args.crf_lr},
    # ]
    optimizer_grouped_parameters = [
        {"params": [param for name, param in bert_param_optimizer if not any(nd in no_decay for nd in name)],
         "weight_decay": args.weight_decay, "lr": args.lr},
        {"params": [param for name, param in bert_param_optimizer if any(nd in no_decay for nd in name)],
         "weight_decay": 0.0, "lr": args.lr},

        {"params": [param for name, param in other_param_optimizer if not any(nd in no_decay for nd in name)],
         "weight_decay": args.weight_decay, "lr": args.crf_lr},
        {"params": [param for name, param in other_param_optimizer if any(nd in no_decay for nd in name)],
         "weight_decay": 0.0, "lr": args.crf_lr},
    ]


    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adamw_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(args.warmup_proportion * t_total),
                                                num_training_steps=t_total)

    return optimizer, scheduler
