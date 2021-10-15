# -*- coding: utf-8 -*-
# author:SHAN
# datetime:2021/9/29 16:35

import json
import os


class CRFFeature():
    def __init__(self, token_idx, attention_mask, token_type_idx, labels, text):
        self.token_idx = token_idx
        self.attention_mask = attention_mask
        self.token_type_idx = token_type_idx
        self.labels = labels
        self.text = text


def read_data(file_name):

    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = sorted(data, key=lambda x: x['id'])

    return data


def convert2features(args, data_type, tokenizer, ent2id):
    examples = None
    features = []

    if data_type == 'train':
        examples = read_data(os.path.join(args.data_dir, 'train.json'))     # 850 组数据
        # dict_keys(['id', 'text', 'labels', 'pseudo', 'candidate_entities'])
    elif data_type == 'eval':
        examples = read_data(os.path.join(args.data_dir, 'dev.json'))       # 150 组数据

    for i, example in enumerate(examples):
        text = example['text']      # 76 个字。 len: 75
        entities = example['labels']
        tokens = fine_grade_tokenize(text, tokenizer)       # 分词，将字转换成 token，将空格和未登录的词转换成指定的 tonken。 len: 76
        encode_dict = tokenizer.encode_plus(text=tokens,
                                           max_length=args.max_seq_len,
                                           pad_to_max_length=True,
                                           return_token_type_ids=True,
                                           return_attention_mask=True,
                                           is_pretokenize=True)
        tokens_idx = encode_dict['input_ids']           # 将 token 转换成 idx，并在首尾添加 cls 和 sep 标签，并补零。 有效长度 len: 78
        token_type_idx = encode_dict['token_type_ids']  # 只有一句话，全为 0
        attention_mask = encode_dict['attention_mask']  # 将 cls 和 sep 标签一并进行 mask。有效长度 len: 78

        labels = [0] * len(tokens)
        for entity in entities:
            ent = entity[1]
            ent_s = entity[2]
            ent_e = entity[3]
            if ent_s == ent_e-1:
                labels[ent_s] = ent2id['S-' + ent]
            else:
                labels[ent_s] = ent2id['B-' + ent]
                labels[ent_e-1] = ent2id['E-' + ent]
                for i in range(ent_s+1, ent_e-1):
                    labels[i] = ent2id['I-' + ent]

        if len(labels) > args.max_seq_len - 2:
            labels = labels[:args.max_seq_len-2]
        elif len(labels) < args.max_seq_len - 2:
            labels = labels + [0] * (args.max_seq_len - 2 - len(labels))

        labels = [0] + labels + [0]
        assert len(labels) == args.max_seq_len

        features.append(CRFFeature(token_idx=tokens_idx,
                                   attention_mask=attention_mask,
                                   token_type_idx=token_type_idx,
                                   labels=labels,
                                   text=tokens))

    return features


def fine_grade_tokenize(text, tokenizer):
    """
    序列标注任务 BERT 分词器可能会导致标注偏移，
    用 char-level 来 tokenize
    """
    tokens = []
    for word in text:
        if word in [' ', '\t', '\n']:
            tokens.append('[BLANK]')
        else:
            if not len(tokenizer.encode(word)):
                tokens.append('[INV]')
            else:
                tokens.append(word)

    return tokens


def get_entities(seq, id2ent):
    """Gets entities from sequence.
    note: BIOS
    Args:
        seq (list): sequence of labels.
    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).
    Example:
        # >>> seq = ['B-PER', 'I-PER', 'O', 'S-LOC']
        # >>> get_entity_bios(seq)
        [['PER', 0, 1], ['LOC', 3, 3]]
    """
    entities = []
    idx_ = 0
    while idx_ < len(seq):
        if seq[idx_].startswith('S'):
            entity = [-1, -1, -1]
            entity_type = seq[idx_].split('-')[1]
            entity[0] = entity_type
            entity[1] = idx_
            entity[2] = idx_
            entities.append(entity)
            idx_ += 1
        elif seq[idx_].startswith('B'):
            entity = [-1, -1, -1]
            entity_type = seq[idx_].split('-')[1]
            entity[0] = entity_type
            entity[1] = idx_
            idx_ += 1
            while idx_ < len(seq):
                t_entity_type = seq[idx_].split('-')
                if t_entity_type[0].startswith('I') and t_entity_type[1] == entity[0]:
                    idx_ += 1
                elif t_entity_type[0].startswith('E') and t_entity_type[1] == entity[0]:
                    entity[2] = idx_
                    entities.append(entity)
                    idx_ += 1
                    break
                else:
                    break
        else:
            idx_ += 1

    return entities


