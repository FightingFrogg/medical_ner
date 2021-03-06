# -*- coding: utf-8 -*-
# author:SHAN
# datetime:2021/9/29 16:35

import json
import logging
import os
import re
from collections import defaultdict
from tqdm import tqdm

class Examples:
    def __init__(self,
                 set_type,
                 text,
                 labels=None):
        self.set_type = set_type
        self.text = text
        self.labels = labels


class BaseFeature():
    def __init__(self, token_idx, attention_mask, token_type_idx, text):
        self.token_idx = token_idx
        self.attention_mask = attention_mask
        self.token_type_idx = token_type_idx
        self.text = text


class CRFFeature(BaseFeature):
    def __init__(self, token_idx, attention_mask, token_type_idx, text, labels):
        super(CRFFeature, self).__init__(token_idx, attention_mask, token_type_idx, text)

        self.labels = labels


class SpanFeature(BaseFeature):
    def __init__(self, token_idx, attention_mask, token_type_idx, text, start_idx, end_idx):
        super(SpanFeature, self).__init__(token_idx, attention_mask, token_type_idx, text)

        self.start_idx = start_idx
        self.end_idx = end_idx


class MRCFeature(BaseFeature):
    def __init__(self, token_idx, attention_mask, token_type_idx, text, start_idx, end_idx, entity_type):
        super(MRCFeature, self).__init__(token_idx, attention_mask, token_type_idx, text)

        self.start_idx = start_idx
        self.end_idx = end_idx
        self.entity_type = entity_type


def read_data(file_name):

    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    data = sorted(data, key=lambda x: x['id'])

    return data


def get_sentence(raw_examples, set_type, max_seq_len=512):
    examples = []
    for i, example in enumerate(raw_examples):
        text = example['text']
        labels = example['labels']
        candidate_entities = example['candidate_entities']

        # 长句子截断
        sentences = cut_merge(text, max_seq_len)
        start_idx = 0

        # 句子截断后，label 的位置需要重新匹配
        for sent in sentences:
            new_labels = refactor_labels(sent, labels, start_idx)

            start_idx += len(sent)

            examples.append(Examples(set_type=set_type,
                                text=sent,
                                labels=new_labels))
    return examples


def cut_merge(text, max_seq_len):
    # 将句子分句，细粒度分句后再重新合并
    sentences = []

    # 细粒度划分
    sentences_v1 = cut_sentences_v1(text)
    for sent_v1 in sentences_v1:
        if len(sent_v1) > max_seq_len - 2:
            sent_v2 = cut_sentences_v2(sent_v1)
            sentences.extend(sent_v2)
        else:
            sentences.append(sent_v1)

    assert ''.join(sentences) == text

    # 合并
    merged_sentences = []
    start_index_ = 0

    while start_index_ < len(sentences):
        t_text = sentences[start_index_]
        end_index_ = start_index_ + 1

        while end_index_ < len(sentences) and len(t_text) + len(sentences[end_index_]) <= max_seq_len -2:
            t_text += sentences[end_index_]
            end_index_ += 1

        start_index_ = end_index_
        merged_sentences.append(t_text)

    return merged_sentences


def cut_sentences_v1(sent):
    """
    the first rank of sentence cut
    """
    sent = re.sub('([。！？\?])([^”’])', r"\1\n\2", sent)  # 单字符断句符，句号加一个非引号    \1代表与第一个小括号中要匹配的内容相同
    sent = re.sub('(\.{6})([^”’])', r"\1\n\2", sent)  # 英文省略号                         \2代表与第二个小括号中要匹配的内容相同
    sent = re.sub('(\…{2})([^”’])', r"\1\n\2", sent)  # 中文省略号
    sent = re.sub('([。！？\?][”’])([^，。！？\?])', r"\1\n\2", sent)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后
    return sent.split("\n")


def cut_sentences_v2(sent):
    """
    the second rank of spilt sentence, split '；' | ';'
    """
    sent = re.sub('([；;])([^”’])', r"\1\n\2", sent)
    return sent.split("\n")


def refactor_labels(sent, labels, start_idx):
    """
    分句后需要重构 labels 的 offset
    :param sent: 切分并重新合并后的句子
    :param labels: 原始文档级的 labels
    :param start_index: 该句子在文档中的起始 offset
    :return (type, entity, offset)
    """
    new_labels = []
    end_index = start_idx + len(sent)

    for _label in labels:
        if start_idx <= _label[2] <= _label[3] <= end_index:
            new_offset = _label[2] - start_idx

            assert sent[new_offset: new_offset + len(_label[-1])] == _label[-1]

            new_labels.append((_label[-1], _label[1], new_offset, new_offset+len(_label[-1])))
        # label 被截断的情况，发生错误，分句的时候有问题
        elif _label[2] < end_index < _label[3]:
            raise RuntimeError(f'{sent}, {_label}')

    return new_labels


def convert_crf_features(args, data_type, tokenizer, ent2id):
    examples = None
    features = []

    if data_type == 'train':
        raw_examples = read_data(os.path.join(args.data_dir, 'train.json'))     # 850 组数据
        # dict_keys(['id', 'text', 'labels', 'pseudo', 'candidate_entities'])
        # 长句子截断
        examples = get_sentence(raw_examples, 'train')

    elif data_type == 'eval':
        raw_examples = read_data(os.path.join(args.data_dir, 'dev.json'))       # 150 组数据
        # 长句子截断
        examples = get_sentence(raw_examples, 'eval')

    for i, example in enumerate(examples):
        text = example.text      # 76 个字。 len: 75
        entities = example.labels
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


def convert_span_features(args, data_type, tokenizer, ent2id):
    examples = None
    features = []

    if data_type == 'train':
        raw_examples = read_data(os.path.join(args.data_dir, 'train.json'))  # 850 组数据
        # dict_keys(['id', 'text', 'labels', 'pseudo', 'candidate_entities'])
        # 长句子截断
        examples = get_sentence(raw_examples, 'train')

    elif data_type == 'eval':
        raw_examples = read_data(os.path.join(args.data_dir, 'dev.json'))  # 150 组数据
        # 长句子截断
        examples = get_sentence(raw_examples, 'eval')

    for i, example in enumerate(examples):
        text = example.text      # 76 个字。 len: 75
        entities = example.labels
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

        start_idx = [0] * len(tokens)
        end_idx = [0] * len(tokens)
        for entity in entities:
            ent_type = ent2id[entity[1]]
            ent_satrt = entity[2]
            ent_end = entity[3] - 1
            start_idx[ent_satrt] = ent_type     # 在命名实体的开始位置和结束位置都设置为实体类型的 idx
            end_idx[ent_end] = ent_type

        if len(start_idx) > args.max_seq_len - 2:
            start_idx = start_idx[:args.max_seq_len - 2]
            end_idx = end_idx[:args.max_seq_len - 2]

        start_idx = [0] + start_idx + [0]
        end_idx = [0] + end_idx + [0]

        if len(start_idx) < args.max_seq_len:
            pad_len = args.max_seq_len - len(start_idx)
            start_idx += [0] * pad_len
            end_idx += [0] * pad_len

        assert len(start_idx) == args.max_seq_len
        assert len(end_idx) == args.max_seq_len

        features.append(SpanFeature(token_idx=tokens_idx,
                                    attention_mask=attention_mask,
                                    token_type_idx=token_type_idx,
                                    text=tokens,
                                    start_idx=start_idx,
                                    end_idx=end_idx))

    return features


def convert_mrc_features(args, data_type, tokenizer, ent2query):
    ENTITY_TYPES = ['DRUG', 'DRUG_INGREDIENT', 'DISEASE', 'SYMPTOM', 'SYNDROME', 'DISEASE_GROUP',
                    'FOOD', 'FOOD_GROUP', 'PERSON_GROUP', 'DRUG_GROUP', 'DRUG_DOSAGE', 'DRUG_TASTE',
                    'DRUG_EFFICACY']
    type2id = {x: i for i, x in enumerate(ENTITY_TYPES)}

    examples = None
    features = []
    if data_type == 'train':
        raw_examples = read_data(os.path.join(args.data_dir, 'train.json'))  # 850 组数据
        # dict_keys(['id', 'text', 'labels', 'pseudo', 'candidate_entities'])
        # 长句子截断
        examples = get_sentence(raw_examples, 'train', args.max_seq_len-args.mrc_query_len)
    elif data_type == 'eval':
        raw_examples = read_data(os.path.join(args.data_dir, 'dev.json'))  # 150 组数据
        # 长句子截断
        examples = get_sentence(raw_examples, 'eval', args.max_seq_len-args.mrc_query_len)

    print('Creat features...')
    for i, example in enumerate(tqdm(examples)):
        feature = []
        text_b = example.text
        entities = example.labels
        tokens_b = fine_grade_tokenize(text_b, tokenizer)  # 分词，将字转换成 token，将空格和未登录的词转换成指定的 tonken。 len: 76

        label_dict = defaultdict(list)      # 会构建一个默认value为list的字典
        for entity in entities:
            ent_type = entity[1]
            ent_start = entity[2]
            ent_end = entity[3] - 1
            label_dict[ent_type].append((ent_start, ent_end, entity[0]))

        for entity in ENTITY_TYPES:
            start_idx = [0] * len(tokens_b)
            end_idx = [0] * len(tokens_b)

            text_a = ent2query[entity]
            tokens_a = fine_grade_tokenize(text_a, tokenizer)

            stop_mask_ranges = []       # 用于随机 mask
            for labels in label_dict[entity]:
                start_idx[labels[0]] = 1
                end_idx[labels[1]] = 1
                stop_mask_ranges.append((labels[0], labels[1]))

            if len(start_idx) > args.max_seq_len - len(tokens_a) - 3:
                start_idx = start_idx[:args.max_seq_len - len(tokens_a) - 3]
                end_idx = end_idx[:args.max_seq_len - len(tokens_a) - 3]
                print('产生了不该有的截断')

            start_idx = [0] + [0] * len(tokens_a) + [0] + start_idx + [0]
            end_idx = [0] + [0] * len(tokens_a) + [0] + end_idx + [0]

            # pad
            if len(start_idx) < args.max_seq_len:
                pad_length = args.max_seq_len - len(start_idx)

                start_idx = start_idx + [0] * pad_length  # CLS SEP PAD label都为O
                end_idx = end_idx + [0] * pad_length

            assert len(start_idx) == args.max_seq_len
            assert len(end_idx) == args.max_seq_len

            # 随机mask ##################

            encode_dict = tokenizer.encode_plus(text=tokens_a,
                                                text_pair=tokens_b,
                                                max_length=args.max_seq_len,
                                                truncation_strategy='only_second',
                                                pad_to_max_length=True,
                                                return_token_type_ids=True,
                                                return_attention_mask=True,
                                                is_pretokenize=True)
            tokens_idx = encode_dict['input_ids']  # 将 token 转换成 idx，并在首尾添加 cls 和 sep 标签，并补零。 有效长度 len: 78
            token_type_idx = encode_dict['token_type_ids']  # 只有一句话，全为 0
            attention_mask = encode_dict['attention_mask']  # 将 cls 和 sep 标签一并进行 mask。有效长度 len: 78

            feature.append(MRCFeature(token_idx=tokens_idx,
                                        attention_mask=attention_mask,
                                        token_type_idx=token_type_idx,
                                        text=tokens_b,
                                        start_idx=start_idx,
                                        end_idx=end_idx,
                                        entity_type=type2id[entity]))

        features.extend(feature)

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


