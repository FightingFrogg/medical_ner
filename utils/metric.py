# -*- coding: utf-8 -*-
# author:SHAN
# datetime:2021/10/14 20:41

from processer import get_entities
from collections import Counter


class Metric(object):
    def __init__(self, id2ent):
        self.id2ent = id2ent
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []


    def update(self, labels, tags, flag='crf'):
        '''
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]

        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        for label, tag in zip(labels, tags):
            if flag == 'crf':
                label_entities = get_entities(label, self.id2ent)
                tag_entities = get_entities(tag, self.id2ent)
            else:
                label_entities = labels
                tag_entities = tags
            self.origins.extend(label_entities)
            self.founds.extend(tag_entities)
            self.rights.extend([tag_entity for tag_entity in tag_entities if tag_entity in label_entities])

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():     # 对每一种不同的标签分别做运算
            origin = count
            found = found_counter.get(type_, 0)     # 返回标签为 type_ 的个数
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"precision": round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}    # 四舍五入

        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)

        # print('Precision: {}\nRecall: {}\nF1: {}\n'.format(precision, recall, f1))
        # 第一个返回值为所有标签的 metric，第二个返回值记录了不同标签分别的 metric
        return {'precision': precision, 'recall': recall, 'f1': f1}, class_info
