# -*- coding: utf-8 -*-
# author:SHAN
# datetime:2021/9/29 20:37

from transformers import BertPreTrainedModel, BertModel
from torch import nn
from torchcrf import CRF


class CRFModel(BertPreTrainedModel):
    def __init__(self, config):
        super(CRFModel, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, token_idx, token_type_idx, attention_mask, labels=None):
        bert_outputs = self.bert(token_idx, attention_mask=attention_mask, token_type_ids=token_type_idx)
        bert_outputs = bert_outputs[0]
        bert_outputs = self.dropout(bert_outputs)
        emission = self.classifier(bert_outputs)
        if labels is not None:
            loss = -1. * self.crf(emissions=emission, mask=attention_mask.byte(), tags=labels.long(), reduction='mean')
            outputs = (loss,)
        else:
            out = self.crf.decode(emissions=emission, mask=attention_mask.byte())   # 返回每一个 batch 的预测序列
            outputs = (out, emission)

        return outputs


class SpanModel(BertPreTrainedModel):
    def __init__(self, config, loss_type='ce', mid_linear_dim=128, dropout_prob=0.1):
        super(SpanModel, self).__init__(config)
        """
        tag the subject and object corresponding to the predicate
        :param loss_type: train loss type in ['ce', 'ls_ce', 'focal']
        
        TODO: 损失函数，参数初始化
        """
        self.num_labels = config.num_labels + 1

        self.bert = BertModel(config)
        self.mid_linear = nn.Sequential(
            nn.Linear(config.hidden_size, mid_linear_dim),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob))

        self.start_linear = nn.Linear(mid_linear_dim, self.num_labels)      # 有 13 个实体，但是有 14 类，第 0 类表示无实体名称
        self.end_linear = nn.Linear(mid_linear_dim, self.num_labels)

        reduction = 'mean'
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        # elif loss_type == 'ls_ce':
        #     self.criterion = LabelSmoothingCrossEntropy(reduction=reduction)
        # else:
        #     self.criterion = FocalLoss(reduction=reduction)

    def forward(self, token_idx, token_type_idx, attention_mask, start_idx=None, end_idx=None, train_label=True):
        bert_outputs = self.bert(token_idx, attention_mask=attention_mask, token_type_ids=token_type_idx)
        bert_outputs = bert_outputs[0]
        seq_out = self.mid_linear(bert_outputs)

        start_logits = self.start_linear(seq_out)
        end_logits = self.end_linear(seq_out)
        out = (start_logits, end_logits)

        if start_idx is not None and end_idx is not None and self.training:
            start_logits = start_logits.view(-1, self.num_labels)
            end_logits = end_logits.view(-1, self.num_labels)
            mask = attention_mask.view(-1) == 1

            start_logits_masked = start_logits[mask]
            end_logits_masked = end_logits[mask]

            start_labels_masked = start_idx.view(-1)[mask]
            end_labels_masked = end_idx.view(-1)[mask]

            start_loss = self.criterion(start_logits_masked, start_labels_masked)
            end_loss = self.criterion(end_logits_masked, end_labels_masked)
            loss = (start_loss + end_loss)

            out = (loss,) + out

        return out




