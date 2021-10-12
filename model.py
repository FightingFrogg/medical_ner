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
            loss = self.crf(emissions=emission, mask=attention_mask.byte(), tags=labels.long(), reduction='mean')
            outputs = (loss,)
        else:
            out = self.crf.decode(emissions=emission, mask=attention_mask.byte())
            outputs = (out, emission)

        return outputs
