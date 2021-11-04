# -*- coding: utf-8 -*-
# author:SHAN
# datetime:2021/9/29 20:37
import torch
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
    def __init__(self, config, loss_type='fl', mid_linear_dim=128, dropout_prob=0.1):
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
        elif loss_type == 'ls_ce':
            self.criterion = LabelSmoothingCrossEntropy(reduction=reduction)
        elif loss_type == 'fl':
            self.criterion = FocalLoss(reduction=reduction)

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


class MRCModel(BertPreTrainedModel):
    def __init__(self, config, loss_type='ce', mid_linear_dim=128):
        super(MRCModel, self).__init__(config)

        self.bert = BertModel(config)
        self.mid_linear = nn.Sequential(
            nn.Linear(config.hidden_size, mid_linear_dim),
            nn.ReLU(),
            nn.Dropout(config.hidden_dropout_prob))

        self.start_linear = nn.Linear(mid_linear_dim, 2)
        self.end_linear = nn.Linear(mid_linear_dim, 2)

        reduction = 'mean'
        if loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss(reduction=reduction)
        # elif loss_type == 'ls_ce':
        #     self.criterion = LabelSmoothingCrossEntropy(reduction=reduction)
        # else:
        #     self.criterion = FocalLoss(reduction=reduction)

    def forward(self, token_idx, token_type_idx, attention_mask, entity_type=None, start_idx=None, end_idx=None):
        bert_outputs = self.bert(token_idx, attention_mask=attention_mask, token_type_ids=token_type_idx)
        bert_outputs = bert_outputs[0]
        seq_out = self.mid_linear(bert_outputs)

        start_logits = self.start_linear(seq_out)
        end_logits = self.end_linear(seq_out)
        out = (start_logits, end_logits)

        if start_idx is not None and end_idx is not None and self.training:
            start_logits = start_logits.view(-1, 2)
            end_logits = end_logits.view(-1, 2)
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


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=-100):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.criterion = nn.KLDivLoss(size_average=False, reduction=reduction)


    def forward(self, out, target):
        # out(8000, 14)
        # target(8000,)
        labels_num = out.size(1)
        smoothing_value = self.eps / (labels_num - 1)
        one_hot = torch.full((labels_num, ), smoothing_value)
        one_hot = one_hot.repeat(out.size(0), 1)            # (8000, 14)
        temp_ = target.view(-1, 1)
        one_hot.scatter_(1, temp_, (1 - self.eps))

        out = self.log_softmax(out)
        loss = self.criterion(out, one_hot)
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, a=0.25, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.a = a
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.softmax = nn.Softmax(dim=1)
        self.nllloss = nn.NLLLoss(reduction=self.reduction, ignore_index=ignore_index)

    def forward(self, out, target):
        pt = self.softmax(out)
        log_pt = self.a * (1 - pt) ** self.gamma * torch.log(pt)
        loss = self.nllloss(log_pt, target)
        return loss


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, eps=1., emb_name='embeddings.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)       # norm: 默认 p=2 求二范数
                if norm != 0 and not torch.isnan(norm):
                    radv = eps * param.grad / norm
                    param.data.add_(radv)

    def restore(self, emb_name='embeddings.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD():
    def __init__(self, model):
        self.model = model
        self.grad_backup = {}
        self.emb_backup = {}

    def attack(self, eps=1., alpha=0.3, emb_name='embeddings.', is_first_attack=False):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)  # norm: 默认 p=2 求二范数
                if norm != 0 and not torch.isnan(norm):
                    r = alpha * param.grad / norm
                    param.data.add_(r)
                    # 如果扰动幅度过大，需要把扰动拉回到球面以内，球的半径位 eps
                    param.data = self.projected(name, param.data, eps)

    def restore(self, emb_name='embeddings.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def projected(self, param_name, param_data, eps):
        dr = param_data - self.emb_backup[param_name]
        if torch.norm(dr) > eps:
            dr = eps * dr / torch.norm(dr)
        return dr + self.emb_backup[param_name]

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
