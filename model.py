# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 01:56:48 2019

@author: msq96
"""


import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertForPreTraining


class Model(BertPreTrainedModel):

    def __init__(self, config, num_labels, model_param_fp=None):
        super(Model, self).__init__(config)
        self.num_labels = num_labels

        if model_param_fp is not None:
            self.bert = BertForPreTraining.from_pretrained(config).load_state_dict(torch.load(open(model_param_fp, 'rb'))).bert
        else:
            self.bert = BertModel(config)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits