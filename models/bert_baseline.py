# -*- coding: utf-8 -*-
"""
Description :   
     Author :   Yang
       Date :   2020/3/22
"""
import re
import os
import csv
import numpy as np
import pandas as pd
import scipy as sp

from functools import partial
from sklearn.metrics import f1_score, accuracy_score

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel, BertConfig, AdamW


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class OptimizedF1(object):
    def __init__(self):
        self.coef_ = []  # 类似于二分类中阈值搜索的参数

    def _kappa_loss(self, coef, X, y):
        """
        y_hat = argmax(coef*X, axis=-1)
        :param coef: (1D array) weights
        :param X: (2D array)logits
        :param y: (1D array) label
        :return: -f1
        """
        X_p = np.copy(X)
        X_p = coef * X_p
        f1 = f1_score(y, np.argmax(X_p, axis=-1), average='macro')
        return -f1

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [1. for _ in range(3)]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        # self.coef_是该batch数据上最佳的参数

    def predict(self, X, y):
        X_p = np.copy(X)
        X_p = self.coef_['x'] * X_p
        return f1_score(y, np.argmax(X_p, axis=-1), average='macro')

    def coefficients(self):
        return self.coef_['x']


# op = OptimizedF1()


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        init_checkpoint = config['init_checkpoint']
        freeze_bert = config['freeze_bert']
        dropout = config['dropout']
        self.use_bigru = config['use_bigru']
        self.output_hidden_states = config['output_hidden_states']
        self.concat_output = config['concat_output']

        self.config = config

        bert_config = BertConfig.from_pretrained(os.path.join(init_checkpoint, 'bert_config.json'),
                                                 output_hidden_states=self.output_hidden_states)
        self.model = BertModel.from_pretrained(os.path.join(init_checkpoint, 'pytorch_model.bin'),
                                               config=bert_config)
        self.dropout = nn.Dropout(dropout)

        if freeze_bert:
            for p in self.model.parameters():
                p.requires_grad = False  # 亦可以针对性的微调或者冻结某层参数

        if self.use_bigru:
            self.biGRU = torch.nn.GRU(768, 768, num_layers=1, batch_first=True, bidirectional=True)
            self.dense = nn.Linear(bert_config.hidden_size * 2, 3)  # 连接bigru的输出层
        elif self.concat_output:
            self.dense = nn.Linear(bert_config.hidden_size * 3, 3)  # 连接concat后的三个向量
        else:
            self.dense = nn.Linear(bert_config.hidden_size, 3)  # 输出3维（3分类）

    def forward(self, inputs):
        # 1.最后几层向量的合并
        if self.output_hidden_states:
            sequence_output, pooler_state, encoder_outputs = self.model(inputs)
            # sequence_output 最后一层所有向量
            # encoder_outputs 所有层的所有向量
            # encoder_outputs[-1][:, 0]  # 倒数第一层的CLS向量
            # encoder_outputs[-2][:, 0]  # 倒数第二层的CLS向量
            # encoder_outputs[-3][:, 0]
            if self.concat_output:
                output = torch.cat([encoder_outputs[-1][:, 0], encoder_outputs[-2][:, 0], encoder_outputs[-3][:, 0]], dim=-1)
            else:
                output = encoder_outputs[-1][:, 0] + encoder_outputs[-2][:, 0] + encoder_outputs[-3][:, 0]
            output = self.dense(output)
            return output
        else:
            sequence_output, pooler_state = self.model(inputs)

        # 2.bert后连接GRU或者直接输出
        if self.use_bigru:
            hidden = self._init_hidden(inputs.size(0))
            _, hidden = self.biGRU(sequence_output, hidden)
            out = torch.cat((hidden[0], hidden[1]), dim=-1)
        else:
            out = self.dropout(pooler_state)
        output = self.dense(out)

        return output

    def _init_hidden(self, batch_size):
        hidden = torch.zeros(2, batch_size, 768).cuda()
        return torch.autograd.Variable(hidden)


if __name__ == "__main__":
    config = {
        "init_checkpoint": "/home/ai/yangwei/wuhan_competition/sentiment_analysis/v1/checkpoints/roBERTa/",
        "freeze_bert": False,
        "dropout": 0.1,
        "use_bigru": False,
        "output_hidden_states": True,
        "concat_output": True,

    }

    model = Model(config)
    inp = np.random.randint(low=101, high=200, size=(32, 128))
    inp = torch.tensor(inp)
    pred = model(inp)
    print(pred.shape)
