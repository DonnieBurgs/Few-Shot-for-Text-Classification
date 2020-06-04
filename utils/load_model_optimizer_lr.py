# -*- coding: utf-8 -*-
"""
Description :   加载模型并自定义各层参数的冻结与否、学习率大小以及衰减与否
     Author :   Yang
       Date :   2020/3/28
"""
import re
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertTokenizer, AdamW
from transformers import get_linear_schedule_with_warmup

from models.bert_baseline import Model
# from models.bfsc import Model


def find_param(name, name_list):
    # name_list 需要找到的参数名列表，正则表达式搜索
    for pattern in name_list:
        if re.search(pattern, name) is not None:
            return True
    return False


def add_weight_decay(model, bert_lr=5e-6, weight_decay=2e-5, skip_list=[]):
    # weight_decay我认为是学习率衰减度，若为0则不衰减
    decay = []  # 新加入的层
    no_decay = []  # layernorm和bias
    bert_base_para = []  # bert 模型中的固有参数，但是需要除开layernorm和bias
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if find_param(name, ['model']):
            if len(param.shape) == 1 or find_param(name, skip_list):
                no_decay.append(param)
            else:
                bert_base_para.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'lr': bert_lr, 'weight_decay': 0.0},
        {'params': decay, 'weight_decay': weight_decay},
        {'params': bert_base_para, 'lr': bert_lr, 'weight_decay': weight_decay},
    ]


class LabelSmoothingLoss(nn.Module):
    """通过标签平滑正则技术来计算损失函数"""
    
    def __init__(self, classes, smoothing=0.0, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim
    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


def load_model_op_lr(config, train_steps):
    finetune_dir = config['finetune_dir']
    continue_train = config['continue_train']
    epochs = config['epoch']
    diff_decay = config['diff_decay']
    lr = config['lr']
    lsr_loss = config['lsr_loss']
    bert_lr = config['bert_lr']
    
    # 加载预训练权重
    sentiment_model = Model(config)  # RoBERTa
    
    # 加载微调权重, 一般用来紧接上次训练
    if continue_train and os.path.exists(os.path.join(finetune_dir, 'pytorch_model.bin')):
        sentiment_model.load_state_dict(torch.load(os.path.join(finetune_dir, 'pytorch_model.bin')))
    
    if torch.cuda.is_available():
        sentiment_model = sentiment_model.cuda()  # 放在optimizer定义之前
    
    if lsr_loss:
        criterion = LabelSmoothingLoss(3, 0.1)
    else:
        criterion = nn.CrossEntropyLoss()
    
    if diff_decay:
        parameters = add_weight_decay(sentiment_model, bert_lr, skip_list=["LayerNorm", "bias"])
        optimizer = AdamW(parameters,
                          lr=lr,
                          eps=1e-8
                          )
    else:
        optimizer = AdamW(sentiment_model.parameters(),
                          lr=lr,
                          eps=1e-8
                          )
    
    total_steps = train_steps * epochs
    # 学习率先warmup策略(如果当前步小于num_warmup_steps)，随后decay策略
    # 由于我们的总step(数据)比较少，所以直接decay
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py is 0
                                                num_training_steps=total_steps)
    return sentiment_model, optimizer, criterion, scheduler


if __name__ == "__main__":
    config = {
        'init_checkpoint': "/home/ai/yangwei/wuhan_competition/sentiment_analysis/v1/checkpoints/roBERTa/",
        "freeze_bert": False,
        'dropout': 0.2,
        'use_bigru': False,
        'output_hidden_states': True,
        'concat_output': False,
        'finetune_dir': "/home/ai/yangwei/wuhan_competition/sentiment_analysis/v1/checkpoints/output",
    }
    train_steps = 100
    
    load_model_op_lr(config, train_steps)