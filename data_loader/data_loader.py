# -*- coding: utf-8 -*-
"""
Description :   
     Author :   Yang
       Date :   2020/3/23
"""
import os
import re
import glob
import codecs
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import *


BERT_PATH = "/home/ai/yangwei/myResources/chinese_roberta_wwm_base_ext_pytorch"
DATA_PATH = "/home/ai/yangwei/myResources/THUCNews"

class my_okenizer(BertTokenizer):
    def tokenize(self, text):
        chars = []
        for c in text:
            # 按字符遍历
            if c in self.vocab:
                chars.append(c)
            elif self._is_whitespace(c):
                chars.append('[unused1]')
            else:
                chars.append('[UNK]')
        return chars

    def _is_whitespace(self, char):
        if char in [" ", "\t", "\n", "\r", "\u3000"]:
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False


# 初始化分词器
tokenizer = my_okenizer(vocab_file=BERT_PATH+"/vocab.txt")


"""
1. 得到所有label及其文本描述
2. 构造Dataset，每个文本有50%概率得到真实标签，50%概率得到其他标签
"""

label_desc = {
    "彩票": "彩票是一种以筹集资金为目的发行的，印有号码、图形、文字、面值的，由购买人自愿按一定规则购买并确定是否获取奖励的凭证。",
    "社会": "社会，即是由生物与环境形成的关系总和。人类的生产、消费娱乐、政治、教育等，都属于社会活动范畴。",
    "游戏": "游戏是一种基于物质需求满足之上的，在一些特定时间、空间范围内遵循某种特定规则的，追求精神世界需求满足的社会行为方式",
    "时政": "时事政治就是指某个时间段发生的国内国际政治大事，是概括性比较强的、从大局出发的事件。主要表现为政党、社会集团、社会势力在处理国家生活和国际关系方面的方针、政策和活动。",
    "星座": "自从古代以来，人类便把三五成群的恒星与他们神话中的人物或器具联系起来，称之为“星座”。",
    "体育": "体育可分为大众体育、专业体育、学校体育等种类。包括体育文化、体育教育、体育活动、体育竞赛、体育设施、体育组织、体育科学技术等诸多要素。",
    "财经": "财经是指财政、金融、经济。",
    "科技": "科学主要是和未知的领域打交道，其进展，尤其是重大的突破，是难以预料的；技术是在相对成熟的领域内工作，可以做比较准确的规划。",
    "时尚": "时尚，就是人们对社会某项事物一时的崇尚，这里的“尚”是指一种高度。在如今社会里，多指流行得体的一些东西。",
    "股票": "股票是股份公司的所有权一部分也是发行的所有权凭证，是股份公司为筹集资金而发行给各个股东作为持股凭证并借以取得股息和红利的一种有价证券。",
    "家居": "家居指的是家庭装修、家具配置、电器摆放等一系列和居室有关的甚至包括地理位置（家居风水）都属于家居范畴。",
    "房产": "房产是指个人或团体保有所有权的房屋及地基。",
    "娱乐": "现代娱乐包含了悲喜剧、各种比赛和游戏、音乐舞蹈表演和欣赏等等。",
    "教育": "教育狭义上指专门组织的学校教育；广义上指影响人的身心发展的社会实践活动。"
}


class my_dataset(Dataset):
    def __init__(self, config, files):
        self.max_len = config['max_len']        
        self.data = files


    def __getitem__(self, item):
        fname = self.data[item]
        text = codecs.open(fname, encoding='utf-8').read()
        
        x = tokenizer.convert_tokens_to_ids(['[CLS]'] + list(text))
        token_ids = self.text_padding(x, self.max_len)
        tag = random.choice([1, -1])

        true_label = fname.split('/')[-2]
        if tag == 1:
            temp_label = label_desc[true_label]
        else:
            sample_label = random.choice(list(label_desc.keys()).remove(true_label))
            temp_label = label_desc[sample_label]
        
        label_ids = tokenizer.convert_tokens_to_ids(['[CLS]'] + list(temp_label))
        label_ids = self.text_padding(label_ids, max_len=100)
        mask_of_label = (torch.LongTensor(label_ids) > 0).float()

        return (torch.LongTensor(token_ids).cuda(),
                torch.LongTensor(tag).cuda(),
                torch.LongTensor(label_ids).cuda(),
                mask_of_label.cuda())


    def text_padding(self, x, max_len, padding=0):
        if len(x) >= max_len:
            # 512及以上
            tokens = x[:max_len-1] + [102]
        else:
            # 512以下
            tokens = x + [102] + (max_len-len(x)-1)*[padding]
        return tokens


    def __len__(self):
        return len(self.data)


config = {
    'max_len': 512,
    'batch_size': 24,
    }


# 文件路径集合
total_files = glob.glob(DATA_PATH+'/股票/*.txt')[:300]
train_valid_texts, test_texts = train_test_split(total_files, test_size=100, random_state=22)
train_texts, valid_texts = train_test_split(train_valid_texts, test_size=0.33, random_state=42)


train_set = my_dataset(config, files=train_texts)
valid_set = my_dataset(config, files=valid_texts)

train_loader = DataLoader(dataset=train_set, batch_size=config['batch_size'], shuffle=True)
valid_loader = DataLoader(dataset=valid_set, batch_size=config['batch_size'], shuffle=True)


if __name__ == "__main__":
    print(train_set.__getitem__(0))
