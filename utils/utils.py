# -*- coding: utf-8 -*-
"""
Description :   
     Author :   Yang
       Date :   2020/3/28
"""
import json
import os


def get_config_from_json(json_file):
    file = '/home/ai/yangwei/wuhan_competition/sentiment_analysis/v1/configs'
    with open(os.path.join(file, json_file), 'r') as config_file:
        config = json.load(config_file)
    return config
