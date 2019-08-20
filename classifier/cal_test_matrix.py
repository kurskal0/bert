# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     cal_test_matrix
   Author :        Xiaosong Zhou
   date：          2019/8/19
-------------------------------------------------
"""
__author__ = 'Xiaosong Zhou'

import pandas as pd
import numpy as np
from sklearn import metrics

# 计算测试结果
# 因为原生的predict生成一个test_results.tsv文件，给出了每一个sample的每一个维度的值
# 却并没有给出具体的类别预测以及指标，这里再对这个“中间结果手动转化一下”


def cal_accuracy(rst_file_dir, y_test_dir):
    rst_contents = pd.read_csv(rst_file_dir, sep='\t', header=None)
    # value_list: ndarray
    value_list = rst_contents.values
    pred = value_list.argmax(axis=1)
    labels = []

    # 这一步是获取y标签到id，id到标签的对应dict，每个人获取的方式应该不一致
    y2id, id2y = get_y_to_id(vocab_y_dir='../data/statutes_small/vocab_y.txt')
    with open(y_test_dir, 'r', encoding='utf-8') as f:
        line = f.readline()
        while line:
            # 这里因为y有多个标签，我要取第一个标签，所以要单独做操作
            label = line.strip().split()[0]
            labels.append(y2id[label])
            line = f.readline()
    labels = np.asarray(labels)

    # 预测，pred，真实标签，labels
    accuracy = metrics.accuracy_score(y_true=labels, y_pred=pred)
    print(accuracy)


def get_y_to_id(vocab_y_dir):
    # 这里把所有的y标签值存在了文件中
    y_vocab = open(vocab_y_dir, 'r', encoding='utf-8').read().splitlines()
    y2idx = {token: idx for idx, token in enumerate(y_vocab)}
    idx2y = {idx: token for idx, token in enumerate(y_vocab)}
    return y2idx, idx2y


if __name__ == '__main__':
    count = len(open(r"../data/statutes_small/test_x.txt", 'r', encoding='utf-8').readlines())
    print(count)
    cal_accuracy(rst_file_dir='../out/test_results.tsv', y_test_dir='../data/statutes_small/test_y.txt')
