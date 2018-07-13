# -*- coding: utf-8 -*-
"""天池，糖尿病预测初赛

根据20180128日的对比实验选出最优的方案(分类器参数None, 分类器数量1, 阈值None, 增加值None，采用未归一化的y值)，对testB预测，并将结果保存，提交

@author: He Kai
@contact: matthewhe@foxmail.com
@file: main20180129.py
@time: 2018/1/29 9:28
"""
import numpy as np

from kmodel_add import KSingleModel, KModelAdd
from get_best_gbm import get_best_baseparam

data_dir = r"data/preprocesskma/"
data_id = r"0000"
submission = r"data/kma.csv"

# 0. 加载数据
train_data = np.loadtxt(data_dir + "train_X_" + data_id + ".csv", delimiter=",", dtype=np.float64, ndmin=2)
train_labels = np.loadtxt(data_dir + "train_Y_" + data_id + ".csv", delimiter=",", dtype=np.float64, ndmin=2)
train_label = train_labels[:, 1].ravel()
test_data = np.loadtxt(data_dir + "test_X_" + data_id + ".csv", delimiter=",", dtype=np.float64, ndmin=2)
assert test_data.shape == (1000, 39)
# 1. 训练模型，预测数据
kma = KModelAdd(KSingleModel(get_best_baseparam()), None, 1, None, None)
kma.fit(train_data, train_label)
test_label = kma.predict(test_data)
# 2. 数据展示与保存
np.savetxt(submission, np.array(test_label, ndmin=2).T)
up_mask = test_label > 9
up_id = np.arange(0, up_mask.shape[0])[up_mask]
print("预测的大值id：%s" % str(up_id))