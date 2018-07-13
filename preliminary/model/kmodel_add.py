# -*- coding: utf-8 -*-
"""将KSingleModel作为baseline，

异常点修改

@author: He Kai
@contact: matthewhe@foxmail.com
@file: kmodel_add.py
@time: 2018/1/22 21:21
"""

import numpy as np
import lightgbm as lgb
from k_single_model import KSingleModel
import matplotlib.pyplot as plt


class KModelAdd:
    """在KSingleModel上，修改高值部分

    内部使用了一个KSingleModel，以它预测的结果

    作为baseline.

    """
    def __init__(self, kmodel, classifier_param=None, n_classifier=1, y_up_border = None, add_value = None):
        """初始化函数

        :param kmodel: KSingleModel
        :param classifier_param: LGBM的分类器的参数
        :param n_classifier: int, 分类器的个数，采用与KSingleModel一样的结构
        :param y_up_border: 将y值中大于y_up_border的点考虑为大值点, 默认为大于均值的y值的均值
        """
        self.kmodel = kmodel
        self.y_up_border = y_up_border
        self.add_value = add_value
        if classifier_param == None:
            classifier_param = {}
        if n_classifier == 1:
            self.classifier = lgb.LGBMClassifier(**classifier_param)
        elif n_classifier > 1:
            self.classifier = KSingleModel(classifier_param, n_classifier, "sd", "classifier")
        else:
            raise ValueError(r"n_classifier应该是大于等于1的整数！")
    def fit(self, X, y):
        """训练模型

        :param X: array, 样本
        :param y: array of 1-D, 标记
        :return: self
        """
        # 训练KSingleModel
        self.train_data = X
        self.train_label = y
        self.kmodel.fit(X, y)

        # 训练分类器
        # 准备分类标签，所有的预测的MSE超过平均MSE的，并且值大于均值的
        y_pred = self.kmodel.predict(X)
        all_mse = (y_pred - y) ** 2 * 0.5
        mean_mse = all_mse.mean()
        wrong_mask = all_mse > mean_mse # 预测错误的点

        y_up_border = self.y_up_border
        if y_up_border == None:
            y_up_border = y[y>y.mean()].mean()
        up_mask = y > y_up_border
        add_target_mask = up_mask & wrong_mask
        # 分类标签
        cls_label = np.zeros(y.shape[0])
        cls_label[add_target_mask] = 1

        self.classifier.fit(X, cls_label)
        if self.add_value == None:
            self.add_value = y[add_target_mask].mean() - y_pred[add_target_mask].mean()
        return self
    def predict(self, X):
        """预测函数

        :param X: 测试集的样本
        :return: array of 1D, 预测的结果
        """
        test_pred = self.kmodel.predict(X)
        test_cls = self.classifier.predict(X)
        add_target = test_cls > 0

        test_pred[add_target] += self.add_value
        return test_pred