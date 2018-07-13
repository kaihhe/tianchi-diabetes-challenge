# -*- coding: utf-8 -*-
"""独立同分布采样的K折交叉验证

可以获得训练集和验证集的下标

训练集K-1折，剩下的1为验证集

@author: He Kai
@contact: matthewhe@foxmail.com
@file: sd_fold.py
@time: 2018/1/19 10:56
"""
import pandas as pd
import numpy as np

class SDFold:
    def __init__(self, n_splits):
        """

        :param n_splits: 交叉验证折数
        """
        self.n_splits = n_splits
    def split(self, X, y):
        """生成数据和样本的切片

        :param X: array-like, DataFrame, 特征
        :param y: array-like, DataFrame, 标记
        :return: train, test
        """
        col = ['label', 'SNP1', 'SNP2', 'SNP3', 'SNP4', 'SNP5', 'SNP6', 'SNP7', 'SNP8',
               'SNP9', 'SNP10', 'SNP11', 'SNP12', 'SNP13', 'SNP14', 'SNP15', 'SNP16',
               'SNP17', 'SNP18', 'SNP19', 'SNP20', 'SNP21', 'SNP22', 'SNP23', 'SNP24', 'SNP25',
               'SNP26', 'SNP27', 'SNP28', 'SNP29', 'SNP30', 'SNP31', 'SNP32', 'SNP33',
               'SNP34', 'SNP35', 'SNP36', 'SNP37', 'SNP38', 'SNP39', 'SNP40',
               'SNP41', 'SNP42', 'SNP43', 'SNP44', 'SNP45', 'SNP46', 'SNP47', 'SNP48',
               'SNP49', 'SNP50', 'SNP51', 'SNP52', 'SNP53', 'SNP54', 'SNP55',
               'RBP4',
               '年龄', '孕次', '产次', '身高', '孕前体重', 'BMI分类', '孕前BMI', '收缩压', '舒张压', '分娩时',
               '糖筛孕周', 'VAR00007', 'wbc', 'ALT', 'AST', 'Cr', 'BUN', 'CHO', 'TG',
               'HDLC', 'LDLC', 'ApoA1', 'ApoB', 'Lpa', 'hsCRP', 'DM家族史', 'ACEID',]
        Xy = pd.concat([X, y], axis=1)
        removes = set(col) - set(Xy.columns)
        for remove in removes:
            col.remove(remove)
        ordered_index = Xy.sort_values(col).index.values
        index_indice = np.zeros(ordered_index.shape[0])
        n_samples = X.shape[0]
        k = self.n_splits
        for i in range(k):
            index_indice[i:n_samples:k] = i
        for i in range(k):
            mask = index_indice == i
            yield ordered_index[~mask], ordered_index[mask]
    def get_n_splits(self):
        return self.n_splits