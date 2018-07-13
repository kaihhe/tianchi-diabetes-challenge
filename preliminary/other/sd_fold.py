# -*- coding: utf-8 -*-
"""独立同分布采样的K折交叉验证

可以获得训练集和验证集的下标

训练集K-1折，剩下的1为验证集

@author: He Kai
@contact: matthewhe@foxmail.com
@file: sd_fold.py
@time: 2018/1/19 10:56
"""
import numpy as np
import matplotlib.pyplot as plt

class SDFold:
    def __init__(self, n_splits):
        self.n_splits = n_splits
    def split(self, X, y):
        """生成数据和样本的切片

        :param X: array-like, 特征
        :param y: array-like, 标记
        :return: train, test
        """
        n_samples = X.shape[0]
        neg_y = np.copy(y) * -1
        # 给label加上索引，并且按照label的大小排序
        index = np.arange(0, y.shape[0], 1, dtype=np.int32)
        y_with_index = [(neg_y[i], index[i]) for i in range(index.shape[0])]
        type = [("y", float), ("index", int)]
        y_with_index = np.array(y_with_index, dtype=type)
        y_with_index.sort(order="y")
        ordered_index = [x[1] for x in y_with_index]
        ordered_index = np.array(ordered_index, dtype=np.int32)
        index_indice = np.zeros(ordered_index.shape[0])
        k = self.n_splits
        for i in range(k):
            index_indice[i:n_samples:k] = i
        for i in range(k):
            mask = index_indice == i
            yield ordered_index[~mask], ordered_index[mask]

    def get_n_splits(self):
        return self.n_splits

if __name__ == "__main__":
    data_dir = "diabetes1_S5A3C4/"
    data_id = "201801101443"
    train_data = np.loadtxt(data_dir + "train_X_" + data_id + ".csv", delimiter=",", dtype=np.float64, ndmin=2)
    train_labels = np.loadtxt(data_dir + "train_Y_" + data_id + ".csv", delimiter=",", dtype=np.float64, ndmin=2)
    train_label = train_labels[:, 0].ravel()

    sdfold = SDFold(5)
    for train_index,test_index in  sdfold.split(train_data, train_label):
        x = train_label[train_index]
        y = train_label[test_index]
        plt.plot(x)
        plt.plot(np.arange(x.shape[0], 5641), y)
        plt.show()