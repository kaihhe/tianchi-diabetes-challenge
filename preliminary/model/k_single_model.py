# -*- coding: utf-8 -*-
""" 天池，糖尿病精准预测，初赛

baseline模型

@author: He Kai
@contact: matthewhe@foxmail.com
@file: k_single_model.py
@time: 2018/1/17 15:05
"""
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sd_fold import SDFold

class KSingleModel:
    """K折交叉单模型

    使用KFold将原始集合分成K个训练集和测试集（将测试集用作验证集，防止过拟合）

    单模型使用的lightGBM的LGBMRegressor

    """
    def __init__(self, param, k=5, fold="base", type="regressor"):
        """初始化函数

        创建k个回归器

        :param param: 回归器的参数
        :param k: 交叉验证倍数
        :param fold: string, CV的类型{"base", "sd"}, base是基本的KFold, sd是同样分布
        :param type: string, 单个模型的类型{"regressor", "classifier"}
        """
        # 基学习器参数
        self.param = param
        # 交叉验证器
        self.k = k
        if fold == "base":
            self.fold = KFold(k, shuffle=True, random_state=520)
        elif fold == "sd":
            self.fold = SDFold(k)
        else:
            raise ValueError(r"fold should be{'base', 'sd'}.")
        # 学习器类型
        if type == "regressor":
            self.regressors = [lgb.LGBMRegressor(**param) for i in range(k)]
        elif type == "classifier":
            self.regressors = [lgb.LGBMClassifier(**param) for i in range(k)]
        else:
            raise ValueError(r"type 应该是 {'regressor', 'classifier'}")

    def fit(self, X, y):
        self.va_pred = np.zeros(y.shape[0])
        kf = self.fold
        for i, (tr_index, va_index) in enumerate(kf.split(X, y)):
            X_tr = X[tr_index]
            y_tr = y[tr_index]
            X_va = X[va_index]
            y_va = y[va_index]
            self.regressors[i].fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100)
            y_va_pred = self.regressors[i].predict(X_va)
            self.va_pred[va_index] = y_va_pred

    def predict(self, X):
        k = self.k
        result = np.zeros((k, X.shape[0]))
        for i in range(k):
            result[i, :] = self.regressors[i].predict(X)
        pred = np.mean(result, axis=0)
        return pred

    def score(self, X, y):
        y_pred = self.predict(X)
        return mean_squared_error(y, y_pred) * 0.5