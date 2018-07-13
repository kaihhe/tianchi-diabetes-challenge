# -*- coding: utf-8 -*-
"""阿里天池，糖尿病预测，初赛

参数调优

@author: He Kai
@contact: matthewhe@foxmail.com
@file: param_tune.py
@time: 2018/1/26 19:26
"""


def get_best_baseparam():
    """返回调参得到的参数

    :return: dictionary
    """
    best_param = {
        "learning_rate": 0.001,
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "mse",
        "num_leaves": 20,
        "feature_fraction": 0.9,
        "min_data_in_leaf": 40,
        "n_estimators": 60000,
        "max_depth": 9,
        "verbose": 0
    }
    return best_param

# 默认baseline的基学习器参数
base_baseparam = {
    'learning_rate': 0.01,
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'mse',
    'num_leaves': 60,
    'feature_fraction': 0.7,
    'min_data': 100,
    'min_hessian': 1,
    'verbose': -1,
    "n_estimators": 6000
}