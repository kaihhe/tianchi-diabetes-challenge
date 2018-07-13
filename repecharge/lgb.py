# -*- coding: utf-8 -*-
"""阿里天池，复赛

LightGBM二分类预测的妊娠糖尿病

SDFold:
    diabetes3
    1. 特征选择， ksm调参  0.73955
    2. 特征选择, lgb, ksm调参 0.75155***
    3. lgb, ksm调参 0.74764
    diabetes6 0.73837
最终testb提交：
    diabetes3, 特征选择, lgb调参, ksm调参
@author: He Kai
@contact: matthewhe@foxmail.com
@file: lgb.py
@time: 2018/2/19 20:51
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.feature_selection import SelectKBest, RFECV
from sklearn.metrics import f1_score
from sd_fold import SDFold

fold = SDFold(10)

def load_data(data_dir):
    """载入妊娠糖尿病数据(1000,84), (1000)

    :param data_dir: 数据目录的路径
    :return: X_train, y_train, X_test
    """
    X_train = pd.read_csv(data_dir + r"\X_train.csv", encoding="gbk")
    y_train = pd.read_csv(data_dir + r"\y_train.csv", encoding="gbk")
    X_test = pd.read_csv(data_dir + r"\X_test.csv", encoding="gbk")

    return X_train, y_train, X_test
def add_feat(X_train, y_train, X_test):
    # VAR*BMI
    X_train_VB = X_train['VAR00007'] * X_train['BMI分类']
    X_train_VB.name = 'VAR*BMI'
    X_train_add = pd.concat([X_train, X_train_VB], axis=1)
    X_test_VB = X_test['VAR00007'] * X_test['BMI分类']
    X_test_VB.name = 'VAR*BMI'
    X_test_add = pd.concat([X_test, X_test_VB], axis=1)
    return X_train_add, y_train, X_test_add
def feat_select(X_train, y_train, X_test, k=5):
    """lgb特征选择

    :param X_train: 训练集特征
    :param y_train: 训练集标记
    :param X_test: 测试集特征
    :param k: 特征选择根据的交叉验证折数
    :return: X_train_slt, y_train, X_test_slt
    """
    X_train = X_train.drop(['id'], axis=1)
    X_test = X_test.drop(['id'], axis=1)
    # 特征评分
    model = lgb.LGBMClassifier()
    rfe = RFECV(model, 1, cv=5, scoring="f1")
    rfe.fit(X_train.values, y_train.values.ravel())
    rank = rfe.ranking_ * -1
    order = rank.argsort()
    order_columns = X_train.columns[order]
    n_features = len(order_columns)
    bst_score, bst_i = 0, 0
    for i in range(n_features//4):
        X_train_slt = X_train.drop(order_columns[0:i], axis=1)
        scores = []
        for i in range(10):
            scores.append(cross_val_score(model, X_train_slt.values, y_train.values.ravel(), scoring="f1", cv=SDFold(k).split(X_train_slt, y_train))) # cv=k
        score = np.array(scores).mean()
        if score > bst_score:
            bst_score, bst_i = score, i
    X_train_slt = X_train.drop(order_columns[0:bst_i], axis=1)
    X_test_slt = X_test.drop(order_columns[0:bst_i], axis=1)
    print("特征选择，删除列: %s" % str(order_columns[0:bst_i]))
    return X_train_slt, y_train, X_test_slt
def lgb_tune(X_train, y_train, k=5):
    """lgb调参

    :param X_train: 训练集特征
    :param y_train: 训练集标记
    :return: param, 参数字典
    """
    base_param = {
        "learning_rate": 0.1,
        "n_estimators": 100,
        "num_leaves": 31,
        "min_child_samples": 20,
        "subsample": 1.,
        "colsample_bytree": 1.,

        "boosting_type": "gbdt",
        "objective": "binary",
        "reg_lambda": 0,
        "metric": "f1",
        "verbosity": 0
    }
    # 1. 学习率、学习器数量
    param_1 = {
        "learning_rate": [0.1, 0.05, 0.01, 0.001]
    }
    # fold = KFold(5, True, 0)
    gsearch1 = GridSearchCV(lgb.LGBMClassifier(**base_param), param_1, "f1", n_jobs=-1, cv=SDFold(k).split(X_train, y_train))
    gsearch1.fit(X_train.values, y_train.values.ravel())
    base_param.update(gsearch1.best_params_)
    param_2 = {
        "n_estimators": [100, 200, 500, 1000, 2000]
    }
    gsearch2 = GridSearchCV(lgb.LGBMClassifier(**base_param), param_2, "f1", n_jobs=-1, cv=SDFold(k).split(X_train, y_train))
    gsearch2.fit(X_train.values, y_train.values.ravel())
    base_param.update(gsearch2.best_params_)
    # 2. 叶子数量、叶节点数据量
    param_3 = {
        "num_leaves": range(15, 51, 1)
    }
    gsearch3 = GridSearchCV(lgb.LGBMClassifier(**base_param), param_3, "f1", n_jobs=-1, cv=SDFold(k).split(X_train, y_train))
    gsearch3.fit(X_train.values, y_train.values.ravel())
    base_param.update(gsearch3.best_params_)
    # 3. 行采样、列采样
    param_4 = {
        "subsample": [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.]
    }
    gsearch4 = GridSearchCV(lgb.LGBMClassifier(**base_param), param_4, "f1", n_jobs=-1, cv=SDFold(k).split(X_train, y_train))
    gsearch4.fit(X_train.values, y_train.values.ravel())
    base_param.update(gsearch4.best_params_)
    # 4. L1、L2正则化
    param_5 = {
        "reg_alpha": [0, 1e-5, 1e-2, 0.1, 1, 10],
        "reg_lambda": [0, 1e-5, 1e-2, 0.1, 1, 10]
    }
    gsearch5 = GridSearchCV(lgb.LGBMClassifier(**base_param), param_5, "f1", n_jobs=-1, cv=SDFold(k).split(X_train, y_train))
    gsearch5.fit(X_train.values, y_train.values.ravel())
    base_param.update(gsearch5.best_params_)
    # 5. 降低学习器数量
    param_6 = {
        "learning_rate": [0.1, 0.05, 0.01, 0.001],
        "n_estimators": [100, 200, 500, 1000, 2000]
    }
    gsearch6 = GridSearchCV(lgb.LGBMClassifier(**base_param), param_6, "f1", n_jobs=-1, cv=SDFold(k).split(X_train, y_train))
    gsearch6.fit(X_train.values, y_train.values.ravel())
    base_param.update(gsearch6.best_params_)
    return base_param
def lgb_predict(input_dir, output_file = None):
    """lgb预测

    :param input_dir: 输入数据目录的路径
    :param output_file: 输出文件的文件路径， 若为空，则不保存
    :return: y_pred, 测试集预测值
    """
    X_train, y_train, X_test = load_data(input_dir)
    X_train, y_train, X_test = feat_select(X_train, y_train, X_test)
    param = lgb_tune(X_train, y_train)
    gbm = lgb.LGBMClassifier(**param)
    # 线下测试
    score = cross_val_score(gbm, X_train.values, y_train.values.ravel(), scoring="f1", cv=SDFold(5).split(X_train, y_train), n_jobs=-1)
    print(" LGB线下得分(5倍交叉验证)：" + str(np.array(score).mean()))
    gbm.fit(X_train.values, y_train.values.ravel())
    y_pred = gbm.predict(X_test.values)
    if output_file != None:
        pd.DataFrame(y_pred).to_csv(output_file, float_format='%.0f', header=False, index=False)
    return y_pred
###################
#K折lgb
###################
class KSingleModel:
    """K折平均分类器

    """
    def __init__(self, param, cv, threshold = 0.5, plus = False):
        """

        :param param: lgb模型的参数
        :param cv: 内部基学习器的个数
        :param threshold: 阈值，大于阈值为1， 否则为0
        """
        self.models = [lgb.LGBMClassifier(**param) for i in range(cv)]
        self.cv = cv
        self.threshold = threshold
        if plus == True:
            self.plus = lgb.LGBMClassifier(**param)
        else:
            self.plus = None
    def fit(self, X, y):
        fold = SDFold(self.cv).split(X, y)
        for i, (tr_index, va_index) in enumerate(fold):
            X_tr = X.iloc[tr_index].values
            y_tr = y.iloc[tr_index].values.ravel()
            X_va = X.iloc[va_index].values
            y_va = y.iloc[va_index].values.ravel()
            self.models[i].fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=100, verbose=False)
        if self.plus != None:
            self.plus.fit(X.values, y.values.ravel())
        return self
    def predict(self, X):
        y_preds = []
        for i in range(self.cv):
            y_pred = self.models[i].predict(X)
            y_preds.append(y_pred)
        y_preds = np.array(y_preds)
        pred = y_preds.mean(axis=0)
        pred[pred>=self.threshold] = 1
        pred[pred<self.threshold] = 0
        if self.plus != None:
            pred_plus = self.plus.predict(X)
            mask = pred_plus == 1
            pred[mask] = 1
        return pred
def ksm_score(base_param, base_cv, base_threshold, base_plus, X_train, y_train, cv=5):
    """KSM f1得分

    :param base_param: KSM第一个参数
    :param base_cv: KSM第二个参数
    :param base_times: KSM个数
    :param base_threshold: KSM阈值
    :param X_train: 训练集特征
    :param y_train: 训练集标记
    :param cv: 交叉验证倍数
    :return: 得分
    """
    scores = []
    for tr_index, te_index in SDFold(cv).split(X_train, y_train):
        X_tr = X_train.iloc[tr_index].set_index(pd.RangeIndex(0, len(tr_index)))
        y_tr = y_train.iloc[tr_index].set_index(pd.RangeIndex(0, len(tr_index)))
        X_te = X_train.iloc[te_index].values
        y_te = y_train.iloc[te_index].values.ravel()
        te_pred = KSingleModel(base_param, base_cv, base_threshold, base_plus).fit(X_tr, y_tr).predict(
            X_te)
        scores.append(f1_score(y_te, te_pred))
    score = np.array(scores).mean()
    return score
def ksm_tune(param, X_train, y_train, cv_list = [2, 3, 4, 5, 6, 7, 8, 9, 10],
             threshold_list = [0.00001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6], k=5):
    """ 通过参数列表，选出集成的参数

    :param param: lgb参数
    :param X_train: 训练集特征
    :param y_train: 训练集标记
    :param cv_list: ksm学习器数量
    :param threshold_list: ksm阈值
    :param cv: 交叉验证倍数
    :return: best_cv, best_threshold
    """
    best_score = 0
    best_cv = 0
    best_threshold = 0
    best_plus = False
    for base_cv in cv_list:
        for base_threshold in threshold_list:
            for plus in [True, False]:
                score = ksm_score(param, base_cv, base_threshold, plus, X_train, y_train, k)
                if score > best_score:
                    best_score = score
                    best_cv = base_cv
                    best_threshold = base_threshold
                    best_plus = plus
    return best_cv, best_threshold, best_plus
def ksm_predict(input_dir, output_file = None):
    """ksm预测

    :param input_dir: 输入数据目录的路径
    :param output_file: 输出文件的文件路径， 若为空，则不保存
    :return: y_pred, 测试集预测值
    """
    # 特征工程
    X_train, y_train, X_test = load_data(input_dir)
    X_train, y_train, X_test = add_feat(X_train, y_train, X_test)
    X_train, y_train, X_test = feat_select(X_train, y_train, X_test, 10)
    best_param = lgb_tune(X_train, y_train, 10)
    # 交叉验证学习阈值
    best_cv, best_threshold, best_plus = ksm_tune(best_param, X_train, y_train, k=10)
    # 线下测试
    best_score = ksm_score(best_param, best_cv, best_threshold, best_plus, X_train, y_train, 10)
    print("best param: " + str(best_param))
    print("best cv: %d" % best_cv)
    print("best threshold: %.2f" % best_threshold)
    print("best plus: "+str(best_plus))
    print("ksm线下得分(10倍交叉验证)： %.5f" % best_score)
    # 预测保存
    ksm = KSingleModel(best_param, best_cv, best_threshold, best_plus)
    ksm.fit(X_train, y_train)
    y_pred = ksm.predict(X_test.values)
    if output_file != None:
        pd.DataFrame(y_pred).to_csv(output_file, float_format='%.0f', header=False, index=False)
    return y_pred, best_score
if __name__ == "__main__":
    train_dir = r'data\f_diabetes3_20180307'
    output_file = r'data\d3sdfold20180307hk.csv'
    _, score = ksm_predict(train_dir, output_file)
    pass