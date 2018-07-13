# -*- coding: utf-8 -*-
"""阿里天池复赛test_b数据预处理

@author: He Kai
@contact: matthewhe@foxmail.com
@file: preprocess3.py
@time: 2018/3/6 14:25
"""
from time import time
import fancyimpute as fi
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, minmax_scale, MinMaxScaler

def preprocess(train_data, test_data, fill_times=1000, ignore_columns = ['id']):
    """对训练数据和测试数据进行预处理

    1. 对数据进行补全
    2. 对数据进行归一化

    :param train_data: DataFrame, 通过pandas读入的训练集数据
    :param test_data: DataFrame, 通过pandas读入的测试集数据
    :return: X_train, y_train, X_test
    """
    feature_name = test_data.columns
    label_name = list(set(train_data.columns) - set(feature_name))
    train_label = train_data[label_name]
    train_feature = train_data[feature_name]
    test_feature = test_data
    test_feature = test_feature.set_index(test_feature.index + 1000)
    train_index = train_feature.index
    test_index = test_feature.index
    all_feature = pd.concat([train_feature, test_feature], axis=0)
    all_index = all_feature.index

    snp_columns = ['SNP1', 'SNP2', 'SNP3', 'SNP4', 'SNP5', 'SNP6', 'SNP7', 'SNP8',
           'SNP9', 'SNP10', 'SNP11', 'SNP12', 'SNP13', 'SNP14', 'SNP15', 'SNP16',
           'SNP17', 'SNP18', 'SNP19', 'SNP20', 'SNP21', 'SNP22', 'SNP23', 'SNP24', 'SNP25',
           'SNP26', 'SNP27', 'SNP28', 'SNP29', 'SNP30', 'SNP31', 'SNP32', 'SNP33',
           'SNP34', 'SNP35', 'SNP36', 'SNP37', 'SNP38', 'SNP39', 'SNP40',
           'SNP41', 'SNP42', 'SNP43', 'SNP44', 'SNP45', 'SNP46', 'SNP47', 'SNP48',
           'SNP49', 'SNP50', 'SNP51', 'SNP52', 'SNP53', 'SNP54', 'SNP55']
    other_columns = set(all_feature.columns) - set(snp_columns)
    other_columns = list(other_columns)
    all_feature[snp_columns] = all_feature[snp_columns].fillna(0)
    snp_feature = all_feature[snp_columns]
    snp_scale = MinMaxScaler()
    snp_scale.fit([[0],[3]])
    snp_feature01 = snp_scale.transform(snp_feature)
    snp_feature_final = pd.DataFrame(snp_feature01, columns=snp_columns, index=all_index)
    ######################
    # 数据填充
    ######################
    feature_complete = []
    t0 = time()
    for i in range(fill_times):
        mice_data = fi.MICE(verbose=False).complete(all_feature[other_columns])
        feature_complete.append(mice_data)
        remain_time = (time() - t0) / (i+1) * (fill_times - i - 1)
        print("第 %2.d/%d 次填充, 剩余时间 %.0f s" % (i+1, fill_times, remain_time))
    feature_complete = np.array(feature_complete)
    feature_filled = feature_complete.mean(axis=0)
    all_feature[other_columns] = feature_filled
    other_feature = all_feature[other_columns]
    int_column = ['年龄', '孕次', '产次', 'BMI分类', '收缩压', '舒张压', 'ALT', 'AST', 'Lpa', 'DM家族史', 'ACEID']
    float2_column = ['BUN',
                    'ApoA1',
                    'CHO',
                    'wbc',
                    '孕前体重',
                    'HDLC',
                    'Cr',
                    'RBP4',
                    'ApoB',
                    '分娩时',
                    '身高',
                    '糖筛孕周',
                    'TG',
                    'LDLC',
                    'hsCRP']
    float5_column = ['孕前BMI']
    float6_column = ['VAR00007']
    other_feature.loc[:][int_column] = other_feature[int_column].round()
    other_feature.loc[:][float2_column] = other_feature[float2_column].round(2)
    other_feature.loc[:][float5_column] = other_feature[float5_column].round(5)
    other_feature.loc[:][float6_column] = other_feature[float6_column].round(6)
    ######################
    # 数据归一化
    ######################
    id_feature = other_feature[ignore_columns]

    feature_need_nml = other_feature.drop(ignore_columns, axis=1)
    feature_nml = scale(feature_need_nml.values, axis=0)
    feature_nml01 = minmax_scale(feature_nml, axis=0)
    other_feature_final = pd.DataFrame(feature_nml01, columns=feature_need_nml.columns, index=all_index)

    feature_final = pd.concat([id_feature, snp_feature_final, other_feature_final], axis=1)
    train_feature = pd.DataFrame(feature_final.ix[train_index], columns=feature_name)
    train_label = pd.DataFrame(train_label, columns=["label"])
    test_feature = pd.DataFrame(feature_final.ix[test_index], columns=feature_name)
    return train_feature, train_label, test_feature
def preprocess_as_file(train_file, test_file, output_dir, fill_times=1000):
    """将test_a作为测试集

    将train作为训练集

    :param train_file: string, 训练集文件路径
    :param test_file: string, 测试集文件路径
    :param fill_times: int, 填充次数
    :return:
    """
    train_data = pd.read_csv(train_file, encoding="gbk")
    test_data = pd.read_csv(test_file, encoding="gbk")
    train_feature, train_label, test_feature = preprocess(train_data, test_data, fill_times)
    train_feature.to_csv(output_dir+"/"+"X_train.csv", index=False, encoding="gbk")
    train_label.to_csv(output_dir+"/"+"y_train.csv", index=False, encoding="gbk")
    test_feature.to_csv(output_dir+"/"+"X_test.csv", index=False, encoding="gbk")
def preprocess_as_file_ab(train_file, test_file_a, test_file_b, output_dir, fill_times=1000):
    """将test_a + test_b作为测试集

    将train作为训练集

    :param train_file: string, 训练集文件路径
    :param test_file_a: string, 测试集a文件路径
    :param test_file_b: string, 测试集b文件路径
    :param output_dir: 输出文件夹路径
    :param fill_times: int, 填充次数
    :return:
    """
    train_data = pd.read_csv(train_file, encoding='gbk')
    test_data_a = pd.read_csv(test_file_a, encoding='gbk')
    test_data_b = pd.read_csv(test_file_b, encoding='gbk')
    n_test_data_a = test_data_a.shape[0]
    n_test_data_b = test_data_b.shape[0]
    test_data = pd.concat([test_data_a, test_data_b.set_index(test_data_b.index + n_test_data_a)], axis=0)
    n_test_data = n_test_data_a + n_test_data_b
    train_feature, train_label, test_feature = preprocess(train_data, test_data, fill_times)
    test_feature_a = test_feature.iloc[0:n_test_data_a]
    test_feature_b = test_feature.iloc[n_test_data_a:n_test_data]
    train_feature.to_csv(output_dir+'/'+'X_train.csv', index=False, encoding='gbk')
    train_label.to_csv(output_dir+'/'+'y_train.csv', index=False, encoding='gbk')
    test_feature_a.to_csv(output_dir+'/'+'X_test_a.csv', index=False, encoding='gbk')
    test_feature_b.to_csv(output_dir+'/'+'X_test_b.csv', index=False, encoding='gbk')
def preprocess_as_file_b(train_file, test_file_a, test_file_answer_a, test_file_b, output_dir, fill_times=1000):
    """将test_b作为测试集

    train + test_a作为训练集

    :param train_file: string, 训练集文件路径
    :param test_file_a: string, 测试集a文件路径
    :param test_file_answer_a: 测试集a答案路径
    :param test_file_b: string, 测试集b文件路径
    :param output_dir: 输出文件夹路径
    :param fill_times: int, 填充次数
    :return:
    """
    train_data_ori = pd.read_csv(train_file, encoding='gbk')
    test_data_a = pd.read_csv(test_file_a, encoding='gbk')
    test_data_ans_a = pd.read_csv(test_file_answer_a, encoding='gbk', header=None)
    test_data_ans_a.columns = ['label']
    test_data_b = pd.read_csv(test_file_b, encoding='gbk')
    train_data_a = pd.concat([test_data_a, test_data_ans_a], axis=1)
    train_data_a = train_data_a.set_index(train_data_a.index + train_data_ori.shape[0])
    train_data = pd.concat([train_data_ori, train_data_a.set_index(train_data_a.index + train_data_ori.shape[0])], axis=0)
    train_feature, train_label, test_feature = preprocess(train_data, test_data_b, fill_times)
    train_feature.to_csv(output_dir+'/'+'X_train.csv', index=False, encoding='gbk')
    train_label.to_csv(output_dir+'/'+'y_train.csv', index=False, encoding='gbk')
    test_feature.to_csv(output_dir+'/'+'X_test.csv', index=False, encoding='gbk')
if __name__ == "__main__":
    train_file = r'data\raw_data\f_train_20180204.csv'
    test_file_a = r'data\raw_data\f_test_a_20180204.csv'
    test_file_ans_a = r'data\raw_data\f_answer_a_20180306.csv'
    test_file_b = r'data\raw_data\f_test_b_20180305.csv'
    times = 1000
    # train作为训练集, test_a, test_b作为测试集
    # output_dir_ab = r'E:\Semifinal\submission\data\f_diabetes3_20180306'
    # preprocess_as_file_ab(train_file, test_file_a, test_file_b, output_dir_ab, times)
    # train+test_a作为训练集， test_b作为测试集
    output_dir_b = r'data\f_diabetes3_20180307'
    preprocess_as_file_b(train_file, test_file_a, test_file_ans_a, test_file_b, output_dir_b, times)