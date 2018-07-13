# coding: utf-8
"""
糖尿病病例 数据预处理
"""
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import fancyimpute as fi
from os import listdir, mkdir
from convert import convert
from datetime import datetime
import matplotlib.pyplot as plt

trainfile = r"d_train_20180102_1.csv" # 训练集文件目录
testfile = r"d_test_B_20180128.csv" # 测试集文件目录
outputdir = r"./data" # 预处理文件保存目录

def preprocess(trainfile, testfile, outputdir, useless_attr, miss_threshold, xstrategy,
               ymin, ymax, ystrategy, fill_method="MICE", normal01 = True):
    """对XY进行数据预处理，矩阵补全、正则化标准化等。

    :param trainfile: string, 训练集(d_train_20180102.csv)的路径
    :param testfile: string, 测试集(d_test_A_20180102.csv)的路径
    :param outputdir: string, 预处理后文件保存的路径
    :param useless_attr: list, 需要删除的无用属性，比如[0, 1, 2, 3]
    :param miss_threshold: float, 属性确实严重忽略的阈值，百分比，比如0.7
    :param xstrategy: string, 对x中奇异点的处理方式{"replace", "nothing"}
    :param ymin: float, 对Y中点的最小值，小于这个值，即为奇异点
    :param ymax: float, 对Y中点的最大值，超过这个值，就是奇异点
    :param ystrategy: string, 对y中奇异点的处理方式("delete", "replace", "nothing")
    :param fill_method: string, 矩阵补全的策略，{"KNN", "SoftI", "MF", "MICE"}
    :param normal01: bool, 如果为真，则对结果进行归一化到01，否则，不归一化
    :return: list, 归一化之后的trainX, trainY, testX
    """
    # 0. 读入训练集，测试集
    train_XY = convert(trainfile)
    test_X = convert(testfile)
    print("读入数据集，开始数据预处理")

    # 1. 删除无用属性列
    train_id = train_XY[:, 0:1]
    test_id = test_X[:, 0:1]
    train_XY = np.delete(train_XY, useless_attr, axis=1)
    test_X = np.delete(test_X, useless_attr, axis=1)
    n_test = test_X.shape[0]
    info1 = "1. 删除train_XY, test_X上的无用属性:%s, train_X.shape=%s, test_X.shape=%s"\
          %(str(useless_attr), str(train_XY.shape), str(test_X.shape))
    print(info1)

    # 2. 删除缺失严重的列
    miss_mask = np.isnan(train_XY)
    n = miss_mask.shape[0]
    column_del = [] # 删除列的list
    for i in range(miss_mask.shape[1]):
        miss_n = miss_mask[:,i].sum()
        if miss_n/n >= miss_threshold:
            column_del.append(i)
    train_XY = np.delete(train_XY, column_del, axis=1)
    test_X = np.delete(test_X, column_del, axis=1)
    info2 = "2. 在train_XY, test_X上删除缺失超过%f%%的属性:%s" %(miss_threshold*100, str(column_del))
    print(info2)

    # 3. 对y进行去噪，手动设置阈值
    train_Y = train_XY[:, -1:]
    upper_mask = train_Y > ymax
    lower_mask = train_Y < ymin
    if ystrategy == "replace":
        train_Y[upper_mask] = ymax
        train_Y[lower_mask] = ymin
    elif ystrategy == "delete":
        index = np.array(np.arange(0, train_Y.shape[0], 1), ndmin=2).T
        chsn_mask = upper_mask | lower_mask
        train_XY = np.delete(train_XY, index[chsn_mask], axis=0)
        train_id = np.delete(train_id, index[chsn_mask], axis=0)
    elif ystrategy == "nothing":
        pass
    else:
        raise ValueError(r"'ystrategy'应该是{nothing, replace, delete}中的一个")
    train_Y = train_XY[:, -1:]
    print("3. 对trainY去噪(%s)，trainXY.shape=%s" %(ystrategy, train_XY.shape))

    # 4. 对X进行操作，通过boxplot计算阈值
    train_X = train_XY[:, :-1]
    all_X = np.concatenate([train_X, test_X], axis=0)
    attr_n = train_XY.shape[1] - 1
    attr_min_max = np.zeros((attr_n, 2), dtype=np.float64)  # 存储每个属性经过boxplot之后的最小最大值，即阈值array
    if xstrategy == "nothing":
        pass
    elif xstrategy == "replace":
        # 对X中的奇异点 替换为 最值
        for i in range(attr_n):
            # 对每列进行浅拷贝，对crt_attr操作相当于对train_XY操作
            crt_attr = all_X[:,i:i+1]
            miss = np.isnan(crt_attr)
            box_dic = plt.boxplot(crt_attr[~miss])
            crt_max = box_dic["caps"][0].get_ydata()[0]
            crt_min = box_dic["caps"][1].get_ydata()[0]
            if crt_max < crt_min:
                tmp = crt_max
                crt_max = crt_min
                crt_min = tmp
            attr_min_max[i, 0] = crt_min
            attr_min_max[i, 1] = crt_max
            crt_attr[miss] = 0
            upper_mask = crt_attr > crt_max
            lower_mask = crt_attr < crt_min
            upper_mask &= ~miss
            lower_mask &= ~miss

            crt_attr[upper_mask] = crt_max
            crt_attr[lower_mask] = crt_min
            crt_attr[miss] = np.nan
    else:
        raise ValueError(r"'xstrategy'应该是{nothing, replace}中的一个")
    print(r"4. 对所有的X进行去噪(%s)." %xstrategy)

    # 5. 矩阵补全
    completer = None
    if fill_method == "KNN":
        completer = fi.KNN(verbose=False)
    elif fill_method == "SoftI":
        completer = fi.SoftImpute(verbose=False)
    elif fill_method == "MF":
        completer = fi.MatrixFactorization(verbose=False)
    elif fill_method == "MICE":
        completer = fi.MICE(verbose=False)
    else:
        ValueError(r"'fill_method'应该是{'KNN','SoftI','MF','MICE'}.")
    all_X_complete = completer.complete(all_X)
    print("5. 在all_X上进行矩阵补全(%s)." % fill_method)

    # train_X = all_X_complete[:-1000, :]
    # test_X = all_X_complete[-1000:, :]
    # 6. 归一化，以及01缩放
    if normal01:
        X_nmler = StandardScaler()
        X_01 = MinMaxScaler()
        Y_nmler = StandardScaler()
        Y_01 = MinMaxScaler()

        X_nmler.fit(all_X_complete)
        Y_nmler.fit(train_Y)
        all_X_nml = X_nmler.transform(all_X_complete)
        train_Y_nml = Y_nmler.transform(train_Y)
        X_01.fit(all_X_nml)
        Y_01.fit(train_Y_nml)
        all_X_nml01 = X_01.transform(all_X_nml)
        train_Y_nml01 = Y_01.transform(train_Y_nml)
        final_train_X = all_X_nml01[:-n_test, :]
        final_test_X = all_X_nml01[-n_test:, :]
        final_train_Y = np.concatenate([train_Y_nml01, train_Y], axis=1)
    else:
        final_train_X = all_X_complete[:-n_test, :]
        final_test_X = all_X_complete[-n_test:, :]
        final_train_Y = train_Y
    print(r"6. 对all_X, train_Y归一化到01(%s)." % normal01)

    # 7. 存储数据
    print(r"7. 存储数据为: 集合_类别_日期.csv(%s)." % outputdir)
    # timestamp = datetime.now().strftime("%Y%m%d%H%M")
    timestamp = "0000"
    np.savetxt(outputdir+r"\train_X_"+timestamp+".csv", final_train_X, delimiter=",")
    np.savetxt(outputdir+r"\test_X_"+timestamp+".csv", final_test_X, delimiter=",")
    np.savetxt(outputdir+r"\train_Y_"+timestamp+".csv", final_train_Y, delimiter=",")
    np.savetxt(outputdir+r"\train_id_"+timestamp+".csv", train_id.astype(np.int64), delimiter=",")
    np.savetxt(outputdir+r"\test_id_"+timestamp+".csv", test_id.astype(np.int64), delimiter=",")
    return train_X, train_Y, test_X, train_id

if __name__ == "__main__":
    if outputdir not in listdir("."):
        mkdir(outputdir)

    print("开始运行")
    preprocess(trainfile, testfile,outputdir,
               [0, 3], 1.0,"nothing", 0, 50, "nothing", "MICE", True)
    print("处理完毕")