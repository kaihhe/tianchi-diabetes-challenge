# -*- coding: utf-8 -*-
"""

@author: He Kai
@contact: matthewhe@foxmail.com
@file: convert.py
@time: 2018/1/5 15:46
"""
import csv
from datetime import date
import numpy as np

def convert(inputfile):
    """读入比赛给出的csv文件，转换为array

    :param inputfile: string, 输入文件路径
    :return: list, XY_float,attr_name, XY转换成浮点数格式，以及属性名
    """
    # 1. 读入csv数据
    with open(inputfile, "r", encoding="gbk") as csv_file:
        reader = csv.reader(csv_file)
        data = []
        for row in reader:
            data.append(row)
    print(r"1. Load data[%s] successfully, shape(%d, %d)" % (inputfile, len(data), len(data[0])))
    attr_name = data[0]
    for i, name in enumerate(attr_name):
        attr_name[i] = str(name).encode("utf-8")
    # 2. 将性别转换为数字(1为男，0为女)
    for i in range(1, len(data)):
        sex = str(data[i][1])
        if sex==u"男":
            data[i][1] = 1
        elif sex==u"女":
            data[i][1] = 0
        else:
            data[i][1] = ""
    print(r"2. Convert sex to int{0,1}, shape(%d, %d)" %(len(data), len(data[0])))

    # 3. 将日期转换为，2017年的第n天
    begin_day = date(2017,1,1).toordinal()
    for i in range(1, len(data)):
        idate = str(data[i][3]).split("/")
        data[i][3] = date(int(idate[2]), int(idate[1]), int(idate[0])).toordinal() - begin_day
    print(r"3. Convert date to no.N day from 2017.01.01, shape(%d, %d)" %(len(data), len(data[0])))

    # 4. 略去，第一行字段名
    data_pure = data[1:][:]
    print("4. Ignore the attribute name, shape(%d, %d)" %(len(data_pure), len(data_pure[0])))

    # 5. 将缺失值补充为nan
    data_string = np.array(data_pure)
    data_empty = data_string == ''
    data_string[data_empty] = 'nan'
    print("5. complete with nan.")

    # 6. 转换为numpy.array, dtype=float64
    XY_float = np.array(data_string, dtype=np.float64)
    print(r"6. Convert to numpy.array, dtype:float64.")
    return XY_float