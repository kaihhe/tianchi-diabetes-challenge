# -*- coding: utf-8 -*-
"""

@author: He Kai
@contact: matthewhe@foxmail.com
@file: main_nml_20180129.py.py
@time: 2018/1/29 10:28
"""
from preprocess.preprocess import preprocess

trainfile = r"data/d_train_20180129.csv"
testfile = r"data/d_test_B_20180128.csv"
outputdir = r"data/preprocesskma"

print("开始运行")
preprocess(trainfile, testfile, outputdir,
           [0, 3], 1.0, "nothing", 0, 50, "nothing", "MICE", True)
print("处理完毕")