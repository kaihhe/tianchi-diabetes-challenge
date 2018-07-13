# TianChi Diabetes Game #

大赛网址：[https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11165320.5678.1.5bed6d79KvOsxW&raceId=231638](https://tianchi.aliyun.com/competition/introduction.htm?spm=5176.11165320.5678.1.5bed6d79KvOsxW&raceId=231638 "Game Page")

## Introduction ##

天池精准医疗大赛，其分为三个阶段初赛、复赛和决赛。初赛的题目是针对2型糖尿病的回归问题，根据受检者的体检数据和临床信息对血糖值进行预测。复赛的题目是针对妊娠糖尿病的二分类问题，通过体检信息和基因信息预测出是否患有妊娠糖尿病。决赛是在现场进行答辩。

本人是积极向上团队中的队员，积极向上团队再精准医疗大赛中取得了初赛top-11和复赛top-6的成绩。当前代码仓库仅仅包含了本人参赛中的思路和代码，当公布初赛结果时，我发现初赛预测结果比我们提交的效果更好，后来也没有仔细分析，仅仅把当时的代码稍微整理一下。团队复赛最终提交版的天池社区技术圈：[https://tianchi.aliyun.com/forum/new_articleDetail.html?spm=5176.8366600.0.0.36a8311fxpFaJX&raceId=231638&postsId=4714](https://tianchi.aliyun.com/forum/new_articleDetail.html?spm=5176.8366600.0.0.36a8311fxpFaJX&raceId=231638&postsId=4714 "提交思路")。

## Path ##

TianChi-Diabetes  
|  
|-- preliminary：初赛代码  
|  
|-- repecharge：复赛代码  

## Preliminary ##

- 预处理：矩阵补全技术SoftImpute填充缺失值、标准化、归一化
- 模型：lightgbm
- 调参：grid search
- 关键：样本中存在两类数据（正常血糖值、高血糖值），通过分类方法找到高血糖值得样本，对其进行整体预测值提高，达到精准得效果。
- 运行方式
	1. 数据预处理：preprocesskma.py
	2. 模型预测：kma.py

## Repecharge ##

- 预处理：SNP基因位点信息与其余信息进行分别填充，SNP填0，其余信息通过SoftImpute填充。
- 模型：lightgbm
- 调参：grid search
- 关键：模型融合,多次填充取均值。
- 运行方式（程序运行较慢）
	1. preprocess3.py
	2. lgb.py

## prerequisites ##

- numpy
- pandas
- scikit-learn
- lightgbm
- fancyimpute