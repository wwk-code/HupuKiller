# Traing Notes

### TODO

构造10000条数据集，具体的形式看 example.json


### 0. General Training Tricks

0.1 Data Augmention L2正则化 Dropout 早停 学习率warm-up 动态学习率

### 1. Fine-tuning BERT

1.1 学习率最好在 x * e-5 数量级，如果到 -4,容易发生灾难性遗忘

1.2 最好冻结BERT-base的前六层(理解通用语义)，微调后六层(更聚焦于下游任务)

### 2. Fine-tuning 标签记录

2.1 针对自己的微调LLAMA 例：詹姆斯和乔丹谁是Goat?

2.2 百科类: 请向我简要介绍乔丹

2.3 询问某场比赛的真实数据: 请告诉我xxx年xxx比赛xxx数据

2.4 斗图、来点表情包、开启斗图模式
