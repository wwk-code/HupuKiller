# HupuKiller

### 0. Brief Introduction

基于 LLAMA3-8B-Chinese-Chat 基座模型的NBA知识问答系统，通过使用爬虫爬取NBA Official Website、各大论坛的NBA相关数据，利用其中15%的数据对模型使用Lora微调，剩余数据接入本地知识库用以RAG。

### 1. Overall Shcheme

**系统开发进度:**   目前项目的开发版本为 version 1 ,详细信息请查看 docs/SysScheme.md

### 2. SubDirectory Notation

**docs**: 其下存放了项目开发过程中的各类文档

**src**: 项目源码目录

**test**: 测试代码目录

**asstes**: 项目资源目录

outputs: 项目输出目录

### TODO-Overall:

完成HupuKiller-1stVersion的基础功能开发

### TODO-Specific:

1.获取NBA1990年后决赛数据源、各大论坛评论数据源,并将它们按80%-20%的比例分割为Train/Validate DataSets

2.对数据源进行前处理，然后取15%对模型进行LoRA微调

3.对微调后的模型做Validation Benchmark



s
