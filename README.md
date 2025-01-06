# HupuKiller - 1stVersionDev

### 0. Brief Introduction

**Project:**

基于 LLAMA3-8B-Chinese-Chat 基座模型的NBA知识问答系统，通过使用爬虫爬取NBA Official Website、各大论坛的NBA相关数据，利用其中15%的数据对模型使用Lora微调，剩余数据接入本地知识库用以RAG。

**Branch:**

目前 1stVersionDev 已实现根据用户问题回复关于1990-2023年的NBA历年总决赛各球员的场均数据(部分情况下模型的输出格式不统一，甚至出现乱回复的情况)。1stVersionDev特点: 1. 并未集成langchain，封装了milvus、mysql操作类 2.高度定制化的LoRA-SFT数据，精确控制了模型的输入和输出格式、内容等  3.尚未集成RLHF和langchain，计划于 2stVersionDev 完成这两个部分的集成，实现稳定回答关于1990-2023年的NBA历年总决赛各球员的场均数据，并且尽量不损失模型的预训练时获得的基础能力，能够回复其他的问题。

### 1. Overall Shcheme

**系统开发进度:**   目前项目的开发版本为 version 1 ,详细信息请查看 docs/SysScheme.md

### 2. SubDirectory Notation

**docs**: 其下存放了项目开发过程中的各类文档

**src**: 项目源码目录

**test**: 测试代码目录

**asstes**: 项目资源目录

**outputs**: 项目输出目录

**scripts**: 项目工具脚本目录
