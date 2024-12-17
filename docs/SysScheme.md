# HupuKiller - System Scheme

**Note:** 关于系统整体技术栈选型、系统全流程设计、系统架构等系统性信息请查看 SysUp2Down.docx 文档

### 1. HupuKiiler - 1st Version系统功能设定:

目前对HupuKiller第一版的功能预期设定如下:

**Finished:**

支持回复1990年后的历年总决赛所有球队/球员的数据、胜率、记录等事实上的信息

数据库Pipeline搭建(Mysql + Milvus),编写了对应的封装类代码

**TODO：**

**Prior:**

使用RLHF对模型输出做规约化处理(符合自定义的结构)  

搭建RAG系统,实现检索知识库并生成模型回复内容

搭建简易chatbot应用

训练一个NLU分类模型,协助生成LLAMA的inputs

用TensorRT部署部署NLU模型,VLLM部署LLama

**Common:**

支持chatBot主观的评价某位球员、某支球队的某些方面的状况，例如这位球员在某场比赛表现如何、最近表现如何，打得好不好等，类似于让chatBot充当一位主观的NBA评论员的角色。这其中涉及到对模型的推理能力的微调以及RAG Pipeline的搭建

支持chatBot基本的主观探讨的能力，例如: "你认为2000年后的决赛之王是谁"。"你认为2015年库里和伊戈达拉谁更应该获得FMVP"等基本的开放性问题等(需爬取虎扑等论坛的用户评论)。

支持模型识别球员外号

上线微信群聊机器人

(支持模型根据不同回答生成对应的GIF功能)
