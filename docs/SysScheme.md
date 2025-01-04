# HupuKiller - System Scheme

**Note:** 关于系统整体技术栈选型、系统全流程设计、系统架构等系统性信息请查看 SysUp2Down.docx 文档

### 1. HupuKiiler - 2st Version系统功能设定:

目前对HupuKiller第一版的功能预期设定如下:

**Finished:**

支持回复1990年后的历年总决赛所有球队/球员的数据、胜率、记录等事实上的信息

数据库Pipeline搭建(Mysql + Milvus),编写了对应的封装类代码

使用DPO对模型针对NBA总决赛球员平均数据的输出做规约化处理(符合自定义的结构)

**TODO：**

**Prior:**

调用Qwen-trubo API，实现实时的NBA近期比赛数据查询。所有的操作全部用Langchain API实现

具体功能(暂定): 在NBA中国官网上，根据用户提问: "xxx.xxx的比赛有哪些" -> llm根据搜索结果列举出来，然后用户提问其中的某一场比赛时，llm可以进一步去其数据统计页面搜索，然后提取出对应逻辑并生成相应的信息给用户


将本地LLAMA3模型推理逻辑迁移至VLLM

**3st Version TODO：**

训练一个NLU分类模型,做为Qwen和本地LLAMA的门控

用TensorRT部署部署NLU模型,将其集成进已有的langchain-agent中

使用 langgraph 重构 agents

搭建简易chatbot应用

**Common:**

支持chatBot主观的评价某位球员、某支球队的某些方面的状况，例如这位球员在某场比赛表现如何、最近表现如何，打得好不好等，类似于让chatBot充当一位主观的NBA评论员的角色。这其中涉及到对模型的推理能力的微调以及RAG Pipeline的搭建

支持chatBot基本的主观探讨的能力，例如: "你认为2000年后的决赛之王是谁"。"你认为2015年库里和伊戈达拉谁更应该获得FMVP"等基本的开放性问题等(需爬取虎扑等论坛的用户评论)。

支持模型识别球员外号

上线微信群聊机器人

(支持模型根据不同回答生成对应的GIF功能)
