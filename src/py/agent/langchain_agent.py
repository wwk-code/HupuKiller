import os 
from langsmith import wrappers, traceable
from langchain.agents import AgentExecutor,create_tool_calling_agent
from langchain_core.tools.simple import Tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from typing import Callable,Any
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.manager import CallbackManager
from langchain.llms.base import LLM
from langchain.memory import ConversationBufferMemory


import os,sys



# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)


from common.myCustomPath import *
from common.myFile import *
from agent.agent_tools import *
from agent.agent_3d_client import *


class LangAgent:

    def __init__(self):
        
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "你是一名AI助手,请根据用户问题回答关于NBA领域的提问,这个过程中你需要根据输入prompts以及问题中可能存在的`NBA问题标签`,如NBATag_0来选择不同的Tool.注意: 你后续如果要调用Tool,请保持给入的query参数为问题本身,不要自己对问题做改动然后给入工具.同时注意,你在中间调用Tool时可能会需要用英文输入,但最终返回给用户的那一次输出,请用中文输出"),
                ("placeholder", "{chat_history}"),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),   # agent_scratchpad 会记录当前对话轮次的中间Tool调用,将它们充当上下文,与Memory的区别是无法跨越多轮对话记录
            ]
        )
        
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",  # 用于注入到 Prompt 的 Key
            return_messages=True   # 以 Message 格式存储，支持 HumanMessage, AIMessage
        )

        self.tavilyTool = tavilyTool()
        self.wikiTool = wikiTool()

        self.tools = [self.tavilyTool,self.wikiTool]

        self.model = ChatOpenAI(
            api_key=os.getenv("QWEN_API_KEY"),  
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model = 'qwen-turbo'  # 目前选择调用的是 qwen_turbo 模型
        )

        self.agent = create_tool_calling_agent(
            llm = self.model, 
            tools = self.tools, 
            prompt = self.prompt
        )
        # self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True, memory=self.memory)
        self.agent_executor = AgentExecutor(agent=self.agent, tools=self.tools, verbose=True)
        
        temp = 1

    @traceable
    def run(self,input: str):
        response = self.agent_executor.invoke(input)
        return response


if __name__ == '__main__':
    
    langAgent = LangAgent()

    input = {'input' : '现在我提供NBA问题标签: NBATag_2 ; 问题: 请向我介绍科比'}

    response = langAgent.run(input)

    # input = {'input' : '现在我提供NBA问题标签: NBATag_0 ; 问题: 那他是在哪一年入选的NBA75大'}
    input = {'input' : '问题: 那他是在哪一年入选的NBA75大'}
    response = langAgent.run(input)
    print(response)
    
    temp = 1

