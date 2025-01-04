import os 
from openai import OpenAI
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


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)


class testTool_1(Tool):

    name: str 
    func: Callable[[] , Any] 
    description: str

    def __init__(self,name='testTool_1',description = "print '1111111111'. For any questions about custom_tag_1, you must use this tool!"):
        
        super(testTool_1,self).__init__(name=name,func=self._run,description=description)

    def _run(self, langchain_input: str = None):
        print('1111111111')
    
    def _arun(self, query):
        # 异步查询的实现
        pass



class testTool_2(Tool):
    name: str 
    func: Callable[[] , Any] 
    description: str

    def __init__(self,name:str = 'testTool_2',description:str = "print '2222222222'. For any questions about custom_tag_2, you must use this tool! And please remember that when you invoke this tool's _run method,you can't provide any parameters!"):
        super().__init__(name=name,func=self._run,description=description)

    def _run(self, langchain_input: str = None):
        print('2222222222')
    
    def _arun(self, query):
        # 异步查询的实现
        pass

@traceable
def pipeline(input: dict):

    response = agent_executor.invoke(input)

    return response

tool_1,tool_2 = testTool_1(),testTool_2()
tools = [tool_1,tool_2]


model = ChatOpenAI(
    api_key=os.getenv("QWEN_API_KEY"), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model = 'qwen-turbo'
)

# context = 'You are a helpful assistant.Please recall custom_tag_2,I need you invoke testTool_2!'

agent = create_tool_calling_agent(model, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

input = {"input": "Please recall custom_tag_2,I need you invoke testTool_2!"}
# input = {"input": "Please recall custom_tag_1,I need you invoke testTool_1! And Please make sure your output dosen't containt any func params!"}

pipeline(input)

temp = 1

