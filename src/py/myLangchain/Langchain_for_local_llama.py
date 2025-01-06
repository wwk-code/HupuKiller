from langchain_core.tools.simple import Tool
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain,SimpleSequentialChain
from langchain_core.language_models.llms import LLM
from typing import Any, List, Optional, Dict, Iterator
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
import os,sys
from typing import Callable,ClassVar,Any, Dict, List
from langchain.chains.base import Chain
from langchain.agents import initialize_agent,AgentType

# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)


from common.myCustomPath import *
from common.myFile import *
from databaseProcessing.vectorDatabaseProcessing import NBAFinalAverageDatasMilvusHandler
from infer.llama3Infer_hg_generate import ThreadSafeLlama3Infer



class CustomLlama3LLM(LLM):
    
    threadsafe_llama3_infer: ThreadSafeLlama3Infer
    
    def __init__(self, threadsafe_llama3_infer: ThreadSafeLlama3Infer):
        super(CustomLlama3LLM,self).__init__(threadsafe_llama3_infer=threadsafe_llama3_infer)

    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        
        # Handle the case where stop is not supported by the model
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")

        # Perform the inference using the llama3_infer object
        response = self.threadsafe_llama3_infer.infer(prompt)
        return response

    def _stream(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[GenerationChunk]:
        # Handle the case where stop is not supported by the model
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        
        # Use the same inference logic, but yield chunks for streaming
        response = self.threadsafe_llama3_infer.infer(prompt)
        for char in response:
            chunk = GenerationChunk(text=char)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "model_name": "CustomLlama3LLM",  # Specify the custom model name
        }

    @property
    def _llm_type(self) -> str:
        return "custom_llama3"



# 封装 NBAFinalAverageDatasMilvusHandler 为 Tool
class QueryNBAFinalAverageDataForQuestionFromMilvusTool(Tool):
    
    name: str
    func: Callable[[list[str],Optional[int]] , Any] 
    description: str
    NBAFinalAverageDatasMilvusHandler_obj: NBAFinalAverageDatasMilvusHandler

    # input_keys: ClassVar[list[str]] = ["questions"]  
    # output_keys: ClassVar[list[str]] = ["query_results"]
    
    
    def __init__(
                    self,
                    name:str = "QueryNBAFinalAverageDataForQuestionFromMilvus" ,
                    description:str = "Tool for querying NBA final average data For Questions from Milvus.",
                    NBAFinalAverageDatasMilvusHandler_obj: NBAFinalAverageDatasMilvusHandler = None
                ):
        # 初始化 Tool 类所需的参数
        super().__init__(name=name, func=self._run, description=description,NBAFinalAverageDatasMilvusHandler_obj=NBAFinalAverageDatasMilvusHandler_obj)
            
    def _run(self, questions:list[str],top_k = None):
        if top_k is None:
            query_results = self.NBAFinalAverageDatasMilvusHandler_obj.queryFromMysqlForQuestions(questions)
        else:
            query_results = self.NBAFinalAverageDatasMilvusHandler_obj.queryFromMysqlForQuestions(questions,top_k)
            
        return query_results[0]

    
    def _arun(self, query):
        # 异步查询的实现
        pass
    

class FormatQueryResultsToContextTool(Tool):
    
    name: str
    func: Callable[[list[str],Optional[int]] , Any] 
    description: str

    # input_keys: ClassVar[list[str]] = ["query_results"]  
    # output_keys: ClassVar[list[str]] = ["context"]
    
    def __init__(
                   self,
                   name:str = "FormatQueryResultsToContextTool" ,
                   description:str = "Tools for formating query results to context.",
               ):
       super(FormatQueryResultsToContextTool,self).__init__(name=name, func=self._run, description=description)    
    
    def _run(self, query_results: tuple) -> str:
        player_name = query_results[2]
        playing_time = query_results[3]
        age = query_results[4]
        avg_points = query_results[5]
        avg_rebounds = query_results[6]
        avg_assists = query_results[7]
        avg_steals = query_results[8]
        avg_blocks = query_results[9]
        context = f"背景知识：球员: {player_name} | 场均出场时间: {playing_time}分钟 | 年龄: {age}岁 | 场均得分: {avg_points}分 | 场均篮板: {avg_rebounds}个 | 场均助攻: {avg_assists}次 | 场均抢断: {avg_steals}次 | 场均盖帽: {avg_blocks}次。"

        return context
        

    @property
    def _arun(self):
        # 如果需要异步支持，可以实现异步版本
        pass






# 自定义Chain,将QueryNBAFinalAverageDataForQuestionFromMilvusTool和FormatQueryResultsToContextTool集成在此chain中
class CustomE2EToolChain(Chain):

    # 集成两个 Tool
    QueryNBAFinalAverageDataForQuestionFromMilvusTool_obj: QueryNBAFinalAverageDataForQuestionFromMilvusTool
    FormatQueryResultsToContextTool_obj: FormatQueryResultsToContextTool

    def __init__(self, QueryNBAFinalAverageDataForQuestionFromMilvusTool_obj: QueryNBAFinalAverageDataForQuestionFromMilvusTool, FormatQueryResultsToContextTool_obj: FormatQueryResultsToContextTool):
        super(CustomE2EToolChain,self).__init__(QueryNBAFinalAverageDataForQuestionFromMilvusTool_obj=QueryNBAFinalAverageDataForQuestionFromMilvusTool_obj,FormatQueryResultsToContextTool_obj=FormatQueryResultsToContextTool_obj)
        
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        questions = inputs["questions"]
        query_results = self.QueryNBAFinalAverageDataForQuestionFromMilvusTool_obj.func(questions)
        context = self.FormatQueryResultsToContextTool_obj.func(query_results)
        
        return {"context": context,'question' : questions[0]}

    @property
    def input_keys(self) -> List[str]:
        # 定义输入变量
        return ["questions"]

    @property
    def output_keys(self) -> List[str]:
        # 定义输出变量
        return ["context",'question']


if __name__ == '__main__':
        
    NBAFinalAverageDatasMilvusHandler_obj = NBAFinalAverageDatasMilvusHandler()
    QueryNBAFinalAverageDataForQuestionFromMilvusTool_obj = QueryNBAFinalAverageDataForQuestionFromMilvusTool(NBAFinalAverageDatasMilvusHandler_obj = NBAFinalAverageDatasMilvusHandler_obj)
    FormatQueryResultsToContextTool_obj = FormatQueryResultsToContextTool()
    threadsafe_llama3_infer = ThreadSafeLlama3Infer()
    langchain_llm = CustomLlama3LLM(threadsafe_llama3_infer=threadsafe_llama3_infer)
    
    questions = ['请告诉我2023年NBA总决赛尼古拉-约基奇的场均数据']
    
    customE2EToolChain = CustomE2EToolChain(QueryNBAFinalAverageDataForQuestionFromMilvusTool_obj, FormatQueryResultsToContextTool_obj)

    # result = customE2EToolChain.invoke({'questions' : questions})['context']

    
    # result = e2eChain.run({'questions' : questions})
    # temp = 1
    
    # queryResults = QueryNBAFinalAverageDataForQuestionFromMilvusTool_obj._run(questions)
    
    # instruction = ''
    # inputs = '背景知识：球员: 尼古拉-约基奇 | 场均出场时间: 41.2分钟 | 年龄: 27岁 | 场均得分: 30.2分 | 场均篮板: 14.0个 | 场均助攻: 7.2次 | 场均抢断: 0.8次 | 场均盖帽: 1.4次。问题:2023年NBA总决赛尼古拉-约基奇的场均数据是多少？'
    # prompts = f"{instruction} ; {inputs}"
    # print(langchain_llm(prompts))
    
    prompt_template_str = '根据用户问题和背景知识回答问题 ; {context} ; {question}'
    
    prompt_template = PromptTemplate(
        input_variables=["context", "questions"],  # 输入变量，包括问题和查询结果
        template=prompt_template_str
    )
    
    llm_chain = LLMChain(llm = langchain_llm,prompt=prompt_template)

    e2eChain = SequentialChain(
        chains=[customE2EToolChain, llm_chain],
        input_variables=["questions"],  # 仅声明第一个工具的输入
        output_variables=["text"]  # 仅声明最后一个工具的输出
    )

    result = e2eChain({'questions' : questions})
    
    temp = 1
