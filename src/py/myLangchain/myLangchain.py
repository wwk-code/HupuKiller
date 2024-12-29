from langchain_core.tools.simple import Tool
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
# from langchain.llms.base import LLM
from langchain_core.language_models.llms import LLM
# 自定义LLM类，复用自己的推理脚本
from typing import Any, List, Optional, Dict, Iterator
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
import os,sys
from typing import Callable

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
        super().__init__()
        self.threadsafe_llama3_infer = threadsafe_llama3_infer
        
    
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
    
    def __init__(
                    self,
                    name:str = "QueryNBAFinalAverageDataForQuestionFromMilvus" ,
                    description:str = "Tool for querying NBA final average data For Questions from Milvus.",
                    NBAFinalAverageDatasMilvusHandler_obj: NBAFinalAverageDatasMilvusHandler = None
                ):
        # 初始化 Tool 类所需的参数
        super().__init__(name=name, func=self._run, description=description,NBAFinalAverageDatasMilvusHandler_obj=NBAFinalAverageDatasMilvusHandler_obj)
        # self.NBAFinalAverageDatasMilvusHandler_obj = NBAFinalAverageDatasMilvusHandler_obj
        this = 1
            
    def _run(self, questions:list[str],top_k = None):
        if top_k is None:
            search_results = self.NBAFinalAverageDatasMilvusHandler_obj.queryFromMysqlForQuestions(questions)
        else:
            search_results = self.NBAFinalAverageDatasMilvusHandler_obj.queryFromMysqlForQuestions(questions,top_k)
            
        return search_results

    # @property
    # def NBAFinalAverageDatasMilvusHandler_obj(self) -> NBAFinalAverageDatasMilvusHandler:
    #     return NBAFinalAverageDatasMilvusHandler_obj
    
    def _arun(self, query):
        # 异步查询的实现
        pass
    



if __name__ == '__main__':
    
    # threadsafe_llama3_infer = ThreadSafeLlama3Infer()
    # langchain_llm = CustomLlama3LLM(threadsafe_llama3_infer=threadsafe_llama3_infer)
    # instruction = '根据用户问题和背景知识回答问题'
    # inputs = '背景知识：球员: 尼古拉-约基奇 | 场均出场时间: 41.2分钟 | 年龄: 27岁 | 场均得分: 30.2分 | 场均篮板: 14.0个 | 场均助攻: 7.2次 | 场均抢断: 0.8次 | 场均盖帽: 1.4次。问题:2023年NBA总决赛尼古拉-约基奇的场均数据是多少？'
    # prompts = f"{instruction} ; {inputs}"
    # print(langchain_llm(prompts))
    
    
    NBAFinalAverageDatasMilvusHandler_obj = NBAFinalAverageDatasMilvusHandler()
    QueryNBAFinalAverageDataForQuestionFromMilvusTool_obj = QueryNBAFinalAverageDataForQuestionFromMilvusTool(NBAFinalAverageDatasMilvusHandler_obj = NBAFinalAverageDatasMilvusHandler_obj)
    questions = ['请告诉我2023年NBA总决赛尼古拉-约基奇的场均数据']
    results = QueryNBAFinalAverageDataForQuestionFromMilvusTool_obj._run(questions)
    
    
    temp = 1
