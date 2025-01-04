import os 
from langchain_core.tools.simple import Tool
from typing import Callable,Any
from langchain_community.tools import TavilySearchResults
import sys,os

# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)


from agent.agent_3d_client import *


class toolDescriptionFactory:

    '''
        获取 tavilyTool 的description, 用于 Agent 进行解析和决策
    '''
    @classmethod
    def getTavilyToolDescription(self):
        tavilyToolDescription = '''
            当输入的prompts中包含的NBA问题标签为 NBATag_1 时，请使用此工具(即tavilyTool),如果输入的prompts中不包含的NBA问题标签或其值不为 NBATag_1 时,不要使用此工具! 
            参照如下示例:
            prompts: NBA问题标签: "NBATag_1" ; 请告诉我谁是NBA的Goat?  此时你应该调用此工具
            prompts: 请告诉我谁是NBA的Goat?  此时你应该不调用此工具
            prompts: NBA问题标签: "NABTag_2" ; 请告诉我谁是NBA的Goat?  此时你应该不调用此工具
        '''
        return tavilyToolDescription.strip()
    

    '''
        获取 wikiTool 的description, 用于 Agent 进行解析和决策
    '''
    @classmethod
    def getWikiToolDescription(self):
        wikiToolDescription = '''
            当输入的prompts中包含的NBA问题标签为 NBATag_2 时，请使用此工具(wikiTool),如果输入的prompts中不包含的NBA问题标签或其值不为 NBATag_2 时,不要使用此工具! 
            参照如下示例:
            prompts: NBA问题标签: "NBATag_2" ; 请告诉我谁是NBA的Goat?  此时你应该调用此工具
            prompts: 请告诉我谁是NBA的Goat?  此时你应该不调用此工具
            prompts: NBA问题标签: "NABTag_3" ; 请告诉我谁是NBA的Goat?  此时你应该不调用此工具
            同时在调用此工具时请注意,此工具的_run()实际上是调用了wikipediaapi的page(),因此当你在调用其 _run() 时需要给如一个准确的名词,假设你(LLM)现在获取的input是
            "请向我介绍科比-布莱恩特",那么你应该调用wikiTool的 _run() 方法时传入的参数(即queryKeyword)是 "Kobe Bryant",而不是完整的input,而是对应的关键词.而且对于关键词有如下几点需
            要注意: 1. 需要是英文,因为目前wikipediaapi是以en作为language的  2. 你需要仔细理解input,构造出完全正确的关键词.例如科比的英文名就是 Kobe Bryant,你不能向
            queryKeyword 输入 Kobe-Bryant 或 Kobe Bryent等,这些不是完全匹配的关键词都有可能导致你获取不到正确的wiki page.
        '''
        return wikiToolDescription.strip()



class tavilyTool(Tool):

    name: str 
    func: Callable[[] , Any] 
    description: str
    tavilySearchResults: TavilySearchResults

    def __init__(self,name='tavilyTool',description = toolDescriptionFactory.getTavilyToolDescription()):
        tavilySearchResults = myAgentTavilyFactory.getClient()
        super(tavilyTool,self).__init__(name=name,func=self._run,description=description,tavilySearchResults = tavilySearchResults)

    def _run(self,query:str)->str:
        print('invoking tavilyTool...')
        res = self.tavilySearchResults.invoke({'query' : query})
        return res
    
    def _format(self,return_value:Any)->str:
        pass

    async def _arun(self,query:str)->str:
        pass



class wikiTool(Tool):

    name: str 
    func: Callable[[] , Any] 
    description: str
    wikiWrapper: WikiWrapper

    def __init__(self,name='wikiTool',description = toolDescriptionFactory.getWikiToolDescription()):
        wikiWrapper = WikiWrapperFactory.getClient()
        super(wikiTool,self).__init__(name=name,func=self._run,description=description,wikiWrapper = wikiWrapper)

    def _run(self,queryKeyword:str)->str:
        print('invoking wikiTool...')
        res = self.wikiWrapper.extractPageSummary(queryKeyword,truncLen=None)
        return res
    
    def _format(self,return_value:Any)->str:
        pass

    async def _arun(self,query:str)->str:
        pass
    

if __name__ == '__main__':
    wikiTool_obj = wikiTool()
    res = wikiTool_obj._run('Kobe Bryant')
    temp = 1