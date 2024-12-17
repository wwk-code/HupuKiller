from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaForCausalLM
from peft import PeftModel, PeftConfig
import torch
import os,sys
from tqdm import tqdm
from sqlalchemy import create_engine, Table, MetaData, select, and_
from sqlalchemy.orm import sessionmaker
from typing import Union


# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)
sys.path.append(project_root_dir)


from databaseProcessing.vectorDatabaseProcessing import *
from common.myFile import *


def adaptNBAFinalAverageDatasUserInputsWithDB(userInput:str) -> str:
    nbaFinalAverageDatasMilvusHandler = NBAFinalAverageDatasMilvusHandler()
    userInputList = [userInput]
    queryResults = nbaFinalAverageDatasMilvusHandler.queryFromMysqlForQuestions(userInputList,top_k = 1)
    # transformUserInput
    queryResult = queryResults[0][2:]   # (1991, '迈克尔-乔丹', 44.0, 27, 31.2, 6.6, 11.4, 2.8, 1.4)
    json_file_path = '/data/workspace/projects/HupuKiller/assets/sft/SFTQATemplate.json'
    templateItemName = 'nbaFinalAverageData_Concise_Right_QA_template_2'
    template = loadJsonTemplate(json_file_path,templateItemName)['input']
    player,minute,age,point,rebound,assist,steal,block = queryResult
    fillData = {
        'player':player,
        'minute':minute,
        'age':age,
        'point':point,
        'rebound':rebound,
        'assist':assist,
        'steal':steal,
        'block':block,
        'question':userInput
    }
    newUserInputs = template.format(**fillData)
    return newUserInputs
    


def testAdaptNBAFinalAverageDatasUserInputsWithDB():
    userInput = '1991年NBA总决赛迈克尔-乔丹的场均数据是多少？'
    adaptNBAFinalAverageDatasUserInputsWithDB(userInput)

if __name__ == "__main__":
    testAdaptNBAFinalAverageDatasUserInputsWithDB()


