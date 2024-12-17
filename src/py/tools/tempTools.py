import os,sys,json

# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

from common.myFile import *


def extractJson(jsonFilePath,outputFilePath):
    jsonContents = loadJsonFile(jsonFilePath)
    inputs = []
    for i in range(len(jsonContents)):
        jsonContent = jsonContents[i]
        input = jsonContent['input']
        inputs.append(input)
    
    refreashFile(outputFilePath)
    writeIterableToFile(outputFilePath,inputs)
    
    
    


if __name__ == '__main__':
    jsonFilePath = '/data/workspace/projects/llamaLearn/LLaMA-Factory/data/HupuKiller/NBAFinalAverageDatasQA_concise.json'
    outputFilePath = '/data/workspace/projects/HupuKiller/outputs/Check/rawLoraCheckInputs.txt'
    extractJson(jsonFilePath,outputFilePath)

