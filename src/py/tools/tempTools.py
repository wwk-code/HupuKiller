import os,sys,json

# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

from common.myFile import *


def extractJson(jsonFilePath,outputFilePath):
    jsonContents = loadJsonFile(jsonFilePath)
    questions = []
    for i in range(len(jsonContents)):
        jsonContent = jsonContents[i]
        input = jsonContent['input']
        question = input.split('问题:')[1]
        questions.append(question)
    
    refreashFile(outputFilePath)
    writeIterableToFile(outputFilePath,questions)
    

if __name__ == '__main__':
    # jsonFilePath = '/data/workspace/projects/llamaLearn/LLaMA-Factory/data/HupuKiller/NBAFinalAverageDatasQA_concise.json'
    jsonFilePath = os.path.join(project_root_dir,'../','llamaLearn','LLaMA-Factory','data','HupuKiller','NBAFinalAverageDatasQA_concise.json')
    # outputFilePath = '/data/workspace/projects/HupuKiller/assets/dpo/questions.txt'
    outputFilePath = os.path.join(project_root_dir,'assets','dpo','questions.txt')
    extractJson(jsonFilePath,outputFilePath)

