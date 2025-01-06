import os,sys,json


project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

from common.myFile import *

def transLoRAQA_1():
    NBAFinalAverageDatasLoRAQAFilePAth = '/data/workspace/projects/llamaLearn/LLaMA-Factory/data/HupuKiller/NBAFinalAverageDatasQA_abstract.json'
    NBAFinalAverageDatasLoRAQA = loadJsonFile(NBAFinalAverageDatasLoRAQAFilePAth)
    refreashFile(NBAFinalAverageDatasLoRAQAFilePAth)
    for i in range(len(NBAFinalAverageDatasLoRAQA)):
        NBAFinalAverageDatasLoRAQAItem = NBAFinalAverageDatasLoRAQA[i]

        # NBAFinalAverageDatasLoRAQAItem['instruction'] = '根据用户问题和背景知识回答问题'   

        output = NBAFinalAverageDatasLoRAQAItem['output']
        new_output = "<自定义开始>" + output + "<自定义开始结束>"
        # new_output = output
        NBAFinalAverageDatasLoRAQAItem['output'] = new_output
    
    append_to_json_file(NBAFinalAverageDatasLoRAQAFilePAth,NBAFinalAverageDatasLoRAQA)


def transLoRAQA_2():

    pass
    # NBAFinalAverageDatasAbstractLoRAQAFilePAth = '/data/workspace/projects/llamaLearn/LLaMA-Factory/data/HupuKiller/NBAFinalAverageDatasQA_abstract.json'
    # NBAFinalAverageDatasConciseLoRAQAFilePAth = '/data/workspace/projects/llamaLearn/LLaMA-Factory/data/HupuKiller/NBAFinalAverageDatasQA_concise.json'

    # refreashFile(NBAFinalAverageDatasConciseLoRAQAFilePAth)

    # NBAFinalAverageDatasAbstractLoRAQA = loadJsonFile(NBAFinalAverageDatasAbstractLoRAQAFilePAth)
    # NBAFinalAverageDatasConciseLoRAQA = loadJsonFile(NBAFinalAverageDatasConciseLoRAQAFilePAth)
    
    # for i in range(len(NBAFinalAverageDatasConciseLoRAQAFilePAth)):
    #     NBAFinalAverageDatasConciseLoRAQAItem = NBAFinalAverageDatasConciseLoRAQAFilePAth[i]

    #     question = NBAFinalAverageDatasConciseLoRAQAItem['instruction'] 
    #     NBAFinalAverageDatasConciseLoRAQAItem['instruction'] = '根据用户问题和背景知识回答问题'
    #     input = NBAFinalAverageDatasAbstractLoRAQA['input']



    #     # output = NBAFinalAverageDatasLoRAQAItem['output']
    #     # output = output.split('答案:')[1].strip().split('答案结束')[0].strip()
    #     # new_output = output
    #     # NBAFinalAverageDatasLoRAQAItem['output'] = new_output
        
    
    # append_to_json_file(NBAFinalAverageDatasLoRAQAFilePAth,NBAFinalAverageDatasLoRAQA)


if __name__ == '__main__':
    transLoRAQA_1()
    # transLoRAQA_2()



