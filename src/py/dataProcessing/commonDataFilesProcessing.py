import os,sys
from tqdm import tqdm


# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

from common.myFile import *


def processLoraCheckOutputsFile() -> list:
    filePath = os.path.join(project_root_dir,'outputs','Check','LoraCheckOutputs.txt')
    processed_outputs = []
    with open(filePath,'r',encoding='utf-8') as f:
        lines = f.readlines()
        i = 0
        realContent = None
        isStart = True
        for line in lines:
            if line.startswith(f'####开始_{i}'):
                if isStart:
                    isStart = False
                    realContent = ''
                else:
                    isStart = True
                    i += 1
                    realContent = realContent.strip()
                    processed_outputs.append(realContent)
            else:
                realContent += line

    outputFilePath = os.path.join(project_root_dir,'outputs','Check','preprocessedLoraCheckOutputs.txt')
    # refreashFile(outputFilePath)
    # writeIterableToFile(outputFilePath,processed_outputs)
    return processed_outputs

    
def constructDpoDataItem():
    processed_outputs = processLoraCheckOutputsFile()
    outputFilePath = os.path.join(project_root_dir,'assets','dpo','dpoDataItems.json')
    loraDatasPath = '/data/workspace/projects/HupuKiller/assets/sft/lora/DPOAlign_LoRA_NBAFinalAverageDatas.json'
    loraDatas = loadJsonFile(loraDatasPath)
    data_num = 60
    dpoDataItemTemplate =  {
        "instruction": "根据用户问题和背景知识回答问题",
        "input": "{{input}}",
        "output_positive": "{{output_positive}}",
        "output_negative": "{{output_negative}}"
    }
    dpoDataItems = []
    for i in range(data_num):
        loraData = loraDatas[i]
        processed_output = processed_outputs[i]
        dpoDataItem = dpoDataItemTemplate.copy()
        dpoDataItem['input'] = loraData['input']
        dpoDataItem['output_positive'] = loraData['output']
        dpoDataItem['output_negative'] = processed_output

        dpoDataItems.append(dpoDataItem)
    
    refreashFile(outputFilePath)
    write_list_of_dicts_to_json(outputFilePath,dpoDataItems)


def modifyDpoDataItem():
    sourceFilePath = os.path.join(project_root_dir,'assets','dpo','dpoDataItems_backup.json')
    outputFilePath = os.path.join(project_root_dir,'assets','dpo','dpoDataItems.json')
    dpoDataItems = loadJsonFile(sourceFilePath)
    # newDpoDataItems = [{**item, 'output_negative': item['output_negative'].split('<自定义结束>')[1]} for item in dpoDataItems]
    for dataItem in dpoDataItems:
        input = dataItem['input']
        player = input.split('NBA总决赛')[1].split('的')[0]
        output_negtive = dataItem['output_negative']
        output_negtive = '<自定义开始>球员:' + player + '<自定义结束>'
        dataItem['output_negative'] = output_negtive
    refreashFile(outputFilePath)
    write_list_of_dicts_to_json(outputFilePath,dpoDataItems)

if __name__ == '__main__':
    # constructDpoDataItem()
    modifyDpoDataItem()
