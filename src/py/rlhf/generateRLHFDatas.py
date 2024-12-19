from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaForCausalLM
from peft import PeftModel, PeftConfig
import torch
import os,sys
from tqdm import tqdm


# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

from infer.llama3Infer_hg_generate import *
from common.myFile import *


def testGenerateDPOQuestions():
    questionsFilePath = os.path.join(project_root_dir,'assets','dpo','questions.txt')
    modleInputsFilePath = os.path.join(project_root_dir,'assets','dpo','inputs.txt')
    questions = readFileContent(questionsFilePath)
    llama3Infer_obj = llama3Infer()
    modelInputs = []
    instruction = '根据用户问题和背景知识回答问题'
        
    for i in tqdm(range(len(questions))):
        question = questions[i]
        if i % 2 == 0:
            newUserInputs = adaptNBAFinalAverageDatasUserInputsWithDB(question)
        else:
            newUserInputs = f'背景知识: 抱歉,我无法回答你这个问题，我的知识库中没有定位到对应信息。问题:{question}' 
        modelInput = f"{instruction} ; {newUserInputs}"
        modelInputs.append(modelInput)
    
    refreashFile(modleInputsFilePath)
    writeIterableToFile(modleInputsFilePath,modelInputs)


def testGenerateDPOResponses():
    inputsFilePath = os.path.join(project_root_dir,'assets','dpo','inputs.txt')
    modleOutputsFilePath = os.path.join(project_root_dir,'assets','dpo','responses.txt')

    inputs = readFileContent(inputsFilePath)[:12]
    llama3Infer_obj = llama3Infer()
    modelResponses = []
    for input in tqdm(inputs):
        modelResponse = llama3Infer_obj.LoRAMergedLlama3InferForInput(input)
        modelResponses.append(modelResponse)
    
    refreashFile(modelResponse)
    writeIterableToFile(modleOutputsFilePath,modelResponses)


if __name__ == '__main__':
    # testGenerateDPODatas()
    # testGenerateDPOQuestions()
    testGenerateDPOResponses()

