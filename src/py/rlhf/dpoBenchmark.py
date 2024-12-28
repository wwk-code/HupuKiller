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


from common.myCustomPath import *
from common.myFile import *
from adapter.llamaInfer_db_adapter import *
from infer.llama3Infer_hg_generate import *

if __name__ == '__main__':
    inputsFilePath = '/data/workspace/projects/HupuKiller/assets/dpo/inputs.txt'
    data_num = 3
    # inputs = readFileContent(inputsFilePath)[:data_num]
    inputs = readFileContent(inputsFilePath)
    llama3Infer_obj = llama3Infer()
    QA_For_DPO_Check_Path = '/data/workspace/projects/HupuKiller/outputs/Check/QA_For_DPO_Check.txt'
    outputs = []
    for input in inputs:
        output = llama3Infer_obj.rawLlama3InferOnLocalBash(input)
        outputs.append(output)
        
    refreashFile(QA_For_DPO_Check_Path)
    with open(QA_For_DPO_Check_Path,'w') as f:
        for i in range(len(inputs)):
            input,output = inputs[i],outputs[i]
            f.write('\n#############\n')
            f.write(f"Q:\n{input}\nA:\n{output}")
            f.write('\n#############\n')
    
    
    


