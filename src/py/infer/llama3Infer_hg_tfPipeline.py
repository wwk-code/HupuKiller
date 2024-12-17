from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import os,sys

project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

from common.myCustomPath import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained(nba_final_average_qa_loraMerged_output_path)
model = AutoModelForCausalLM.from_pretrained(nba_final_average_qa_loraMerged_output_path)

loRaMergedPipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)

instruction = '根据用户问题和背景知识回答问题'
inputs = '背景知识：球员: 尼古拉-约基奇 | 场均出场时间: 41.2分钟 | 年龄: 27岁 | 场均得分: 30.2分 | 场均篮板: 14.0个 | 场均助攻: 7.2次 | 场均抢断: 0.8次 | 场均盖帽: 1.4次。问题:2023年NBA总决赛尼古拉-约基奇的场均数据是多少？'
userInputs = f"{instruction} ; {inputs}"

kwargs = {
    "max_tokens" : 256,
    "do_sample" : True,
    "top_p" : 0.3,
    "temperature" : 0.2,
    "repetition_penalty" : 1.2,  # 惩罚重复
    "bos_token_id" : tokenizer.bos_token_id,
    "eos_token_id" : tokenizer.eos_token_id
}

output = loRaMergedPipeline(
    inputs,
    max_length=kwargs['max_tokens'], 
    do_sample=kwargs['do_sample'],  
    top_p=kwargs['top_p'],       
    temperature=kwargs['temperature'], 
    repetition_penalty=kwargs['repetition_penalty'], 
    bos_token_id=kwargs['bos_token_id'],
    eos_token_id=kwargs['eos_token_id']
)


print(output[0]["generated_text"])
temp = 1