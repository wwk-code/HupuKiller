from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
import os,sys

# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

from common.myCustomPath import *


def mergeLora(base_model_path,lora_weights_path,merged_model_path):
    print("加载原始模型...")
    model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    print("加载LoRA权重...")
    model = PeftModel.from_pretrained(model, lora_weights_path)

    print("合并LoRA权重到模型...")
    model = model.merge_and_unload()  
    model.save_pretrained(merged_model_path)  
    tokenizer.save_pretrained(merged_model_path) 


# 对NBA总决赛场均数据问答对模式进行Lora SFT得到的权重进行合并并输出为新模型
def mergeLoraNBAFinalAverageDatas():
    mergeLora(base_model_path,nba_final_average_qa_lora_weights_path,nba_final_average_qa_loraMerged_output_path)
    print("合并完成！")


if __name__ == "__main__":
    mergeLoraNBAFinalAverageDatas()  # 对NBA总决赛场均数据问答对模式进行Lora SFT得到的权重进行合并并输出为新模型

    
