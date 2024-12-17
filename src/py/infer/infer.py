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


from llama3Infer_hg_generate import *


class LLAMA3Infer():

    def __init__(self):
        self.llama3Infer = llama3Infer()

    def execute_llama3Infer_hg_generate_infer(self):
        # self.llama3Infer.rawLlama3InferOnLocalBash()
        self.llama3Infer.LoRAMergedLlama3InferOnLocalBash()



if __name__ == '__main__':
    llama3Infer = LLAMA3Infer()
    llama3Infer.execute_llama3Infer_hg_generate_infer()

