from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaForCausalLM
from peft import PeftModel, PeftConfig
import torch
import os,sys
from tqdm import tqdm


base_model_path = '/root/.cache/huggingface/hub/models--hfl--llama-3-chinese-8b-instruct/snapshots/06bd938075968adc98bc4080bfcd65a8c2a25250/'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16).to(device)  # 原始模型

kwargs = {
    "max_tokens" : 256,
    "do_sample" : True,
    "top_p" : 0.3,
    "temperature" : 0.2,
    "repetition_penalty" : 1.2,  # 惩罚重复
    "bos_token_id" : tokenizer.bos_token_id,
    "eos_token_id" : tokenizer.eos_token_id
}

inputs = "Who are you?"
tokenizedInputs = tokenizer(inputs, return_tensors="pt").to(device)

response = model.generate(**tokenizedInputs, max_length=kwargs['max_tokens'], do_sample=kwargs['do_sample'], top_p=kwargs['top_p'], temperature=kwargs['temperature'],repetition_penalty=kwargs['repetition_penalty'], bos_token_id=kwargs['bos_token_id'],eos_token_id=kwargs['eos_token_id'])



print(response)

temp = 1
