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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def llama3Infer():
    model = AutoModelForCausalLM.from_pretrained(nba_final_average_qa_loraMerged_output_path, torch_dtype=torch.bfloat16).to(device)

    # model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(nba_final_average_qa_loraMerged_output_path)

    # temp = 1

    def generate_response(inputs: str):
        tokenizedInputs = tokenizer(inputs, return_tensors="pt").to(device)
        # print(tokenizer.bos_token_id, tokenizer.eos_token_id)
        # print(tokenizer.convert_tokens_to_ids("<|begin_of_text|>"), tokenizer.convert_tokens_to_ids("<|eot_id|>"))
        kwargs = {
            "max_tokens" : 256,
            "do_sample" : True,
            "top_p" : 0.4,
            "temperature" : 0.4,
            "repetition_penalty" : 1.2,  # 惩罚重复
            "bos_token_id" : tokenizer.bos_token_id,
            "eos_token_id" : tokenizer.eos_token_id
        }
        # outputs = model.generate(**tokenizedInputs, max_length=128, do_sample=True, top_p=0.7, temperature=0.7)
        outputs = model.generate(**tokenizedInputs, max_length=kwargs['max_tokens'], do_sample=kwargs['do_sample'], top_p=kwargs['top_p'], temperature=kwargs['temperature'],repetition_penalty=kwargs['repetition_penalty'], bos_token_id=kwargs['bos_token_id'],eos_token_id=kwargs['eos_token_id'])
        # return tokenizer.decode(outputs[0], skip_special_tokens=True)
        return tokenizer.decode(outputs[0], skip_special_tokens=False)

    userInputs = '背景信息: 球员: 尼古拉-约基奇 | 场均出场时间: 41.2分钟 | 年龄: 28岁 | 场均得分: 30.2分 | 场均篮板: 13.5个 | 场均助攻: 8.1次 | 场均抢断: 0.8次 | 场均盖帽: 0.6次 ; 问题：2023年NBA总决赛尼古拉-约基奇的场均数据是多少？'
    # userInputs = '背景信息: 球员: 尼古拉-约基奇 | 场均出场时间: 41.2分钟 | 年龄: 28岁 | 场均得分: 30.2分 | 场均篮板: 13.5个 | 场均助攻: 8.1次 | 场均抢断: 0.8次 | 场均盖帽: 0.6次 ; 问题：2023年NBA总决赛尼古拉-约基奇的场均得分是多少？'
    # 
    
    # input_data = '背景信息: 球员: 尼古拉-约基奇 | 场均出场时间: 41.2分钟 | 年龄: 28岁 | 场均得分: 30.2分 | 场均篮板: 13.5个 | 场均助攻: 8.1次 | 场均抢断: 0.8次 | 场均盖帽: 0.6次 ; 问题：2023年NBA总决赛尼古拉-约基奇的场均数据是多少？'
    # input_data = '背景信息: 抱歉，我无法在自身记忆数据库和本地知识库中定位到对应信息 ; 问题：2023年NBA总决赛尼古拉-约基奇的场均数据是多少？'
    # input_data = '背景信息: 抱歉，我无法在自身记忆数据库和本地知识库中定位到对应信息 ; 问题：请告诉我尼古拉-约基奇2023年NBA总决赛的场均数据'
    
    # response = generate_response(userInputs)
    # print(response)
    # temp = 1

    while(userInputs != 'exit' or userInputs != 'quit' ):
        # userInputs = input('请输入问题：')
        response = generate_response(userInputs)
        print(response)
        continue
        response = response.split("答案：")[-1].strip()  # 按照自定义展示给用户的回复开始token进行分割
        response = response.split("答案结束")[0].strip()  # 按照自定义展示给用户的回复结束token进行分割
        print(response)
        temp = 1


llama3Infer()
