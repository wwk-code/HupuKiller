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

class llama3Infer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        #Lora-Merged模型
        self.tokenizer = AutoTokenizer.from_pretrained(nba_final_average_qa_loraMerged_output_path)
        self.model = AutoModelForCausalLM.from_pretrained(nba_final_average_qa_loraMerged_output_path, torch_dtype=torch.bfloat16).to(self.device)  # LoRA合并后的模型

        #原始模型
        # self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        # self.model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.bfloat16).to(self.device)  # 原始模型

        self.custom_bos_token = '<自定义开始>'
        self.custom_eos_token = '<自定义结束>'
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [self.custom_bos_token, self.custom_eos_token]
        })
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.custom_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.custom_bos_token)
        self.custom_eos_token_id = self.tokenizer.convert_tokens_to_ids(self.custom_eos_token)

    def isLoraMerged(self,model):
        for name, _ in model.named_parameters():
            if "lora" in name:  
                return False
        return True

    def generate_response(self,inputs: str):
        tokenizedInputs = self.tokenizer(inputs, return_tensors="pt").to(self.device)
        # 效果还行的参数组合
        kwargs = {
            "max_tokens" : 256,
            "do_sample" : True,
            "top_p" : 0.3,
            "temperature" : 0.2,
            "repetition_penalty" : 1.2,  # 惩罚重复
            "bos_token_id" : self.tokenizer.bos_token_id,
            "eos_token_id" : self.tokenizer.eos_token_id
            # "bos_token_id" : self.custom_bos_token_id,
            # "eos_token_id" : self.custom_eos_token_id
        }
        # kwargs = {
        #     "max_tokens" : 256,
        #     "do_sample" : True,
        #     "top_p" : 0.5,
        #     "temperature" : 0.5,
        #     "repetition_penalty" : 1.2,  # 惩罚重复
        #     "bos_token_id" : self.tokenizer.bos_token_id,
        #     "eos_token_id" : self.tokenizer.eos_token_id
        #     # "bos_token_id" : self.custom_bos_token_id,
        #     # "eos_token_id" : self.custom_eos_token_id
        # }
        outputs = self.model.generate(**tokenizedInputs, max_length=kwargs['max_tokens'], do_sample=kwargs['do_sample'], top_p=kwargs['top_p'], temperature=kwargs['temperature'],repetition_penalty=kwargs['repetition_penalty'], bos_token_id=kwargs['bos_token_id'],eos_token_id=kwargs['eos_token_id'])
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=False)

    # 基座模型的本地推理
    def rawLlama3InferOnLocalBash(self):
        instruction = '根据用户问题和背景知识回答问题'
        inputs = ''
        # inputs = '背景知识：球员: 尼古拉-约基奇 | 场均出场时间: 41.2分钟 | 年龄: 27岁 | 场均得分: 30.2分 | 场均篮板: 14.0个 | 场均助攻: 7.2次 | 场均抢断: 0.8次 | 场均盖帽: 1.4次。问题:2023年NBA总决赛尼古拉-约基奇的场均数据是多少？'
        userInputs = f"{instruction} ; {inputs}"

        while(userInputs != 'exit' or userInputs != 'quit' ):
            os.system('clear')
            userInputs = input('请输入问题：')
            response = self.generate_response(userInputs)
            print(f'response:\n{response}')


    # LoRA-Merged模型的本地推理
    def LoRAMergedLlama3InferOnLocalBash(self):
        instruction = '根据用户问题和背景知识回答问题'
        userInputs = ""

        while(userInputs != 'exit' or userInputs != 'quit' ):
            # os.system('clear')
            userInputs = input('请输入问题：')
            newUserInputs = adaptNBAFinalAverageDatasUserInputsWithDB(userInputs)
            modelInputs = f"{instruction} ; {newUserInputs}"
            response = self.generate_response(modelInputs)
            # print(f'response:\n{response}')
            if '抱歉' not in response:
                processed_response = response.split(self.custom_bos_token)[1].strip().split(self.custom_eos_token)[0].strip()
            else:
                processed_response = "抱歉,我无法回答你这个问题，我的知识库中没有定位到对应信息"
            print(f'{processed_response}')

    # 根据用户输入的问题回答
    def LoRAMergedLlama3InferForQuestion(self,userInputs: str) -> str:
        instruction = '根据用户问题和背景知识回答问题'
        newUserInputs = adaptNBAFinalAverageDatasUserInputsWithDB(userInputs)
        modelInputs = f"{instruction} ; {newUserInputs}"
        response = self.generate_response(modelInputs)

        # if '抱歉' not in response:
        #     processed_response = response.split(self.custom_bos_token)[1].strip().split(self.custom_eos_token)[0].strip()
        # else:
        #     processed_response = "抱歉,我无法回答你这个问题，我的知识库中没有定位到对应信息"
        # return processed_response
        
        return response
    

    # 根据预先制定的完整模型input回答
    def LoRAMergedLlama3InferForInput(self,userInput: str) -> str:

        response = self.generate_response(userInput)

        # if '抱歉' not in response:
        #     processed_response = response.split(self.custom_bos_token)[1].strip().split(self.custom_eos_token)[0].strip()
        # else:
        #     processed_response = "抱歉,我无法回答你这个问题，我的知识库中没有定位到对应信息"
        # return processed_response
        
        return response

    
    # LoRA-Merged模型推理输出到文件中做检查
    def LoRAMergedLlama3InferCheck(self):
        inputsFilePath = '/data/workspace/projects/HupuKiller/outputs/Check/rawLoraCheckInputs.txt'
        outputsFilePath = '/data/workspace/projects/HupuKiller/outputs/Check/LoraCheckOutputs.txt'
        instruction = '根据用户问题和背景知识回答问题'
        testInputs = readFileContent(inputsFilePath)
        testOutuputs = []
        for input in tqdm(testInputs):
            userInputs = f"{instruction} ; {input}"
            response = self.generate_response(userInputs)
            # print(f'response:\n{response}')
            if '抱歉' not in response:
                processed_response = response.split(self.custom_bos_token)[1].strip().split(self.custom_eos_token)[0].strip()
            else:
                processed_response = "抱歉,我无法回答你这个问题，我的知识库中没有定位到对应信息"
            # print(f'processed_response:\n{processed_response}')
            testOutuputs.append(processed_response)
        writeIterableToFile(outputsFilePath,testOutuputs)

if __name__ == '__main__':
    llama3Infer = llama3Infer()
    llama3Infer.rawLlama3InferOnLocalBash()
    # llama3Infer.LoRAMergedLlama3InferCheck()
    # llama3Infer.LoRAMergedLlama3InferCheck()
