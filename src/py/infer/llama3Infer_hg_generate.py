from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaForCausalLM
from peft import PeftModel, PeftConfig
import torch
import os,sys, threading
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
        self.device = torch.device('cuda:0')
        
        # 合并第一次Lora结果后的模型
        loraMergedOutputPat = os.path.join(project_root_dir,'outputs','LoraMerged','NBAFinalAverageDatas')
        # self.model,self.tokenizer = getLoRAMergedModel(loraMergedOutputPat = loraMergedOutputPat)
        # 动态加载为DPOLoRA结果的模型
        # lora_output_path = os.path.join(project_root_dir,'outputs','SFT','LoRAForDPO_NABFinalAverageDatas')
        lora_output_path = os.path.join(project_root_dir,'outputs','DPO','DPO_NABFinalAverageDatas')
        # lora_output_path = '/data/workspace/projects/llamaLearn/LLaMA-Factory/saves/Llama-3-8B-Chinese-Chat/lora/DPOAlign_LoRA_NBAFinalAverageDatas'
        
        self.model,self.tokenizer = self.getDynamicLoRAMergedModel(rawModelPath = loraMergedOutputPat,lora_output_path = lora_output_path,device=self.device)
        # self.model,self.tokenizer = getDynamicLoRAMergedModel(rawModelPath = base_model_path,lora_output_path = '/data/workspace/projects/llamaLearn/LLaMA-Factory/saves/Llama-3-8B-Chinese-Chat/lora/HupuKillerNBAFinalAverageSFT_abstract_and_concise_epoch30',device=self.device)

        self.custom_bos_token = '<自定义开始>'
        self.custom_eos_token = '<自定义结束>'
        self.tokenizer.add_special_tokens({
            "additional_special_tokens": [self.custom_bos_token, self.custom_eos_token]
        })
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.custom_bos_token_id = self.tokenizer.convert_tokens_to_ids(self.custom_bos_token)
        self.custom_eos_token_id = self.tokenizer.convert_tokens_to_ids(self.custom_eos_token)

        self.kwargs = {
            "max_tokens" : 256,
            "do_sample" : True,
            "top_p" : 0.3,
            "temperature" : 0.2,
            "repetition_penalty" : 1.2,  # 惩罚重复
            "bos_token_id" : self.tokenizer.bos_token_id,
            "eos_token_id" : self.tokenizer.eos_token_id
        }

    def getBaseModel(self,device: str = 'cuda:0'):
        return AutoModelForCausalLM.from_pretrained(base_model_path,torch_dtype=torch.float16,trust_remote_code=True).to(device),AutoTokenizer.from_pretrained(base_model_path,trust_remote_code=True)
    
    def getLoRAMergedModel(self,loraMergedOutputPat: str = None,device: str = 'cuda:0'):
        return AutoModelForCausalLM.from_pretrained(loraMergedOutputPat,torch_dtype=torch.float16,trust_remote_code=True).to(device),AutoTokenizer.from_pretrained(loraMergedOutputPat,trust_remote_code=True)
    
    def getDynamicLoRAMergedModel(self,rawModelPath: str = base_model_path,lora_output_path = None,device: str = 'cuda:0'):
        base_model = AutoModelForCausalLM.from_pretrained(rawModelPath, torch_dtype=torch.bfloat16).to(device)
        lora_model = PeftModel.from_pretrained(base_model, lora_output_path, torch_dtype=torch.bfloat16).to(device)
        tokenizer = AutoTokenizer.from_pretrained(lora_output_path)
        tokenizer.pad_token = tokenizer.eos_token  # 设置填充标记
        return lora_model,tokenizer

    def isLoraMerged(self,model):
        for name, _ in model.named_parameters():
            if "lora" in name:  
                return False
        return True

    def generate_response(self,inputs: str, **kwargs):
        tokenizedInputs = self.tokenizer(inputs, return_tensors="pt").to(self.device)

        outputs = self.model.generate(**tokenizedInputs, max_length=kwargs['max_tokens'], do_sample=kwargs['do_sample'], top_p=kwargs['top_p'], temperature=kwargs['temperature'],repetition_penalty=kwargs['repetition_penalty'], bos_token_id=kwargs['bos_token_id'],eos_token_id=kwargs['eos_token_id'])
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=False)

    # 针对原始的用户输入进行推理
    def llama3InferForUserInputs(self,userInputs: str = None):
        # inputs = '背景知识：球员: 尼古拉-约基奇 | 场均出场时间: 41.2分钟 | 年龄: 27岁 | 场均得分: 30.2分 | 场均篮板: 14.0个 | 场均助攻: 7.2次 | 场均抢断: 0.8次 | 场均盖帽: 1.4次。问题:2023年NBA总决赛尼古拉-约基奇的场均数据是多少？'
        instruction = '根据用户问题和背景知识回答问题'
        prompts = f"{instruction} ; {userInputs}"
        response = self.generate_response(prompts,**self.kwargs)
        return response
    
    
    # 针对完整的prompts进行推理
    def llama3InferForPrompts(self,prompts: str = None):
        # inputs = '背景知识：球员: 尼古拉-约基奇 | 场均出场时间: 41.2分钟 | 年龄: 27岁 | 场均得分: 30.2分 | 场均篮板: 14.0个 | 场均助攻: 7.2次 | 场均抢断: 0.8次 | 场均盖帽: 1.4次。问题:2023年NBA总决赛尼古拉-约基奇的场均数据是多少？'
        response = self.generate_response(prompts,**self.kwargs)
        return response

    
    # LoRA-Merged模型推理输出到文件中做检查
    def LoRAMergedLlama3InferCheck(self):
        inputsFilePath = '/data/workspace/projects/HupuKiller/outputs/Check/rawLoraCheckInputs.txt'
        outputsFilePath = '/data/workspace/projects/HupuKiller/outputs/Check/LoraCheckOutputs.txt'
        instruction = '根据用户问题和背景知识回答问题'
        data_num = 60
        testInputs = readFileContent(inputsFilePath)[:data_num]
        testOutuputs = []
        i = 0
        for input in tqdm(testInputs):
            userInputs = f"{instruction} ; {input}"
            response = self.generate_response(userInputs,**self.kwargs)
            # if '抱歉' not in response:
            #     processed_response = response.split(self.custom_bos_token)[1].strip().split(self.custom_eos_token)[0].strip()
            # else:
            #     processed_response = "抱歉,我无法回答你这个问题，我的知识库中没有定位到对应信息"
            processed_start = f'\n####开始_{i}####\n'
            processed_end = f'\n####开始_{i}####\n'
            testOutuputs.append(f"{processed_start}{response}{processed_end}")
            i += 1
        refreashFile(outputsFilePath)
        writeIterableToFile(outputsFilePath,testOutuputs)

# 线程安全的 llama3Infer变量
class ThreadSafeLlama3Infer():
    
    def __init__(self):
        self.llama3_infer = llama3Infer()
        self.lock = threading.Lock()

    def infer(self, prompt: str):
        # 使用锁确保同一时刻只有一个线程能够进行推理
        with self.lock:
            return self.llama3_infer.llama3InferForPrompts(prompt)


if __name__ == '__main__':
    llama3_infer = llama3Infer()
    temp = 1
    # llama3Infer.rawLlama3InferOnLocalBash(inputs)
    # llama3Infer.LoRAMergedLlama3InferCheck()
