from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset
from trl import DPOTrainer
import os,sys

'''
    "instruction": "根据用户问题和背景知识回答问题",
    "input": "背景知识：球员: 迈克尔-乔丹 | 场均出场时间: 44.0分钟 | 年龄: 27岁 | 场均得分: 31.2分 | 场均篮板: 6.6个 | 场均助攻: 11.4次 | 场均抢断: 2.8次 | 场均盖帽: 1.4次。问题:1991年NBA总决赛迈克尔-乔丹的场均数据是多少？",
    "output_positive": "<自定义开始>球员: 迈克尔-乔丹 | 场均出场时间: 44.0分钟 | 年龄: 27岁 | 场均得分: 31.2分 | 场均篮板: 6.6个 | 场均助攻: 11.4次 | 场均抢断: 2.8次 | 场均盖帽: 1.4次<自定义结束>",
    "output_negative": "乔丹很强，1991年总决赛的场均数据非常优秀。"
'''


# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

from infer.llama3Infer_hg_generate import *
from common.myFile import *
from common.myCustomPath import *


gpu_device_0 = "cuda:0"
gpu_device_1 = "cuda:1"


class DPO:
    def __init__(self):
        self.strategy_model = AutoModelForCausalLM.from_pretrained(nba_final_average_qa_loraMerged_output_path, torch_dtype=torch.bfloat16).to(gpu_device_0)  # LoRA合并后的模型
        self.tokenizer = AutoTokenizer.from_pretrained(nba_final_average_qa_loraMerged_output_path)
        #参考模型，冻结权重
        self.ref_model = AutoModelForCausalLM.from_pretrained(nba_final_average_qa_loraMerged_output_path, device_map=gpu_device_1)
        self.ref_model.eval()
      

# 加载数据集
data_path = "dpo_data.json"
dataset = load_dataset("json", data_files=data_path)

# 预处理数据：Tokenize
def preprocess_data(examples):
    """
    将 instruction + input 拼接，并分别对正样本和负样本进行 tokenization
    """
    inputs = [
        f"{instruction}\n\n{input_text}"
        for instruction, input_text in zip(examples["instruction"], examples["input"])
    ]
    positive_outputs = examples["output_positive"]
    negative_outputs = examples["output_negative"]

    tokenized_inputs = tokenizer(inputs, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    tokenized_pos = tokenizer(positive_outputs, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    tokenized_neg = tokenizer(negative_outputs, padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    return {
        "input_ids": tokenized_inputs["input_ids"],
        "attention_mask": tokenized_inputs["attention_mask"],
        "labels_positive": tokenized_pos["input_ids"],
        "labels_negative": tokenized_neg["input_ids"],
    }

# 数据预处理
tokenized_dataset = dataset.map(preprocess_data, batched=True)



# 配置 DPOTrainer
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    max_length=512,
    logging_dir="./logs",  # 日志目录
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    beta=0.1,  # KL 散度正则化的权重
    output_dir="./dpo_outputs"  # 输出目录
)

# 开始训练
dpo_trainer.train()

# 保存模型
dpo_trainer.save_model(output_dir="./dpo_trained_model")


