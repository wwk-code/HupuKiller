import os,sys,json,torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset, Dataset, DatasetDict
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

from infer.llama3Infer_hg_generate import *
from common.myFile import *
from common.myCustomPath import *



class LoRA_handler:
    def __init__(self):
        self.device_0 = "cuda:0"
        self.device_1 = "cuda:1"
        self.RAW_MODEL_PATH = nba_final_average_qa_loraMerged_output_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.RAW_MODEL_PATH)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        self.MAX_SEQ_LENGTH = 256
        self.model = AutoModelForCausalLM.from_pretrained(self.RAW_MODEL_PATH, device_map=self.device_0, torch_dtype=torch.bfloat16)
        
        self.LORA_OUTPUT_DIR = os.path.join(project_root_dir,'outputs','SFT','LoRAForDPO_NABFinalAverageDatas') 

        self.lora_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=16, # LoRA Rank
            bias="none",
            # target_modules="all-linear",
            target_modules=self.lora_target_modules,
            task_type="CAUSAL_LM",
        )
        
        # 设置训练参数
        self.training_args = TrainingArguments(
            output_dir=self.LORA_OUTPUT_DIR,
            overwrite_output_dir=True,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=500,
            save_steps=500,
            logging_steps=50,
            learning_rate=2e-4,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=16,
            num_train_epochs=10,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            fp16=False,
            bf16=True,
            save_total_limit=2,
            load_best_model_at_end=True,
            report_to="none",  # 禁用默认的wandb日志
            # ddp_find_unused_parameters=False if torch.cuda.device_count() > 1 else None,
            ddp_find_unused_parameters = None
        )

        loraDataPath = os.path.join(project_root_dir,'assets','sft','lora','DPOAlign_LoRA_NBAFinalAverageDatas.json')
        self.raw_dataset = self.prepare_data(loraDataPath)
        self.tokenized_dataset = self.tokenize_data(self.raw_dataset, self.tokenizer)

        eval_dataset_ratio = 0.2
        datasets = self.tokenized_dataset.train_test_split(test_size=eval_dataset_ratio)
        self.tokenized_train_dataset = datasets['train']
        self.tokenized_eval_dataset = datasets['test']

        # 初始化 Trainer
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.tokenized_train_dataset,
            eval_dataset=self.tokenized_eval_dataset,
            tokenizer=self.tokenizer,
        )

        # 给模型添加LoRA层
        self.model = get_peft_model(self.model, self.lora_config)


    def e2eLoRATrain(self):

        self.trainer.train()

        # 保存最终模型和适配器
        self.model.save_pretrained(self.LORA_OUTPUT_DIR)
        self.tokenizer.save_pretrained(self.LORA_OUTPUT_DIR)
        print("Training complete. Model and tokenizer saved to:", self.LORA_OUTPUT_DIR)
        

if __name__ == "__main__":
    
    lora_handler = LoRA_handler() 
    lora_handler.e2eLoRATrain()
