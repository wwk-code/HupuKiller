from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import PeftModel
from datasets import load_dataset
from trl import DPOTrainer,DPOConfig,DPOConfig
import os,sys
import torch; assert torch.cuda.get_device_capability()[0] >= 8, 'Hardware not supported for Flash Attention'
from peft import LoraConfig
import torch
import logging
from transformers import TrainerCallback,EarlyStoppingCallback

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

# from infer.llama3Infer_hg_generate import *
from common.myFile import *
from common.myCustomPath import *


gpu_device_0 = "cuda:0"
gpu_device_1 = "cuda:1"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 屏蔽 GPU 1  

torch.set_num_threads(14) # 最多分配14个线程给此脚本


class DPO_handler:
    def __init__(self):

        # 初始化日志记录器
        log_file_path = os.path.join(project_root_dir, "outputs","DPO", "DPO_NABFinalAverageDatas" , "training.log")

        # 配置 TensorBoard logging
        # self.tb_callback = TensorBoardCallback()

        self.tokenizer = AutoTokenizer.from_pretrained(nba_final_average_qa_loraMerged_output_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        
        # 加载DPO数据集
        
        data_path = os.path.join(project_root_dir,'assets','dpo','dpoDataItems.json')
        self.dataset = load_dataset("json", data_files=data_path)['train'] # 流式加载数据集，减小主机内存消耗
        total_length = len(self.dataset)
        data_num = 20
        # self.dataset = self.dataset.select(range(total_length - data_num, total_length))  # 剪裁训练数据
        # 数据预处理
        self.dataset = self.dataset.map(self.preprocess_data, batched=True,remove_columns=self.dataset.features)
        eval_datasets_weight = 0.25
        self.datasetDict = self.dataset.train_test_split(test_size=eval_datasets_weight)  # eval datasize takes up 1/4
        self.train_dataset,self.test_dataset = self.datasetDict['train'],self.datasetDict['test']
        
        # BitsAndBytesConfig int-4 config
        # bnb_config = None
        # bnb_config = BitsAndBytesConfig(
        #     load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16  # 4bit
        # )

        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,  # 启用 8bit 量化
            llm_int8_threshold=6.0,  # 阈值，用于区分大权重和小权重的处理
            llm_int8_skip_modules=None,  # 跳过量化的模块（如果需要精细控制）
            llm_int8_enable_fp32_cpu_offload=False,  # 是否将 FP32 权重卸载到 CPU
            llm_int8_has_fp16_weight=False,  # 权重是否已经以 FP16 加载
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            nba_final_average_qa_loraMerged_output_path, 
            # base_model_path,  # 原始模型路径 
            use_cache=False,  # Turn on KV-Cache
            attn_implementation="flash_attention_2",  # using Flash-Attention
            device_map=gpu_device_0,
            torch_dtype=torch.bfloat16, 
            quantization_config=bnb_config,  # 不以 8 位精度加载模型
        )
        
        # lora_output_path = os.path.join(project_root_dir,'outputs','SFT','LoRAForDPO_NABFinalAverageDatas')
        # lora_output_path = '/data/workspace/projects/llamaLearn/LLaMA-Factory/saves/Llama-3-8B-Chinese-Chat/lora/DPOAlign_LoRA_NBAFinalAverageDatas'
        # self.training_adapter_name = 'training_adapter'
        # self.reference_adapter_name = 'reference_adapter'
        # self.model.load_adapter(lora_output_path,self.training_adapter_name)
        # self.model.load_adapter(lora_output_path,self.reference_adapter_name)

        self.lora_target_modules = ['q_proj','k_proj','v_proj','gate_proj']
        
        self.peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.05,
            r=16,
            bias="none",
            target_modules=self.lora_target_modules,
            task_type="CAUSAL_LM",
        )

        self.training_args = DPOConfig(
            output_dir=os.path.join(project_root_dir,'outputs','DPO','DPO_NABFinalAverageDatas'),               # directory to save and repository id
            num_train_epochs=15,                   # number of training epochs
            per_device_train_batch_size=2,          # batch size per device during training
            per_device_eval_batch_size=1,           # batch size for evaluation
            gradient_accumulation_steps=16,          # number of steps before performing a backward/update pass
            gradient_checkpointing=True,            # use gradient checkpointing to save memory
            optim="adamw_torch_fused",              # use fused adamw optimizer
            learning_rate=5e-5,                     # 10x higher LR than QLoRA paper
            max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
            warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
            lr_scheduler_type="cosine",             # use cosine learning rate scheduler
            logging_steps=1,                       # log every 25 steps
            save_steps=2,                         # when to save checkpoint
            save_total_limit=5,                     # limit the total amount of checkpoints
            evaluation_strategy="steps",            # evaluate every 1000 steps
            eval_steps=2,                         # when to evaluate
            bf16=True,                              # use bfloat16 precision
            tf32=True,                              # use tf32 precision
            push_to_hub=False,                      # push model to hub
            logging_dir=os.path.join(project_root_dir,'outputs','DPO','DPO_NABFinalAverageDatas'),
            report_to="tensorboard",                # report mestrics to tensorboard
            load_best_model_at_end = True
        )
    
        self.dpo_args = {
            "beta": 0.1,                        # KL-Divergency hyper-param, The beta factor in DPO loss. Higher beta means less divergence
            "loss_type": "sigmoid",              # The loss type for DPO.
            'max_length': 256,
            'max_prompt_length' : 128
        }

        # 早停回调
        self.early_stopping_callback = EarlyStoppingByLossCallback(early_stopping_patience=5, early_stopping_threshold=0.01)

        self.trainer = DPOTrainer(
            model = self.model,
            ref_model = None,
            # model_adapter_name=self.training_adapter_name,
            # ref_adapter_name=self.reference_adapter_name,
            peft_config=self.peft_config,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.test_dataset,
            tokenizer=self.tokenizer,
            max_length=self.dpo_args['max_length'],
            max_prompt_length=self.dpo_args['max_prompt_length'],
            beta=self.dpo_args["beta"],
            loss_type=self.dpo_args["loss_type"],
            callbacks=[self.early_stopping_callback] # 添加早停回调
        )
        

    # 修正 preprocess_data 函数
    def preprocess_data(self, examples):
        """
        将 instruction + input 拼接，并分别对正样本和负样本进行 tokenization
        """
        inputs = [
            f"{instruction}\n\n{input_text}"
            for instruction, input_text in zip(examples["instruction"], examples["input"])
        ]
        positive_outputs = examples["output_positive"]
        negative_outputs = examples["output_negative"]

        return {
            'prompt' : inputs,
            'chosen' : positive_outputs,
            'rejected' : negative_outputs
        }


    def do_train(self):
        self.trainer.train()
        self.trainer.save_model()


class EarlyStoppingByLossCallback(EarlyStoppingCallback):
    def __init__(self, early_stopping_patience=3, early_stopping_threshold=0.05):
        super().__init__(early_stopping_patience=early_stopping_patience)
        self.threshold = early_stopping_threshold
        self.best_loss = float("inf")
        self.patience_counter = 0

    def on_log(self, args, state, control, **kwargs):
        """
        在日志记录时检查训练和验证的 `loss`
        """
        # 获取最新的训练损失
        log_history = state.log_history
        if not log_history:
            return

        # 取最新的 loss 值
        current_loss = None
        if "loss" in log_history[-1]:
            current_loss = log_history[-1]["loss"]
        elif "eval_loss" in log_history[-1]:
            current_loss = log_history[-1]["eval_loss"]

        if current_loss is not None:
            print(f"[EarlyStopping] 当前 Loss: {current_loss}")
            if current_loss < self.best_loss:
                self.best_loss = current_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # 检查是否达到早停条件
            if current_loss < self.threshold or self.patience_counter >= self.early_stopping_patience:
                print(f"[EarlyStopping] 早停触发！Loss: {current_loss}, Best Loss: {self.best_loss}")
                control.should_training_stop = True


if __name__ == '__main__':
    dpo_handler = DPO_handler()
    dpo_handler.do_train()