import math,os,time,psutil,torch,sys,shutil
# from grouped_batch_sampler import GroupedBatchSampler, create_lengths_groups
from torch import nn
from torch.optim import AdamW
from torch.utils.data import BatchSampler, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from utils import logger
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, BertTokenizer,DistilBertConfig,DistilBertForSequenceClassification,DistilBertTokenizer,AutoTokenizer,BertConfig
import torch.nn.functional as F


# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

from common.myFile import * 
from customDistilBert import CustomDistilBert

def getTrainingDatas():
    filePath = os.path.join(project_root_dir,'assets','distilBert','datas','train.json')
    datas = loadJsonFile(filePath)
    texts,labels = [],[]
    for data in datas:
        texts.append(data['text'].strip())
        labels.append(data['label'].strip())
    return texts,labels    
    


class customDatasets(Dataset):
    def __init__(self, trainingDatas: dict[str,list], tokenizer: BertTokenizer, max_length: int = 256):
        """
        初始化数据集类。
        参数:
        - datas: list[str], 数据列表，格式为 ['text_1 : tag_1', 'text_2 : tag_2', ...]
        - tokenizer: BertTokenizer, 用于文本的 tokenization
        - max_length: int, 输入序列的最大长度
        """
        self.texts,self.labels = trainingDatas['rawTexts'],trainingDatas['rawLabels']
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        assert len(self.texts) == len(self.labels) , 'len of training texts not equals to len of training labels!'
        
        return len(self.texts)

    def sanityCheck(self,input_ids,attention_mask):
        assert len(input_ids) == len(attention_mask) , 'len of input_ids not equals to len of attention_mask!'
        assert sum(input_ids != 0) == sum(attention_mask) , 'valid len of input_ids not equals to valid len of attention_mask!'
        
    def __getitem__(self, idx):
        """
        根据索引返回一个数据样本。
        参数:
        - idx: int, 数据索引
        返回:
        - 一个字典，包含以下键值对:
            - input_ids: torch.tensor, tokenized 文本的 ID 序列
            - attention_mask: torch.tensor, attention mask
            - labels: torch.tensor, 标签
        """
        text = self.texts[idx]
        label = int(self.labels[idx])

        # Tokenize 文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,  # 添加 [CLS] 和 [SEP]
            max_length=self.max_length,  # 设置最大长度
            padding='max_length',  # 填充到最大长度
            truncation=True,  # 截断超过最大长度的部分
            return_attention_mask=True,  # 返回 attention_mask
            return_tensors='pt',  # 返回 PyTorch 张量
        )

        input_ids,attention_mask = encoding['input_ids'].flatten(),encoding['attention_mask'].flatten()
        self.sanityCheck(input_ids,attention_mask)
        
        return {
            'input_ids': input_ids.to(device_0),  
            'attention_mask': attention_mask.to(device_0),  
            'labels': torch.tensor(label, dtype=torch.long).to(device_0)  
        }


class Distiller:
    def __init__(
        self, params: dict, dataset: customDatasets,student: nn.Module, teacher: nn.Module
    ):
        logger.info("Initializing Distiller")
        self.params = params
        self.dump_path = params.dump_path
        self.student = student
        self.teacher = teacher
        self.vocab_size = teacher.config.vocab_size
        self.checkpoint_interval = params.checkpoint_interval
        self.max_grad_norm = params.max_grad_norm
        sampler = RandomSampler(dataset)
        sampler = BatchSampler(sampler=sampler, batch_size=params.batch_size, drop_last=False)
        self.dataloader = DataLoader(dataset=dataset, batch_sampler=sampler)
        self.temperature = params.temperature
        assert self.temperature > 0.0
        
        logger.info("Using Sequence-Classification loss for LM step.")
        
        self.log_interval_steps = params.log_interval_steps
        self.training_epoch = params.training_epoch
        self.epoch = 0
        self.n_iter = 0
        self.n_total_iter = 0
        self.n_sequences_epoch = 0
        self.total_loss_epoch = 0
        self.last_loss = 0
        self.last_loss_logits = 0
        self.last_loss_hidden_states = 0

        self.last_log = 0
        
        self.kl_loss_fct = nn.KLDivLoss(reduction="batchmean")  # logits损失函数
        self.mse_loss_fct = nn.MSELoss()   # hidden_state损失函数
        
        self.logits_loss_ratio = params.logits_loss_ratio  # logits-loss权重
        self.hidden_state_loss_ratio = 1 - self.logits_loss_ratio
        
        logger.info("--- Initializing model optimizer")
        self.gradient_accumulation_steps = params.gradient_accumulation_steps
        assert self.gradient_accumulation_steps >= 1
        self.num_steps_epoch = len(self.dataloader)   # 一个训练周期（epoch）中的总步数
        num_train_optimization_steps = (    # 训练过程中总的参数更新次数（用于供scheduler参考以及用于监控）
            int(self.num_steps_epoch / self.gradient_accumulation_steps * params.training_epoch) + 1
        )
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in student.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": params.weight_decay,
            },
            {
                "params": [
                    p for n, p in student.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
            },
        ]
        
        logger.info(
            "------ Number of trainable parameters (student): %i"
            % sum([p.numel() for p in self.student.parameters() if p.requires_grad])
        )
        logger.info("------ Number of parameters (student): %i" % sum([p.numel() for p in self.student.parameters()]))
        self.optimizer = AdamW(    # optimizer
            optimizer_grouped_parameters, lr=params.learning_rate, eps=params.adam_epsilon, betas=(0.9, 0.98)
        )
        warmup_steps = math.ceil(num_train_optimization_steps * params.warmup_prop)   # warm up 
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_train_optimization_steps
        )
        
        logger.info("--- Initializing Tensorboard")
        self.tensorboard = SummaryWriter(log_dir=os.path.join(self.dump_path, "log"))
        self.tensorboard.add_text(tag="config/training", text_string=str(self.params), global_step=0)

    
    def train(self):
        """
            The real training loop.
        """
        logger.info("Starting training")
        self.last_log = time.time()  # 开始记录
        self.student.train()
        self.teacher.eval()
        
        for _ in range(self.training_epoch):
            logger.info(f"--- Starting epoch {self.epoch}/{self.params.training_epoch-1}")
            iter_batch = tqdm(self.dataloader, desc="-Iter")
            for batch in iter_batch:
                token_ids,attn_mask,label = batch['input_ids'], batch['attention_mask'],batch['labels']
                self.step(input_ids=token_ids, attention_mask=attn_mask)
                iter_batch.update()
                iter_batch.set_postfix(
                    {"Last_loss": f"{self.last_loss:.2f}", "Avg_cum_loss": f"{self.total_loss_epoch/self.n_iter:.2f}"}
                )
            iter_batch.close()
            logger.info(f"--- Ending epoch {self.epoch}/{self.training_epoch-1}")
            self.end_epoch()
        
        logger.info("Save very last checkpoint as `pytorch_model.bin`.")
        self.save_checkpoint(checkpoint_name="pytorch_model.bin")
        logger.info("Distillation Training is finished !")
            
            
    def step(self,input_ids:torch.tensor, attention_mask:torch.tensor):
        student_outputs = self.student(input_ids=input_ids, attention_mask=attention_mask)
        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
        s_logits, s_hidden_states = student_outputs["logits"], student_outputs["hidden_states"]
        t_logits, t_hidden_states = teacher_outputs["logits"], teacher_outputs["hidden_states"]
        t_hidden_states = t_hidden_states[0]  # 这里原本t_hidden_states是一个tuple
        assert s_logits.size() == t_logits.size() , "student's and teacher's logits size not match"
        assert s_hidden_states.size() == t_hidden_states.size() , "student's and teacher's hidden_states size not match"
        logits_loss = self.kl_loss_fct(  
            F.log_softmax(s_logits / self.temperature, dim=-1),  
            F.softmax(t_logits / self.temperature, dim=-1)       
        ) * (self.temperature ** 2)
        hidden_states_loss = self.mse_loss_fct(s_hidden_states, t_hidden_states)
        loss = self.logits_loss_ratio * logits_loss + self.hidden_state_loss_ratio * hidden_states_loss
        
        self.total_loss_epoch += loss.item()
        self.last_loss = loss.item()
        self.last_loss_logits = logits_loss.item()
        self.last_loss_hidden_states = hidden_states_loss.item()
        
        self.optimize(loss)
        self.n_sequences_epoch += input_ids.size(0)
        
    
    def optimize(self, loss):
        if (loss != loss).data.any():
            logger.error("NaN detected in loss!")
            exit(1)
        
        if self.gradient_accumulation_steps > 1:
            loss = loss / self.gradient_accumulation_steps
        
        loss.backward()
        self.iter()
        if self.n_iter % self.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
        
    def iter(self):
        """
        Update global counts, write to tensorboard and save checkpoint.
        """
        self.n_iter += 1
        self.n_total_iter += 1

        if self.n_total_iter % self.log_interval_steps == 0:
            self.log_tensorboard()
            self.last_log = time.time()
        if self.n_total_iter % self.checkpoint_interval == 0:
            self.save_checkpoint()
        
    
    
    def log_tensorboard(self):
        """
        Log into tensorboard. 
        """
        for param_name, param in self.student.named_parameters():
            self.tensorboard.add_scalar(
                tag="parameter_mean/" + param_name, scalar_value=param.data.mean(), global_step=self.n_total_iter
            )
            self.tensorboard.add_scalar(
                tag="parameter_std/" + param_name, scalar_value=param.data.std(), global_step=self.n_total_iter
            )
            if param.grad is None:
                continue
            self.tensorboard.add_scalar(
                tag="grad_mean/" + param_name, scalar_value=param.grad.data.mean(), global_step=self.n_total_iter
            )
            self.tensorboard.add_scalar(
                tag="grad_std/" + param_name, scalar_value=param.grad.data.std(), global_step=self.n_total_iter
            )

        self.tensorboard.add_scalar(
            tag="losses/cum_avg_loss_epoch",
            scalar_value=self.total_loss_epoch / self.n_iter,
            global_step=self.n_total_iter,
        )
        self.tensorboard.add_scalar(tag="losses/loss", scalar_value=self.last_loss, global_step=self.n_total_iter)
        self.tensorboard.add_scalar(
            tag="losses/loss_logits", scalar_value=self.last_loss_logits, global_step=self.n_total_iter
        )
        self.tensorboard.add_scalar(
            tag="losses/loss_hidden_states", scalar_value=self.last_loss_hidden_states, global_step=self.n_total_iter
        )
        self.tensorboard.add_scalar(
            tag="learning_rate/lr", scalar_value=self.scheduler.get_lr()[0], global_step=self.n_total_iter
        )
        self.tensorboard.add_scalar(
            tag="global/memory_usage",
            scalar_value=psutil.virtual_memory()._asdict()["used"] / 1_000_000,
            global_step=self.n_total_iter,
        )
    
    
    def end_epoch(self):
        """
            Finally arrived at the end of epoch (full pass on dataset).
            Do some tensorboard logging and checkpoint saving.
        """
        logger.info(f"{self.n_sequences_epoch} sequences have been trained during this epoch.")

        self.save_checkpoint(checkpoint_name=f"model_epoch_{self.epoch}.pth")
        self.tensorboard.add_scalar(
            tag="epoch/loss", scalar_value=self.total_loss_epoch / self.n_iter, global_step=self.epoch
        )

        self.epoch += 1
        self.n_sequences_epoch = 0
        self.n_iter = 0
        self.total_loss_epoch = 0
    
    
    def save_checkpoint(self,checkpoint_name:str = f"checkpoint.pth"):
        """
            Save the current weights state
        """
        mdl_to_save = self.student.module if hasattr(self.student, "module") else self.student
        state_dict = mdl_to_save.state_dict()
        torch.save(state_dict, os.path.join(self.dump_path, checkpoint_name))


class DistillerParams:
    def __init__(self, params_dict: dict):
        for key, value in params_dict.items():
            setattr(self, key, value)


device_0 = torch.tensor('cpu') if not torch.cuda.is_available() else torch.device('cuda:0')


def main():
    distillerParams = {
        'max_length': 256,
        'overwrite_dump_path': True,
        'dump_path': os.path.join(project_root_dir,'outputs','DistilBert','distillation'),
        'batch_size': 1,
        'temperature': 2.0,
        'training_epoch': 5,
        'gradient_accumulation_steps': 8,  
        'weight_decay': 0.0,  
        'adam_epsilon': 1e-6,
        'learning_rate': 5e-4,
        'warmup_prop': 0.05 ,  # Linear warmup proportion.
        'logits_loss_ratio': 0.5,  # logits_loss的权重
        'log_interval_steps': 50,  # Tensorboard logging interval.
        'checkpoint_interval': 100,  # Checkpoint interval.
        'max_grad_norm': 5.0,
        'seed': 56,
    }
    
    distillerParams_obj = DistillerParams(distillerParams)

    rawTexts,rawLabels = getTrainingDatas()
    trainingDatas = {'rawTexts':rawTexts,'rawLabels':rawLabels}
    model_path = os.path.join(project_root_dir,'outputs','DistilBert','fineTune')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    datasets = customDatasets(trainingDatas=trainingDatas,tokenizer=tokenizer,max_length=distillerParams_obj.max_length)
        
    if os.path.exists(distillerParams_obj.dump_path):
        if not distillerParams_obj.overwrite_dump_path:
            raise ValueError(
                f"Serialization dir {distillerParams_obj.dump_path} already exists, but you have not precised wheter to overwrite"
                " itUse `--force` if you want to overwrite it"
            )
        else:
            shutil.rmtree(distillerParams_obj.dump_path)

    os.makedirs(distillerParams_obj.dump_path,exist_ok=True)
    logger.info(f"Experiment will be dumped and logged in {distillerParams_obj.dump_path}")
    with open(os.path.join(distillerParams_obj.dump_path, "distilParameters.json"), "w") as f:
        json.dump(distillerParams, f, indent=4)

    teacher_config = BertConfig.from_pretrained(model_path, output_hidden_states=True)
    teacher = BertForSequenceClassification.from_pretrained(model_path,config=teacher_config).to(device_0)
    rawDistilBertPath = os.path.join(project_root_dir,'assets','distilBert','rawDistilBertModel','rawDistilBertModel.pth')
    student = torch.load(rawDistilBertPath).to(device_0)

    torch.cuda.empty_cache()
    distillert = Distiller(distillerParams_obj,dataset=datasets,student=student,teacher=teacher)
    distillert.train()

    logger.info("Let's go get some drinks.")


if __name__ == '__main__':

    main() 


