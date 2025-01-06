from transformers import pipeline
from transformers import BertForSequenceClassification, BertTokenizer
import os,sys,torch
from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaForCausalLM
from peft import PeftModel, PeftConfig
import torch
import os,sys
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer,AutoTokenizer, AutoModelForSequenceClassification,DistilBertForSequenceClassification,BertModel
from torch import nn 


# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

from common.myFile import * 

# 自定义 DistilBert 类,直接从Fine-tuned Bert模型中提取有效层来构造
class CustomDistilBert(nn.Module):
    def __init__(self, bert_model: BertForSequenceClassification):
        super(CustomDistilBert, self).__init__()
        self.embeddings = bert_model.bert.embeddings
        self.encoder = nn.ModuleList([bert_model.bert.encoder.layer[i] for i in range(6)])  # 只提取fine-tuned bert-base 的前六层encoder
        self.pre_classifier = nn.Linear(bert_model.config.hidden_size, bert_model.config.hidden_size)
        self.act_afeter_pre_classifier = nn.ReLU()
        self.dropout = bert_model.dropout
        self.classifier = bert_model.classifier

        # 加载预训练DistilBert的 pre_classifier 权重
        self._load_pretrained_pre_classifier()

    def _load_pretrained_pre_classifier(self):
        ref_distilbert_model = DistilBertForSequenceClassification.from_pretrained("bardsai/finance-sentiment-zh-fast")  # 一个较为合适的多语言DistilBert-Finetuned 模型,只是tokenizer和词表与google official bert-base-uncase 不同,其它方面都较为合适.且提供了被蒸馏的bert-base模型
        pretrained_pre_classifier_weight = ref_distilbert_model.pre_classifier.weight.data
        pretrained_pre_classifier_bias = ref_distilbert_model.pre_classifier.bias.data
        # 加载到 CustomDistilBert 的 pre_classifier 层
        self.pre_classifier.weight.data = pretrained_pre_classifier_weight
        self.pre_classifier.bias.data = pretrained_pre_classifier_bias


    def forward(self, input_ids:torch.tensor = None, attention_mask:torch.tensor = None, labels:torch.tensor = None):
        # 获取 embedding 输出
        embedding_output = self.embeddings(input_ids=input_ids)
        # 通过前六层 Transformer 编码器
        hidden_states = embedding_output
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        attention_mask = attention_mask.to(hidden_states.dtype)
        for layer in self.encoder:
            hidden_states = layer(hidden_states, attention_mask)[0]
            # hidden_states = layer(hidden_states, attention_mask)
        last_layer_hidden_states = hidden_states  # 目前只让其forward返回第6层的hidden_states,供以在蒸馏训练中和bert-base的第12层hidden_states对齐
        # 取 [CLS] 对应的向量
        cls_output = hidden_states[:, 0, :]
        # 通过 pre_classifier 层
        cls_output = self.pre_classifier(cls_output)
        cls_output = self.act_afeter_pre_classifier(cls_output)  
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)

        return {"logits": logits, 'hidden_states' : last_layer_hidden_states}



