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
from distilBert.customDistilBert import CustomDistilBert

def testLlama():
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


# 获取 Bert 类模型的输入数据
def getBertTestDatas():
    dataFilePath = os.path.join(project_root_dir,'assets','distilBert','datas','train.json')
    datas = loadJsonFile(dataFilePath)
    return datas


def testBert():
    model_path = '/data/workspace/projects/HupuKiller/outputs/DistilBert/fineTune'
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)

    import inspect
    print(inspect.signature(model.forward))
    
    datas = getBertTestDatas()

    testText,testLabel = datas[0]['text'],datas[0]['label']
    testInputs = tokenizer(
        testText,                      # 输入文本
        return_tensors="pt",           # 返回 PyTorch 张量
        padding=True,                  # 填充到相同长度
        truncation=True,               # 截断到最大长度
        max_length=256                 # 最大序列长度
    )
    input_ids,attention_mask = testInputs['input_ids'],testInputs['attention_mask']        # shape: (1, max_sequence_length)
    with torch.no_grad():  # 禁用梯度计算
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=-1).item()  # 预测的标签
    
    print(f"输入文本: {testText}")
    print(f"真实标签: {testLabel}")
    print(f"预测标签: {predicted_label}")
    temp = 1
    


def compareBertAndDistilBert():

    # distilBertName = 'distilbert/distilbert-base-uncased'
    distilBertName = 'bardsai/finance-sentiment-zh-fast'  # 一个较为合适的多语言DistilBert-Finetuned 模型,只是tokenizer和词表与google official bert-base-uncase 不同,其它方面都较为合适.且提供了被蒸馏的bert-base模型
    tokenizer1 = AutoTokenizer.from_pretrained(distilBertName)
    model_1 = DistilBertForSequenceClassification.from_pretrained(distilBertName)

    model_path = '/data/workspace/projects/HupuKiller/outputs/DistilBert/fineTune'
    model_2 = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer2 = BertTokenizer.from_pretrained(model_path)

    # 替换 distilbert 的 embedding层
    chinese_bert_word_embeddings = model_2.bert.embeddings.word_embeddings
    new_word_embeddings = nn.Embedding(chinese_bert_word_embeddings.weight.shape[0], chinese_bert_word_embeddings.weight.shape[1],padding_idx=0)
    new_word_embeddings.weight.data = chinese_bert_word_embeddings.weight.data
    model_1.distilbert.embeddings.word_embeddings = new_word_embeddings
    model_1.distilbert.embeddings.position_embeddings.weight.data = model_2.bert.embeddings.position_embeddings.weight.data

    # 复制bert前六层的权重到DistilBert中
    bert_layers = model_2.bert.encoder.layer[:6]
    for i, distilbert_layer in enumerate(model_1.distilbert.transformer.layer):
        distilbert_layer.load_state_dict(bert_layers[i].state_dict())

    print(
        "Word embeddings 复制成功:",
        torch.allclose(
            model_1.distilbert.embeddings.word_embeddings.weight,
            model_2.bert.embeddings.word_embeddings.weight,
        )
    )
    print(
        "Position embeddings 复制成功:",
        torch.allclose(
            model_1.distilbert.embeddings.position_embeddings.weight,
            model_2.bert.embeddings.position_embeddings.weight,
        )
    )

    vocab1 = tokenizer1.get_vocab()
    vocab2 = tokenizer2.get_vocab()
    if vocab1 == vocab2:
        print("词汇表完全相同")
    else:
        print("词汇表不同")
    text = "Hello, how are you?"
    tokens1 = tokenizer1.tokenize(text)
    tokens2 = tokenizer2.tokenize(text)
    if tokens1 == tokens2:
        print("Tokenization 结果完全相同")
    else:
        print("Tokenization 结果不同")
    encoded1 = tokenizer1.encode(text)
    encoded2 = tokenizer2.encode(text)

    if encoded1 == encoded2:
        print("编码结果完全相同")
    else:
        print("编码结果不同")
    decoded1 = tokenizer1.decode(encoded1)
    decoded2 = tokenizer2.decode(encoded2)

    if decoded1 == decoded2:
        print("解码结果完全相同")
    else:
        print("解码结果不同")


def constructDistilBert():
    model_path = '/data/workspace/projects/HupuKiller/outputs/DistilBert/fineTune'
    bert_base = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    customDistilBert = CustomDistilBert(bert_base)
    datas = getBertTestDatas()
    testText,testLabel = datas[0]['text'],datas[0]['label']
    testInputs = tokenizer(
        testText,                      # 输入文本
        return_tensors="pt",           # 返回 PyTorch 张量
        padding=True,                  # 填充到相同长度
        truncation=True,               # 截断到最大长度
        max_length=256                 # 最大序列长度
    )
    input_ids,attention_mask = testInputs['input_ids'],testInputs['attention_mask']        # shape: (1, max_sequence_length)
    with torch.no_grad():  # 禁用梯度计算
        outputs = customDistilBert(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs['logits']

    predicted_label = torch.argmax(logits, dim=-1).item()  # 预测的标签
    print(f'rawLabel: {testLabel}, predictedLabel: {predicted_label}')

    model_output_base_dir = os.path.join(project_root_dir,'assets','distilBert','rawDistilBertModel')
    torch.save(customDistilBert, os.path.join(model_output_base_dir,'rawDistilBertModel.pth'))

    # Check
    # loadModel: torch.nn.Module = torch.load('/data/workspace/projects/HupuKiller/assets/distilBert/rawDistilBertModel/rawDistilBertModel.pth')
    # loadModel.eval()
    # loadModelOutputs = loadModel(input_ids=input_ids, attention_mask=attention_mask)
    # loadModelLogits = loadModelOutputs['logits']
    # temp = 1



def benchmarkDistilBert():
    model_path = '/data/workspace/projects/HupuKiller/outputs/DistilBert/fineTune'
    bert_base = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    customDistilBert = CustomDistilBert(bert_base)
    checkpointFilePath = '/data/workspace/projects/HupuKiller/outputs/DistilBert/distillation/model_epoch_4.pth'
    distilCheckpoints = torch.load(checkpointFilePath)
    customDistilBert.load_state_dict(distilCheckpoints)
    datas = getBertTestDatas()
    dataIdx = 1
    testText,testLabel = datas[dataIdx]['text'],datas[dataIdx]['label']
    testInputs = tokenizer(
        testText,                      # 输入文本
        return_tensors="pt",           # 返回 PyTorch 张量
        padding=True,                  # 填充到相同长度
        truncation=True,               # 截断到最大长度
        max_length=256                 # 最大序列长度
    )
    input_ids,attention_mask = testInputs['input_ids'],testInputs['attention_mask']        # shape: (1, max_sequence_length)
    with torch.no_grad():  # 禁用梯度计算
        outputs = customDistilBert(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs['logits']

    predicted_label = torch.argmax(logits, dim=-1).item()  # 预测的标签
    print(f'rawLabel: {testLabel}, predictedLabel: {predicted_label}')




if __name__ == '__main__':
    # testLlama()
    # testBert()
    # compareBertAndDistilBert()
    # constructDistilBert()
    benchmarkDistilBert()
