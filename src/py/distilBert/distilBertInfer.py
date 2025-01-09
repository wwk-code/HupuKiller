import os,sys

# 将本地包路径添加到 sys.path 的最前面
local_packages_path = os.path.abspath("/data/workspace/projects/HupuKiller/local_libs/site-packages")
sys.path.insert(0, local_packages_path)

import torch,time
from transformers import pipeline,AutoTokenizer
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, TensorDataset

# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

from common.myFile import * 
from customDistilBert import CustomDistilBert 


def getDataLoaderAndLabels(batchSize:int = 8, tokenizer: AutoTokenizer = None,device:torch.device = 'cuda:0',dataNum:int = None):
    datas = getNewTestDatas()
    if not dataNum is None:
        datas = datas[:dataNum]
    rawTexts,rawLabels = [],[]
    for i in range(len(datas)):
        if ':' not in datas[i]:
            continue
        data = datas[i].split(':')
        rawTexts.append(data[0].strip())
        rawLabels.append(data[1].strip())
    input_ids_arr,attention_mask_arr = [],[]
    testInputs = tokenizer(
        rawTexts,                      # 输入文本
        return_tensors="pt",           # 返回 PyTorch 张量
        padding=True,                  # 填充到相同长度
        truncation=True,               # 截断到最大长度
        max_length=256                 # 最大序列长度
    )
    input_ids,attention_mask = testInputs['input_ids'].to(device),testInputs['attention_mask'].to(device)        
    assert input_ids.size() == attention_mask.size() , "size of input_ids != attention_mask when generating a dataLoader!"
    dataset = TensorDataset(input_ids, attention_mask)
    dataloader = DataLoader(dataset, batch_size=batchSize)
    return  dataloader,rawLabels

def getNewTestDatas():
    numOfFile = 4
    baseFilePath = '/data/workspace/projects/HupuKiller/assets/distilBert/datas'
    datas = []
    for i in range(numOfFile):
        filePath = f"{baseFilePath}/tag{i}_1.txt"
        datas += readFileContent(filePath)
    return datas 
        
# best precision: 0.9994385176866929
# current Test accuracy: 0.9455159112825458
# 2st Test accuracy: 0.8855255916345625
# 检测微调后的模型的推理精度
def fineTuneBenchmark():
    model_path = '/data/workspace/projects/HupuKiller/outputs/DistilBert/fineTune'
    # model_path = '/data/workspace/projects/HupuKiller/outputs/DistilBert/fineTune_bestPerf'
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer,device=torch.device('cuda:0'))

    data_base_dir = os.path.join(project_root_dir,'assets','distilBert','datas')
    # datas = readFileContent(os.path.join(data_base_dir,'testDataItems.txt'))
    datas = getNewTestDatas()
    lenOfTestDatas = len(datas)
    texts,labels = [],[]
    for i in range(lenOfTestDatas):
        if ':' not in datas[i]:
            continue
        splitData = datas[i].split(':')
        text,label = splitData[0].strip(),splitData[1].strip()
        texts.append(text)
        labels.append(label)
    
    res = pipe(texts)
    predictLables = [predictLable['label'] for predictLable in res]
    assert len(predictLables) == len(labels)
    zippedLabelPair =  zip(predictLables,labels)
    accuracy = sum(1 for x,y in zippedLabelPair if x == y) / len(labels)
    print(accuracy)
    misMatchedIndexs = []
    
    zippedLabelPair =  zip(predictLables,labels)
    for i,(x,y) in enumerate(zippedLabelPair):
        if x != y:
            misMatchedIndexs.append(i)
    
    for idx in misMatchedIndexs:
        print(f"idx: {idx} text: {texts[idx]} label: {labels[idx]} predict: {predictLables[idx]}")


# 蒸馏后的精度Benchmark
# best precision: 0.9994385176866929
def benchmarkDistilBert():
    device = torch.device('cuda:0')
    model_path = '/data/workspace/projects/HupuKiller/outputs/DistilBert/fineTune'
    bert_base = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    customDistilBert = CustomDistilBert(bert_base)
    checkpointFilePath = '/data/workspace/projects/HupuKiller/outputs/DistilBert/distillation/model_epoch_4.pth'
    distilCheckpoints = torch.load(checkpointFilePath)
    customDistilBert.load_state_dict(distilCheckpoints).to(device)
    predicted_labels = []
    batch_size = 16
    dataLoader,rawLabels = getDataLoaderAndLabels(batchSize = batch_size,tokenizer=tokenizer)
    start_time = time.time()
    with torch.no_grad():  # 禁用梯度计算
        for batch in dataLoader:
            batch_input_ids, batch_attention_mask = batch
            outputs = customDistilBert(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs['logits']
            batch_predicted_labels = [torch.argmax(logit, dim=-1).item() for logit in logits]  # 预测的标签
            predicted_labels += batch_predicted_labels
    end_time = time.time()
    new_raw_label = list(map(int,rawLabels))
    accuracy = sum(1 for x,y in zip(predicted_labels,new_raw_label) if x == y) / len(new_raw_label)
    print(f"classification accuracy:  {accuracy}")  
    diffIdxs = [idx for idx,(x,y) in enumerate(zip(predicted_labels,new_raw_label)) if x != y]
    for diffIdx in diffIdxs:
        print(f"diffIdx: {diffIdx}  rawLabel:{rawLabels[diffIdx]}  predictLabel: {predicted_labels[diffIdx]}")

    print(f'average_infer_time: {(end_time - start_time) / len(rawLabels)} s')


# 量化后的DistilBert的精度Benchmark
# best precision: 
def benchmarkQuantDistilBert():
    # device = torch.device('cuda:0')
    device = torch.device('cpu')
    model_path = '/data/workspace/projects/HupuKiller/outputs/DistilBert/fineTune'
    # bert_base = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # customDistilBert = CustomDistilBert(bert_base)
    checkpointFilePath = '/data/workspace/projects/HupuKiller/outputs/DistilBert/quant/ptq/full_ptq_checkpoints.pth'
    customDistilBert = torch.load(checkpointFilePath).to(device)
    # distilCheckpoints = torch.load(checkpointFilePath)
    # customDistilBert.load_state_dict(distilCheckpoints)
    predicted_labels = []
    batch_size = 16
    start_time = time.time()
    dataLoader,rawLabels = getDataLoaderAndLabels(batchSize = batch_size,tokenizer=tokenizer,device = device,dataNum = None)
    with torch.no_grad():  # 禁用梯度计算
        for batch in dataLoader:
            batch_input_ids, batch_attention_mask = batch
            batch_attention_mask = batch_attention_mask.float()
            outputs = customDistilBert(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
            logits = outputs['logits']
            batch_predicted_labels = [torch.argmax(logit, dim=-1).item() for logit in logits]  # 预测的标签
            predicted_labels += batch_predicted_labels
    end_time = time.time()
    new_raw_label = list(map(int,rawLabels))
    accuracy = sum(1 for x,y in zip(predicted_labels,new_raw_label) if x == y) / len(new_raw_label)
    print(f"classification accuracy:  {accuracy}")  
    diffIdxs = [idx for idx,(x,y) in enumerate(zip(predicted_labels,new_raw_label)) if x != y]
    for diffIdx in diffIdxs:
        print(f"diffIdx: {diffIdx}  rawLabel:{rawLabels[diffIdx]}  predictLabel: {predicted_labels[diffIdx]}")

    print(f'average_infer_time: {(end_time - start_time) / len(rawLabels)} s')


if __name__ == '__main__':
    # fineTuneBenchmark()
    benchmarkQuantDistilBert()