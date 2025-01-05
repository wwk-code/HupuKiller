from transformers import pipeline
from transformers import BertForSequenceClassification, BertTokenizer
import os,sys,torch

# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

from common.myFile import * 


def getNewTestDatas():
    numOfFile = 4
    baseFilePath = '/data/workspace/projects/HupuKiller/assets/distilBert/datas'
    datas = []
    for i in range(numOfFile):
        filePath = f"{baseFilePath}/tag{i}_1.txt"
        datas += readFileContent(filePath)
    return datas 
        

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
    
# best precision: 0.9994385176866929
# current Test accuracy: 0.9455159112825458
# 2st Test accuracy: 0.8855255916345625

if __name__ == '__main__':
    fineTuneBenchmark()