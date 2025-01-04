import json,os,sys,logging,random


# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

from common.myFile import * 

def split_list(lst: list, ratio: float):
    split_index = int(len(lst) * ratio)
    return lst[:split_index], lst[split_index:]


def transTxt2Json():
    base_dir = os.path.join(project_root_dir,'assets','distilBert','datas')
    trainJsonFilePath = os.path.join(project_root_dir,'assets','distilBert','datas','train.json')
    evaluationJsonFilePath = os.path.join(project_root_dir,'assets','distilBert','datas','evaluation.json')

    tag_0_datas = readFileContent(os.path.join(base_dir,'tag0_0.txt'))
    tag_1_datas = readFileContent(os.path.join(base_dir,'tag1_0.txt'))
    tag_2_datas = readFileContent(os.path.join(base_dir,'tag2_0.txt'))
    tag_3_datas = readFileContent(os.path.join(base_dir,'tag3_0.txt'))

    tag_data_list = tag_0_datas + tag_1_datas + tag_2_datas + tag_3_datas

    outputJsonDataItems = []

    for tag_data in tag_data_list:
        if tag_data == '':
            continue
        split_tag_data = tag_data.split(':')
        text , tag = split_tag_data
        text , tag = text.strip() , tag.strip()
        outputJsonDataItems.append({'text' : text,'label' : tag})

    trainRatio = 0.85
    random.shuffle(outputJsonDataItems)
    trainDataItems,evalDataItems = split_list(outputJsonDataItems,trainRatio)
    write_list_of_dicts_to_json(trainJsonFilePath,trainDataItems)
    write_list_of_dicts_to_json(evaluationJsonFilePath,evalDataItems)
    

if __name__ == '__main__':
    transTxt2Json()
