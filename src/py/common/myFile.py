import pandas as pd
import os,sys,json
from typing import Union

project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)


def read_csv_file(csv_file_path: str):
    try:
        data = pd.read_csv(csv_file_path)
        return data
    except FileNotFoundError:
        print(f"文件未找到: {csv_file_path}")
        return None
    except pd.errors.EmptyDataError:
        print("文件是空的")
        return None
    except pd.errors.ParserError:
        print("文件解析错误")


def loadJsonFile(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        jsonContent = json.load(f)
    return jsonContent


def loadJsonTemplate(json_file_path: str,templateItemName: str):
    jsonContent = loadJsonFile(json_file_path)
    template = jsonContent[templateItemName]
    return template


def write_list_of_dicts_to_json(file_path,data: list[dict]):
    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)
    print(f"数据成功写入到 {file_path}")




def append_to_json_file(file_path, new_data):
    if not os.path.exists(file_path):
        data = []
    else:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    data.extend(new_data)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def writeIterableToFile(filePath,contents,startDelimeter: str = '',endDelimeter: str = '\n'):
    for content in contents:
        with open(filePath,mode='a',encoding='utf-8') as f:
            f.write(startDelimeter + content + endDelimeter)


def readFileContent(filePath):
    contents = []
    with open(filePath,mode='r',encoding='utf-8') as f:
        for line in f:
            contents.append(line.strip())
    return contents

# 清空文件内容
def refreashFile(filePath):
    with open(filePath,mode='w') as f:
        pass  





