import json,os,sys,logging,random


# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

from common.myFile import * 


def transTxt():
    filePath = '/data/workspace/projects/HupuKiller/assets/distilBert/datas/tag3_1.txt'
    outputFilePath = '/data/workspace/projects/HupuKiller/assets/distilBert/datas/new_tag3_1.txt'
    content = readFileContent(filePath)
    splitContent = content[0].split(';')
    splitContent = list(map(str.strip,splitContent))
    writeIterableToFile(outputFilePath,splitContent,mode='a')    


def removeEmptyLine():
    filePath = '/data/workspace/projects/HupuKiller/assets/distilBert/datas/new_tag3_1.txt'
    contents = readFileContent(filePath)
    newContents = []
    for line in contents:
        if len(line.strip()) == 0:
            continue
        newContents.append(line)

    refreashFile(filePath)
    writeIterableToFile(filePath,newContents,mode='a')
    
if __name__ == '__main__':
    # transTxt()
    removeEmptyLine()
