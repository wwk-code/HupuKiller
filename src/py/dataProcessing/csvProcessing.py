import pandas as pd
import os,sys,json

# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)
sys.path.append(project_root_dir)


from common.myFile import *
from databaseProcessing.mysqlProcessing import *


class NBAFinalAverageDataCsvProcessor():

    def __init__(self,skipYears=0,numOfYears=3):
        self.numOfYears = numOfYears
        self.skipYears = skipYears
        pass

    def extractNBAFinalAverageData(self):
        # 获取当前csv表中每个年份主客队球员场均数据子表的起始行索引
        def getEveryYearDataStartIndex(df: pd.DataFrame):
            startIndex = []
            numOfRows = len(df)
            lastIndex = 0
            for i in range(1,numOfRows):
                if df.loc[i][0][:4].isdigit():
                    startIndex.append([lastIndex,i])
                    lastIndex = i
            startIndex.append([lastIndex,numOfRows])
            return startIndex
        
        averageFinalGameDatas = []
        NBAFinalAverageDataCsvPath = os.path.join(project_root_dir,"outputs","NBAFinalsStatisticCrawler","finalAverageDatas.csv")
        df = read_csv_file(NBAFinalAverageDataCsvPath)
        startIndex = getEveryYearDataStartIndex(df)

        for i in range(2 * self.skipYears,2 * (self.skipYears + self.numOfYears),2):
            winnerDatasDict = {}
            winnerDatasDict['title'] = df.loc[startIndex[i][0]][0]  # 标题  （字符串）
            winnerDatasDict['headers'] = df.loc[startIndex[i][0]+1]  # 表头  (一级列表)
            winnerDatasDict['items'] = df.loc[startIndex[i][0]+2:startIndex[i][1] - 1].values.tolist()  # 表项 (二级列表)
            winnerDatasDict['year'] = winnerDatasDict['title'][:4]  

            loserDatasDict = {}
            loserDatasDict['title'] = df.loc[startIndex[i+1][0]][0]  # 标题  （字符串）
            loserDatasDict['headers'] = df.loc[startIndex[i+1][0]+1]  # 表头  (一级列表)
            loserDatasDict['items'] = df.loc[startIndex[i+1][0]+2:startIndex[i+1][1] - 1].values.tolist()  # 表项 (二级列表)
            loserDatasDict['year'] = loserDatasDict['title'][:4]

            averageFinalGameDatas.append({"winner" : winnerDatasDict,"loser" : loserDatasDict})
            
        return averageFinalGameDatas
    


    '''
        根据 assets/sft/SFTQATemplate.json 中的 nbaFinalAverageData_Concise_Right_QA_template_1 和 nbaFinalAverageData_Concise_Error_QA_template_1 构造对应SFT QA对
        构造出的数据形式如下:

    '''
    def constructNbaFinalAverageDataConsiceQATemplate_1(self,datas:dict):
        SFTQATemplateFilePath = os.path.join(project_root_dir,"assets","sft","SFTQATemplate.json")   # project output dir
        jsonContent = loadJsonFile(SFTQATemplateFilePath)
        year = datas['year']
        headers = datas['headers']
        items = datas['items']
        nbaFinalAverageDataQATemplates = []
        rawNbaFinalAverageDataQATemplate = jsonContent['nbaFinalAverageData_Concise_Right_QA_template_2']
        rawNbaFinalAverageDataErrorQATemplate = jsonContent['nbaFinalAverageData_Concise_Error_QA_template_2']

        for item in items:
            player, age, minute, point, rebound, assist, steal, block = item
            questions = [
                f"{year}年NBA总决赛{player}的场均数据是多少？",
                f"请告诉我{year}年NBA总决赛{player}的场均数据",
                f"{year}年NBA总决赛{player}的场均数据如何？"
            ]
            fillData = {
                "question" : "",
                "player" : player,
                "age": age,
                "minute" : minute,
                "point" : point,
                "rebound" : rebound,
                "assist" : assist,
                "steal" : steal,
                "block" : block,
                "question" : ""
            }
            for question in questions:
                fillData["question"] = question
                nbaFinalAverageDataQATemplate = rawNbaFinalAverageDataQATemplate.copy()
                nbaFinalAverageErrorDataQATemplate = rawNbaFinalAverageDataErrorQATemplate.copy()
                nbaFinalAverageErrorDataQATemplate['input'] = nbaFinalAverageErrorDataQATemplate['input'].format(**fillData)
                nbaFinalAverageDataQATemplate['input'] = nbaFinalAverageDataQATemplate['input'].format(**fillData)
                nbaFinalAverageDataQATemplate['output'] = nbaFinalAverageDataQATemplate['output'].format(**fillData)
                nbaFinalAverageDataQATemplates.append(nbaFinalAverageDataQATemplate)
                nbaFinalAverageDataQATemplates.append(nbaFinalAverageErrorDataQATemplate)
                temp = 1


        return nbaFinalAverageDataQATemplates        
    

    def exportAverageFinalGameDatas2SftJson(self,averageFinalGameDatas: list):
        # 输出SFTQA对文件路径
        # outputSFTQAFilePath = os.path.join(project_root_dir,"outputs","SFTDatas","NBAFinalAverageDatasQA.json")
        outputSFTQAFilePath = "/data/workspace/projects/llamaLearn/LLaMA-Factory/data/HupuKiller/NBAFinalAverageDatasQA_concise.json"
        refreashFile(outputSFTQAFilePath)
        for averageFinalGameData in averageFinalGameDatas:
            winnerData = averageFinalGameData['winner']
            loserData = averageFinalGameData['loser']
            nbaFinalWinnerAverageDataQATemplates = self.constructNbaFinalAverageDataConsiceQATemplate_1(winnerData)
            nbaFinalLoserAverageDataQATemplates = self.constructNbaFinalAverageDataConsiceQATemplate_1(loserData)
            append_to_json_file(outputSFTQAFilePath,nbaFinalWinnerAverageDataQATemplates)
            append_to_json_file(outputSFTQAFilePath,nbaFinalLoserAverageDataQATemplates)

    def exportAverageFinalGameDatas2DataBase(self,averageFinalGameDatas: list):
        databaseName = MYSQL_DATABASE_NAME
        databaseTableName = MYSQL_TABLE_NAMES[0]
        dataItemIdx = 0
        mysqlhandler = MySQLDatabaseHandler(username=MYSQL_USERNAME, password=MYSQL_PASSWORD, host="localhost", port=3306, database=MYSQL_DATABASE_NAME)
        Insertdatas = []
        for averageFinalGameData in averageFinalGameDatas:
            winnerData = averageFinalGameData['winner']
            loserData = averageFinalGameData['loser']
            YearDatas = (winnerData, loserData)
            for datas in YearDatas:
                dataItems = datas['items']
                for dataItem in dataItems:
                    data = {
                        "id": dataItemIdx,
                        "year": datas['year'],
                        "player": dataItem[0],
                        "minutes": dataItem[2],
                        "age": dataItem[1],
                        "score": dataItem[3],
                        "rebound": dataItem[4],
                        "assist": dataItem[5],
                        "steal": dataItem[6],
                        "block": dataItem[7]
                    } 
                    Insertdatas.append(data)
                    dataItemIdx += 1

        mysqlhandler.insertDatas(databaseTableName,Insertdatas)
        # mysqlhandler.deleteDatas(databaseTableName, filter_condition=None)
        print('Insert NBAFinalAverageDatas into NBAFinalAverageDatasTable finished!')   
            
        
    # 自 CSV 文件中提取数据并输出至对应的LoRA数据json文件中
    def extractNBAFinalAverageDatasCsvAndExportQA(self):
        averageFinalGameDatas = self.extractNBAFinalAverageData()
        self.exportAverageFinalGameDatas2SftJson(averageFinalGameDatas)
    
    
    # 自 CSV 文件中提取数据并输出至对应的数据库表中
    def extractNBAFinalAverageDatasCsvAndExport2DataBase(self):
        averageFinalGameDatas = self.extractNBAFinalAverageData()
        self.exportAverageFinalGameDatas2DataBase(averageFinalGameDatas)



    '''
        根据 assets/sft/SFTQATemplate.json 中的  FakeNbaFinalAverageData_QA_template_1 构造对应SFT QA对
        构造出的数据形式如下:
        {
            "instruction": "2023年NBA总决赛勒布朗-詹姆斯的场均数据是多少？",
            "input": "根据用户问题和背景知识，回答与 NBA 球员相关的数据。如果背景知识中包含与用户问题匹配的信息，按照以下格式输出：球员: {球员姓名} | {场均出场时间: 数值分钟} | {年龄: 数值岁} | {场均得分: 数值分} | {场均篮板: 数值个} | {场均助攻: 数值次} | {场均抢断: 数值次} | {场均盖帽: 数值次}。如果无法匹配到明确答案，请说明'无法找到相关数据'",
            "output": "无法找到相关数据"
        },
        {
            "instruction": "请提供勒布朗-詹姆斯在2023年NBA总决赛的数据表现",
            "input": "根据用户问题和背景知识，回答与 NBA 球员相关的数据。如果背景知识中包含与用户问题匹配的信息，按照以下格式输出：球员: {球员姓名} | {场均出场时间: 数值分钟} | {年龄: 数值岁} | {场均得分: 数值分} | {场均篮板: 数值个} | {场均助攻: 数值次} | {场均抢断: 数值次} | {场均盖帽: 数值次}。如果无法匹配到明确答案，请说明'无法找到相关数据'",
            "output": "无法找到相关数据"
        },
    '''
    def exportFakeNBAFinalAverageDatasCsvAndExportQA(self):
        FakeSFTQATemplateFilePath = os.path.join(project_root_dir,"assets","sft","SFTQATemplate.json")
        output_file_path = "/data/workspace/projects/llamaLearn/LLaMA-Factory/data/HupuKiller/FakeNBAFinalAverageDatasQA.json"
        fakeInstructionsFilePath = os.path.join(project_root_dir,"assets","sft","FakeNBAFinalAverageInstructions.txt")
        jsonContent = loadJsonFile(FakeSFTQATemplateFilePath)
        rawFakeNbaFinalAverageDataQATemplate = jsonContent['FakeNbaFinalAverageData_QA_template_1']
        fakeNbaFinalAverageDataQATemplates = []
        fakeInstructions = []
        fillData = {"fakeInstruction" : ""}
        with open(fakeInstructionsFilePath,'r',encoding='utf-8') as f:
            for line in f.readlines():
                fakeInstructions.append(line.strip())

        for fakeInstruction in fakeInstructions:
            fakeNbaFinalAverageDataQATemplate = rawFakeNbaFinalAverageDataQATemplate.copy()
            fillData['fakeInstruction'] = fakeInstruction
            fakeNbaFinalAverageDataQATemplate['instruction'] = rawFakeNbaFinalAverageDataQATemplate['instruction'].format(**fillData)
            fakeNbaFinalAverageDataQATemplates.append(fakeNbaFinalAverageDataQATemplate)

        refreashFile(output_file_path)
        append_to_json_file(output_file_path,fakeNbaFinalAverageDataQATemplates)



    '''
        根据 assets/sft/SFTQATemplate.json 中的  NbaFinalAverageData_QA_template_needRAG_1 构造对应SFT QA对
        构造出的数据形式如下:
    '''
    def exportNeedRAGNBAFinalAverageDatasCsvAndExportQA(self):
        NeedRAGSFTQATemplateFilePath = os.path.join(project_root_dir,"assets","sft","SFTQATemplate.json")
        output_file_path = "/data/workspace/projects/llamaLearn/LLaMA-Factory/data/HupuKiller/NeedRAGNBAFinalAverageDatasQA.json"
        needRAGInstructionsFilePath = os.path.join(project_root_dir,"assets","sft","NeedRAGNBAFinalAverageInstructions.txt")
        jsonContent = loadJsonFile(NeedRAGSFTQATemplateFilePath)
        rawNeedRAGNbaFinalAverageDataQATemplate = jsonContent['NeedRAGNbaFinalAverageData_QA_template_1']
        needRAGNbaFinalAverageDataQATemplates = []
        needRAGInstructions = []
        fillData = {"needRAGInstruction" : ""}
        with open(needRAGInstructionsFilePath,'r',encoding='utf-8') as f:
            for line in f.readlines():
                needRAGInstructions.append(line.strip())

        for needRAGInstruction in needRAGInstructions:
            needRAGNbaFinalAverageDataQATemplate = rawNeedRAGNbaFinalAverageDataQATemplate.copy()
            fillData['needRAGInstruction'] = needRAGInstruction
            needRAGNbaFinalAverageDataQATemplate['instruction'] = needRAGNbaFinalAverageDataQATemplate['instruction'].format(**fillData)
            needRAGNbaFinalAverageDataQATemplates.append(needRAGNbaFinalAverageDataQATemplate)

        refreashFile(output_file_path)
        append_to_json_file(output_file_path,needRAGNbaFinalAverageDataQATemplates)


if __name__ == "__main__":

    skipYears = 0
    numOfYears = 34

    # assert (skipYears+numOfYears) == (2023-1990+1)

    nbaFinalAverageDataCsvProcessor = NBAFinalAverageDataCsvProcessor(skipYears = skipYears, numOfYears = numOfYears)
    
    # 从CSV文件中提取数据再构造出LoRA QA对并输出至对应的json文件中
    # nbaFinalAverageDataCsvProcessor.extractNBAFinalAverageDatasCsvAndExportQA()  # 生成正确且不需要RAG问题的QA对
    # nbaFinalAverageDataCsvProcessor.exportFakeNBAFinalAverageDatasCsvAndExportQA()  # 生成错误问题的QA对
    # nbaFinalAverageDataCsvProcessor.exportNeedRAGNBAFinalAverageDatasCsvAndExportQA()  # 生成需要RAG的QA对

    
    # 从CSV文件中提取数据再存入mysql数据库中
    nbaFinalAverageDataCsvProcessor.extractNBAFinalAverageDatasCsvAndExport2DataBase()
    