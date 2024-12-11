import pandas as pd
import os,sys,json

# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)


from common.myFile import *


class NBAFinalAverageDataCsvProcessor():

    def __init__(self,numOfYears=3):
        self.numOfYears = numOfYears
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

        for i in range(0,2 * self.numOfYears,2):
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
        根据 assets/sft/SFTQATemplate.json 中的  nbaFinalAverageData_QA_template_1 构造对应SFT QA对
        构造出的数据形式如下(当前已验证此种数据形式容易导致模型出现幻觉):
    '''
    def constructNbaFinalAverageDataQATemplate_1(self,datas:dict):
        SFTQATemplateFilePath = os.path.join(project_root_dir,"assets","sft","SFTQATemplate.json")   # project output dir
        # SFTQATemplateFilePath = "/data/workspace/projects/llamaLearn/LLaMA-Factory/data/HupuKiller/SFTQATemplate.json"  # llamafaFactory data dir
        jsonContent = loadJsonFile(SFTQATemplateFilePath)
        year = datas['year']
        headers = datas['headers']
        items = datas['items']
        nbaFinalAverageDataQATemplates = []
        rawNbaFinalAverageDataQATemplate = jsonContent['nbaFinalAverageData_QA_template_1']

        for item in items:
            player, yearsOld, playTime, point, playground, assist, steal, block = item
            # 目前构造五条含有不同的instructions的每个球员的总决赛历史数据SFT QA对
            Instructions = [
                f"{player}在{year}年NBA总决赛的场均数据是多少？",
                f"请告诉我{year}年NBA总决赛中{player}的场均数据",
                f"{year}年NBA总决赛，{player}的数据表现怎么样",
                f"在{year}年NBA总决赛中，{player}的平均统计数据是什么",
                f"{player}在{year}年NBA总决赛的表现如何？请给出他的场均数据。"
            ]
            fillData = {
                "instruction" : "",
                "player" : player,
                "yearsOld": yearsOld,
                "playTime" : playTime,
                "point" : point,
                "playground" : playground,
                "assist" : assist,
                "steal" : steal,
                "block" : block
            }
            for Instruction in Instructions:
                nbaFinalAverageDataQATemplate = rawNbaFinalAverageDataQATemplate.copy()
                fillData['instruction'] = Instruction
                nbaFinalAverageDataQATemplate['instruction'] = nbaFinalAverageDataQATemplate['instruction'].format(**fillData)
                nbaFinalAverageDataQATemplate['output'] = nbaFinalAverageDataQATemplate['output'].format(**fillData)

                nbaFinalAverageDataQATemplates.append(nbaFinalAverageDataQATemplate)

        return nbaFinalAverageDataQATemplates        
    

    def exportAverageFinalGameDatas2SftJson(self,averageFinalGameDatas: list):
        # 输出SFTQA对文件路径
        # outputSFTQAFilePath = os.path.join(project_root_dir,"outputs","SFTDatas","NBAFinalAverageDatasQA.json")
        outputSFTQAFilePath = "/data/workspace/projects/llamaLearn/LLaMA-Factory/data/HupuKiller/NBAFinalAverageDatasQA.json"
        refreashFile(outputSFTQAFilePath)
        for averageFinalGameData in averageFinalGameDatas:
            winnerData = averageFinalGameData['winner']
            loserData = averageFinalGameData['loser']
            nbaFinalWinnerAverageDataQATemplates = self.constructNbaFinalAverageDataQATemplate_1(winnerData)
            nbaFinalLoserAverageDataQATemplates = self.constructNbaFinalAverageDataQATemplate_1(loserData)
            append_to_json_file(outputSFTQAFilePath,nbaFinalWinnerAverageDataQATemplates)
            append_to_json_file(outputSFTQAFilePath,nbaFinalLoserAverageDataQATemplates)

    def extractNBAFinalAverageDatasCsvAndExportQA(self):
        averageFinalGameDatas = self.extractNBAFinalAverageData()
        self.exportAverageFinalGameDatas2SftJson(averageFinalGameDatas)


    # 导出回复错误问题的QA对数据
    def exportFakeNBAFinalAverageDatasCsvAndExportQA(self):
        FakeSFTQATemplateFilePath = os.path.join(project_root_dir,"assets","sft","SFTQATemplate.json")
        output_file_path = "/data/workspace/projects/llamaLearn/LLaMA-Factory/data/HupuKiller/FakeNBAFinalAverageDatasQA.json"
        jsonContent = loadJsonFile(FakeSFTQATemplateFilePath)
        rawFakeNbaFinalAverageDataQATemplate = jsonContent['FakeNbaFinalAverageData_QA_template_1']

        
        
        refreashFile(output_file_path)
        


if __name__ == "__main__":

    numOfYears = 4

    # Extract NBAFinalAverageData and export to QA json file
    nbaFinalAverageDataCsvProcessor = NBAFinalAverageDataCsvProcessor(numOfYears = numOfYears)
    nbaFinalAverageDataCsvProcessor.extractNBAFinalAverageDatasCsvAndExportQA()
