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
        
        

    def exportAverageFinalGameDatas2SftJson(self,averageFinalGameDatas: list):
        
        # 输出SFTQA对文件路径
        outputSFTQAFilePath = os.path.join(project_root_dir,"outputs","SFTDatas","NBAFinalAverageDatasQA.json")

        # 根据 assets/sft/SFTQATemplate.json 中的  nbaFinalAverageData_QA_template_1 构造对应SFT QA对
        def constructNbaFinalAverageDataQATemplate_1(datas:dict):
            SFTQATemplateFilePath = os.path.join(project_root_dir,"assets","sft","SFTQATemplate.json")
            jsonContent = loadJsonFile(SFTQATemplateFilePath)
            year = datas['year']
            headers = datas['headers']
            items = datas['items']
            nbaFinalAverageDataQATemplates = []
            rwaNbaFinalAverageDataQATemplate = jsonContent['nbaFinalAverageData_QA_template_1']
            for item in items:
                nbaFinalAverageDataQATemplate = rwaNbaFinalAverageDataQATemplate.copy()
                datasTemplate = ""
                for i in range(1,len(item)):
                    seperator = ' , ' if i != len(item) - 1 else '.'
                    datasTemplate = datasTemplate + headers[i] + " : " + item[i] + seperator
                fillData = {
                    "player" : item[0],
                    "year" : year,
                    "datas" : datasTemplate
                }
                nbaFinalAverageDataQATemplate['input'] = nbaFinalAverageDataQATemplate['input'].format(**fillData)
                nbaFinalAverageDataQATemplate['output'] = nbaFinalAverageDataQATemplate['output'].format(**fillData)
                nbaFinalAverageDataQATemplates.append(nbaFinalAverageDataQATemplate)

            return nbaFinalAverageDataQATemplates

        refreashFile(outputSFTQAFilePath)

        for averageFinalGameData in averageFinalGameDatas:
            winnerData = averageFinalGameData['winner']
            loserData = averageFinalGameData['loser']

            nbaFinalWinnerAverageDataQATemplates = constructNbaFinalAverageDataQATemplate_1(winnerData)
            nbaFinalLoserAverageDataQATemplates = constructNbaFinalAverageDataQATemplate_1(loserData)

            append_to_json_file(outputSFTQAFilePath,nbaFinalWinnerAverageDataQATemplates)
            append_to_json_file(outputSFTQAFilePath,nbaFinalLoserAverageDataQATemplates)

    def extractNBAFinalAverageDatasCsvAndExportQA(self):
        averageFinalGameDatas = self.extractNBAFinalAverageData()
        self.exportAverageFinalGameDatas2SftJson(averageFinalGameDatas)


if __name__ == "__main__":

    numOfYears = 4

    # Extract NBAFinalAverageData and export to QA json file
    nbaFinalAverageDataCsvProcessor = NBAFinalAverageDataCsvProcessor(numOfYears = numOfYears)
    nbaFinalAverageDataCsvProcessor.extractNBAFinalAverageDatasCsvAndExportQA()
