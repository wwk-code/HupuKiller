# Common Modules
import requests,re,os,sys,bs4,random,time
from bs4 import BeautifulSoup
import pandas as pd
from fake_useragent import UserAgent
from itertools import cycle



# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

# Custom Modules
from common.myLogging import logNBAStatisCrawler
from common.myNet import fetch_html


# Base URL
BASE_URL_Prefix = "https://www.nba-stat.com"
BASE_URL = f"{BASE_URL_Prefix}/playoffs/final.html"


# 提取单场所有球员的比赛数据
def parseSingleFinalGame(singleGameUrl: str):
    game_data = {}
    game_MetaData = {}
    game_AwayTeamData = {}
    game_HomeTeamData = {}
    
    html = fetch_html(singleGameUrl)
    soup = BeautifulSoup(html, 'html.parser',from_encoding='utf-8')

    # Parse Game's Meta Info
    gameMetaInfo = soup.select_one('div.p-fixed > article.m-main > div.m-content')
    gameTitle = gameMetaInfo.find('h1').getText()
    gameType = gameMetaInfo.select_one('p.match_style').getText()
    gameTime = gameMetaInfo.select_one('p.match_time').getText()
    gameOfficials = gameMetaInfo.select_one('p.match_officials').getText()

    gameTeams = gameMetaInfo.select('p.match_team')
    gameAwayTeam = gameTeams[0].getText()
    gameHomeTeam = gameTeams[1].getText()
    gameInActives = gameMetaInfo.select('p.match_inactive')

    gameAwayInActives = 'None' if len(gameInActives) == 0 else gameInActives[0].getText()
    gameHomeInActives = 'None' if len(gameInActives) == 0 else gameInActives[1].getText()

    game_MetaData['赛事标题'] = gameTitle
    game_MetaData['赛事类型'] = gameType
    game_MetaData['比赛时间'] = gameTime
    game_MetaData['比赛主/客队'] = f"{gameHomeTeam}/{gameAwayTeam}"
    game_MetaData['比赛裁判'] = gameOfficials
    game_MetaData['比赛客队未上场球员'] = gameAwayInActives
    game_MetaData['比赛主队未上场球员'] = gameHomeInActives

    # Parse Game's Team's Basic Data
    gameTeamInfos = soup.select('div.p-fixed > article.m-main > div.basic_advanced_stats.div_mainta')
    
    for i in [0,2]:
        gameTeamData = {}
        gameTeamInfo = gameTeamInfos[i]
        gameTeamInfoTitle = gameTeamInfo.find('h3').getText()
        gameTeamInfoTable = gameTeamInfo.find('table')
        if gameTeamInfoTable == None:
            logNBAStatisCrawler('w',"parseSingleFinalGame",f"get a none table when fetch {gameTime} {gameTeamInfoTitle}")
            if i == 0:
                game_AwayTeamData = None
            else:
                game_HomeTeamData = None
            continue
        gameTeamInfoTableHeaders = [header.get_text(strip=True) for header in gameTeamInfoTable.find_all('th')][2:][:21]
        gameTeamInfoTableHeaders[-1] = '正负值'
        gameTeamInfoTableData = []
        for row in gameTeamInfoTable.find_all('tr')[2:]:  # 跳过表头
            player_name = row.find_all('th')[0].getText()
            if player_name == "替补":
                continue
            cols = row.find_all('td')
            if cols:  # 确保行不为空
                row_list = [col.get_text(strip=True) for col in cols]
                row_list.insert(0,player_name)
                gameTeamInfoTableData.append(row_list)
        gameTeamInfoTableData.pop()
        gameTeamData['数据标题'] = gameTeamInfoTitle
        gameTeamData['数据表头'] = gameTeamInfoTableHeaders
        gameTeamInfoTableData = [[item if item != '' else '0' for item in sublist] for sublist in gameTeamInfoTableData]
        gameTeamData['数据表项'] = gameTeamInfoTableData
        if i == 0:
            game_AwayTeamData = gameTeamData.copy()
        else:
            game_HomeTeamData = gameTeamData.copy()

    game_data['比赛元数据'] = game_MetaData
    game_data['比赛客队球员数据'] = game_AwayTeamData
    game_data['比赛主队球员数据'] = game_HomeTeamData

    return game_data


def parseSingleYearAverageTable(playerAverageTable: bs4.element.Tag):
    singleYearPlayerTablesDict = {}
    tableTitle = playerAverageTable.find('caption').getText().strip()
    gameTeamInfoTableHeaders = [header.get_text(strip=True) for header in playerAverageTable.find_all('th')]
    gameTeamInfoTableHeaders_seg1 = gameTeamInfoTableHeaders[5:7]
    gameTeamInfoTableHeaders_seg2 = gameTeamInfoTableHeaders[27:33]
    newGameTeamInfoTableHeaders = gameTeamInfoTableHeaders_seg1 + gameTeamInfoTableHeaders_seg2
    newGameTeamInfoTableItems = []
    # Item's Name Transition
    newGameTeamInfoTableHeaders[2] = '场均上场时间(分钟)'
    for i in range(3,len(newGameTeamInfoTableHeaders)):
        newGameTeamInfoTableHeaders[i] = '场均' + newGameTeamInfoTableHeaders[i]
    # gameTeamInfoTableItems = playerAverageTable.find_all('tr')[2:]
    gameTeamInfoTableItems = playerAverageTable.find_all('tr')
    for gameTeamInfoTableItem in gameTeamInfoTableItems:
        th = gameTeamInfoTableItem.find('th').getText().strip()
        if not th.isdigit():
            continue
        tds = gameTeamInfoTableItem.select('td') 
        tds_seg1 = tds[0:2]
        tds_seg2 = tds[22:len(tds)]
        newTds = tds_seg1 + tds_seg2
        newGameTeamInfoTableItems.append([newTd.getText().strip() for newTd in newTds])

    singleYearPlayerTablesDict['数据标题'] = tableTitle
    singleYearPlayerTablesDict['数据表头'] = newGameTeamInfoTableHeaders
    singleYearPlayerTablesDict['数据表项'] = newGameTeamInfoTableItems

    return singleYearPlayerTablesDict


# Function to parse finals data from a year
def parse_singleYear_finalGame(year_url):
    year_datas = {}  # xxx年总决赛总体数据字典
    game_datas = []  # 单场比赛数据列表
    html = fetch_html(year_url)
    soup = BeautifulSoup(html, 'html.parser',from_encoding='utf-8')

    year = soup.select('div.m-content > h1.title')[0].getText().strip()[:4]    
    year_datas['年份'] = year
    tables = soup.find_all("table")  # Assuming data is stored in tables

    numOfGames = 0   # num of final games
    for table in tables:
        if not table.get('class') is None and table.get('class')[0] == 'sijie':
            numOfGames += 1

    # 获取某年总决赛数据页面中每个单场总决赛数据子页面url列表
    href_links = soup.select('body > main > div.m-content > ul.today_match > li > p.ke_zhu > a')
    single_finalGame_urls = [f"{BASE_URL_Prefix}/{link['href'][1:]}" for link in href_links if 'href' in link.attrs]
    
    if numOfGames != len(single_finalGame_urls):
        logNBAStatisCrawler('e','',f"num of final games: {numOfGames} != len of single final game urls: {len(single_finalGame_urls)}")

    # 提取当年总决赛的场均球员数据
    for i in [8,9]:
        tableJPath = f"body > main > div > div:nth-child({i}) > table"
        playerAverageTable = soup.select_one(tableJPath)
        playerAverageDataDict = parseSingleYearAverageTable(playerAverageTable)
        if i == 8:
            year_datas['主队球员场均数据'] = (playerAverageDataDict)
        else:
            year_datas['客队球员场均数据'] = (playerAverageDataDict)

    for single_finalGame_url in single_finalGame_urls:
        game_datas.append(parseSingleFinalGame(single_finalGame_url))
    
    return year_datas,game_datas


def saveYearDatas2Csv(year_data: list):
    finalDatas_outputPath_prefix = f"{project_root_dir}/outputs/NBAFinalsStatisticCrawler"
    finalAverageDatas_csv_path = f"{finalDatas_outputPath_prefix}/finalAverageDatas.csv" 
    yearAverageData = [year_data['主队球员场均数据'],year_data['客队球员场均数据']]  
    year = year_data['年份']
    new_yearAverageDatas = []  
    for data_dict in yearAverageData:
        title = year + "NBA总决赛" + data_dict['数据标题']
        headers = data_dict['数据表头']
        items = data_dict['数据表项']
        new_yearAverageDatas.append([title] + [''] * (len(headers) - 1))  # 表标题
        new_yearAverageDatas.append(headers)  # 表头
        for item in items:
            new_yearAverageDatas.append(item)
    if os.path.exists(finalAverageDatas_csv_path):
        open(finalAverageDatas_csv_path, 'w').close()  # 清空文件
    df = pd.DataFrame(new_yearAverageDatas)
    df.to_csv(finalAverageDatas_csv_path, index=False, encoding='utf-8-sig')



def saveGamesDatas2Csv(game_datas: list):
    finalDatas_outputPath_prefix = f"{project_root_dir}/outputs/NBAFinalsStatisticCrawler"
    # for yearGameData in game_datas:
    year = game_datas[0]['比赛元数据']['赛事标题'][:4]
    finalGameDatasCsvDir = ""
    try:
        finalGameDatasCsvDir = f"{finalDatas_outputPath_prefix}/{year}" 
        os.makedirs(finalGameDatasCsvDir,exist_ok=True)
    except Exception as e:
        print(f"创建目录时发生错误: {e}")
    gameIdx = 1
    for game_data in game_datas:
        numOfCols = len(game_data['比赛客队球员数据']['数据表头'])
        # 一个csv文件中存储三个子表，分别是: "比赛元数据"、"比赛客队球员数据"、"比赛主队球员数据"
        finalGameDatasCsvPath = f"{finalGameDatasCsvDir}/finalGame_{gameIdx}.csv"
        if os.path.exists(finalGameDatasCsvPath):
            open(finalGameDatasCsvPath, 'w').close()  # 清空文件
        # 比赛元数据 子表
        gameMetaDict = game_data['比赛元数据']
        df = pd.DataFrame(list(gameMetaDict.items()), columns=['比赛元数据项', '比赛元数据值'])
        additional_cols = numOfCols - len(df.columns)  # 计算需要补齐的列数
        if additional_cols > 0:
            for i in range(additional_cols):
                df[f"c_{i}"] = ""  # 添加空列
        df.to_csv(finalGameDatasCsvPath, index=False, encoding='utf-8-sig')
        # 比赛客队球员数据
        gameAwayTeamSubTableDatas = []
        gameAwayTeamDatas = game_data['比赛客队球员数据']
        if gameAwayTeamDatas != None:
            gameAwayTeamDataTitle = [gameAwayTeamDatas['数据标题']] + [''] * (numOfCols - 1)  # 子表标题
            gameAwayTeamSubTableDatas.append([''] * numOfCols) # 空行
            gameAwayTeamSubTableDatas.append(gameAwayTeamDataTitle)
            gameAwayTeamDataHeader = gameAwayTeamDatas['数据表头']
            gameAwayTeamSubTableDatas.append(gameAwayTeamDataHeader)
            for gameAwayTeamDataItem in gameAwayTeamDatas['数据表项']:
                gameAwayTeamSubTableDatas.append(gameAwayTeamDataItem)  
            df = pd.DataFrame(gameAwayTeamSubTableDatas)
            df.to_csv(finalGameDatasCsvPath, mode='a', header=False, index=False, encoding='utf-8-sig')
        # 比赛主队球员数据
        gameHomeTeamSubTableDatas = []
        gameHomeTeamDatas = game_data['比赛主队球员数据']
        if gameHomeTeamDatas != None:
            gameHomeTeamDataTitle = [gameHomeTeamDatas['数据标题']] + [''] * (numOfCols - 1)  # 子表标题
            gameHomeTeamSubTableDatas.append([''] * numOfCols) # 空行
            gameHomeTeamSubTableDatas.append(gameHomeTeamDataTitle)
            gameHomeTeamDataHeader = gameHomeTeamDatas['数据表头']
            gameHomeTeamSubTableDatas.append(gameHomeTeamDataHeader)
            for gameHomeTeamDataItem in gameHomeTeamDatas['数据表项']:
                gameHomeTeamSubTableDatas.append(gameHomeTeamDataItem)  
            df = pd.DataFrame(gameHomeTeamSubTableDatas)
            df.to_csv(finalGameDatasCsvPath, mode='a', header=False, index=False, encoding='utf-8-sig')
        gameIdx += 1



# Main function to scrape all years
def scrape_nba_finals():
    skip_year = 27  # 要跳过的年份
    fetch_years = 34  # 1990 - 2023 
    # fetch_years = 1  # test 
    html = fetch_html(BASE_URL)
    soup = BeautifulSoup(html, 'html.parser')
    # Find links to each year's finals
    year_links = soup.find_all("a", href=True)
    real_year_links = [
        year_link['href'][1:]  # 去掉开头的斜杠
        for year_link in year_links
        if re.match(r'^/playoffs/(199[0-9]|20[0-1][0-9]|202[0-3])-finals', year_link['href'])
    ]
    for year_url in real_year_links[:fetch_years]:
        if "finals" in year_url:  # Ensure it's a finals link
            if skip_year > 0:
                skip_year -= 1
                continue
            new_year_url = f"{BASE_URL_Prefix}/{year_url}"
            year_datas,game_datas = parse_singleYear_finalGame(new_year_url)
            # saveGamesDatas2Csv(game_datas)
            # saveYearDatas2Csv(year_datas)
            temp = 1
    
    temp = 1

    

# Run the script
if __name__ == "__main__":
    
    scrape_nba_finals()
    
    # logNBAStatisCrawler('i','','123')
