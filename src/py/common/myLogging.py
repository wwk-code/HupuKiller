import logging,os


# //////////////////////////////////////////////////////////////  Common  //////////////////////////////////////////////////////////////
# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

turnOnNBAStatisCrawler = True


# //////////////////////////////////////////////////////////////  NBAStatisCrawlerLogging  //////////////////////////////////////////////////////////////
logging_output_dir = os.path.join(project_root_dir,"outputs","NBAFinalsStatisticCrawler","loggings")

os.makedirs(logging_output_dir, exist_ok=True)

# NBA历史数据爬虫Logger
NBAStatisCrawlerLogging_file_path = os.path.join(logging_output_dir,"NBAStatisCrawlerLogging.txt")

loggerNBAStatisCrawler = logging.getLogger('NBAStatisCrawlerLogger')
# loggerNBAStatisCrawler.setLevel(logging.INFO)
loggerNBAStatisCrawler.setLevel(logging.WARNING)
NBAStatisCrawlerFormatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

NBAStatisCrawlerFileHandler = logging.FileHandler(NBAStatisCrawlerLogging_file_path,mode='w')
NBAStatisCrawlerFileHandler.setFormatter(NBAStatisCrawlerFormatter)

NBAStatisCrawlerConsoleHandler = logging.StreamHandler()
NBAStatisCrawlerConsoleHandler.setFormatter(NBAStatisCrawlerFormatter)


loggerNBAStatisCrawler.addHandler(NBAStatisCrawlerFileHandler)
loggerNBAStatisCrawler.addHandler(NBAStatisCrawlerConsoleHandler)


def logNBAStatisCrawler(mode: str ='i',preFix: str = '',content:str = ''):

    if not turnOnNBAStatisCrawler:
        return

    if len(content) == 0:
        raise Exception("Invalid Parameters for logNBAStatisCrawler!")
    
    if mode == 'i':
        loggerNBAStatisCrawler.info(f"{preFix}: {content}")
    elif mode == 'w':
        loggerNBAStatisCrawler.warning(f"{preFix}: {content}")
    elif mode == 'e':
        loggerNBAStatisCrawler.error(f"{preFix}: {content}")
    else: 
        raise Exception("Invalid Parameters for logNBAStatisCrawler!")
        


# logger2 = logging.getLogger('database')
# logger2.setLevel(logging.DEBUG)
# handler2 = logging.FileHandler('database.log')
# formatter2 = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
# handler2.setFormatter(formatter2)
# logger2.addHandler(handler2)


