import requests,re,os,sys,bs4,random,time
from bs4 import BeautifulSoup
import pandas as pd
from fake_useragent import UserAgent
from itertools import cycle



# my module
from .myLogging import logNBAStatisCrawler


requestTimeout = 10
sleepTime = random.uniform(2,4)
ipProxies = {
    "http://34.87.84.105:80",  # Singapore
    "http://183.240.196.53:33333", # China
    "http://218.75.102.198:8000",  # China
    "http://218.205.43.68:99"  # China
}   
proxy_pool = cycle(ipProxies)
turnOnProxy = False

# Function to get HTML content
def fetch_html(url):
    time.sleep(sleepTime)
    HEADERS = {"User-Agent": UserAgent().random}
    if turnOnProxy: 
        proxy = next(proxy_pool)
        response = requests.get(url, headers=HEADERS,verify=False,proxies={"http":proxy,"https":proxy},timeout=requestTimeout)
    else: 
        response = requests.get(url, headers=HEADERS,verify=False,timeout=requestTimeout)
    response.encoding = 'utf-8'
    response.raise_for_status()
    logNBAStatisCrawler('i','',f"fetch url:{url} 's status code: {response.status_code}")
    return response.text
