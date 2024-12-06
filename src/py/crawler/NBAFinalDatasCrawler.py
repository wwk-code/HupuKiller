# Common Modules
import requests,re,os,sys
from bs4 import BeautifulSoup
import pandas as pd


# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

# Custom Modules
from common.logging import logNBAStatisCrawler




# Base URL
BASE_URL_Prefix = "https://www.nba-stat.com"
BASE_URL = f"{BASE_URL_Prefix}/playoffs/final.html"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}

# Function to get HTML content
def fetch_html(url):
    response = requests.get(url, headers=HEADERS,verify=False)
    response.raise_for_status()
    response.encoding = 'utf-8'
    return response.text

# Function to parse finals data from a year
def parse_final_data(year_url):
    year_data = []
    html = fetch_html(year_url)
    soup = BeautifulSoup(html, 'html.parser',from_encoding='utf-8')
    
    # Example: Find player stats table
    tables = soup.find_all("table")  # Assuming data is stored in tables

    numOfGames = 0   # num of final games
    for table in tables:
        if not table.get('class') is None and table.get('class')[0] == 'sijie':
            numOfGames += 1
    single_final_games_url = []

    href_links = soup.select('body > main > div.m-content > ul.today_match > li > p.ke_zhu > a')
    href_values = [link['href'] for link in href_links if 'href' in link.attrs]

    # if numOfGames != len(href_values):
        


    # for table in tables:
    #     headers = [th.text.strip() for th in table.find_all("th")]
    #     rows = table.find_all("tr")
    #     for row in rows[1:]:  # Skip header row
    #         values = [td.text.strip() for td in row.find_all("td")]
    #         if values:
    #             year_data.append(dict(zip(headers, values)))

    return year_data

# Main function to scrape all years
def scrape_nba_finals():
    all_data = []
    html = fetch_html(BASE_URL)
    soup = BeautifulSoup(html, 'html.parser')

    # Find links to each year's finals
    year_links = soup.find_all("a", href=True)
    # real_year_links = [
    #     year_link['href'][1:]  # 去掉开头的斜杠
    #     for year_link in year_links
    #     if re.match(r'^/playoffs/\d{4}-finals', year_link['href'])  # 匹配类似 '/playoffs/xxxx-finals'
    # ]
    real_year_links = [
        year_link['href'][1:]  # 去掉开头的斜杠
        for year_link in year_links
        if re.match(r'^/playoffs/(199[0-9]|20[0-1][0-9]|202[0-3])-finals', year_link['href'])
    ]
    for year_url in real_year_links:
        if "finals" in year_url:  # Ensure it's a finals link
            new_year_url = f"{BASE_URL_Prefix}/{year_url}"
            year_data = parse_final_data(new_year_url)
            all_data.extend(year_data)

    # Save data to CSV
    df = pd.DataFrame(all_data)
    # df.to_csv("nba_finals_data.csv", index=False, encoding='utf-8-sig')
    print("Data saved to nba_finals_data.csv")

# Run the script
if __name__ == "__main__":
    
    # scrape_nba_finals()
    
    logNBAStatisCrawler('i','','123')
