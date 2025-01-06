import wikipediaapi
from fake_useragent import UserAgent

# 创建 Wikipedia 对象（选择语言）
# wiki = wikipediaapi.Wikipedia(user_agent = UserAgent().random,language = 'cn')  
wiki = wikipediaapi.Wikipedia(user_agent = UserAgent().random,language = 'en')  


page = wiki.page("詹姆斯")
page = wiki.page("Lebron James")
# page = wiki.page("Python_(programming_language)")


def print_sections(sections, level=0):
        for s in sections:
                print("%s: %s - %s" % ("*" * (level + 1), s.title, s.text))
                print_sections(s.sections, level + 1)
print_sections(page.sections)

# 检查页面是否存在
if page.exists():
    # print(f"Page title: {page.title}")
    # print(f"Page summary: {page.summary[:500]}")  # 摘要（前500字符）
    # print(f"Page full-text: {page.text}")  # 
    temp = 1 
else:
    print("Page does not exist.")