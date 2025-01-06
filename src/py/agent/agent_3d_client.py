from langchain_community.tools import TavilySearchResults
import wikipediaapi
from fake_useragent import UserAgent


# Tavily Factory
class myAgentTavilyFactory:

    @classmethod
    def getClient(self):
        self.tavilySearchResults = TavilySearchResults(
            max_results=1,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=True,
            include_images=False,
            # include_domains=['url','content',''],
            # exclude_domains=[...],
            # name="...",            # overwrite default tool name
        )

        return self.tavilySearchResults





# Wiki Wrapper
class WikiWrapper:
    def __init__(self):
        self.wiki = wikipediaapi.Wikipedia(user_agent = UserAgent().random,language = 'en')  
        # self.wiki = wikipediaapi.Wikipedia(user_agent = UserAgent().random,language = 'zh-cn')  
    
    def extractPage(self,query: str):
        page = self.wiki.page(query)
        if not page.exists():
            print('Warning: wiki search page not exists!')
            page = None
        return page
    def extractPageSummary(self,query: str, truncLen: int = None):
        if truncLen is None:
            pageSummary = self.wiki.page(query).summary
        else:
            pageSummary = self.wiki.page(query).summary[:truncLen]
        return pageSummary
    def extractPageText(self,query: str):
        pageText = self.wiki.page(query).text
        return pageText
    def extractPageSections(self,query: str, truncLen: int = None):
        
        def print_sections(sections, level=0,truncLen: int = None):
            if truncLen is None:
                for s in sections:
                    print("%s: %s - %s" % ("*" * (level + 1), s.title, s.text))
                    print_sections(s.sections, level + 1) 
            else:
                for s in sections:
                    print("%s: %s - %s" % ("*" * (level + 1), s.title, s.text[:truncLen]))
                    print_sections(s.sections, level + 1) 
        pageSections = self.wiki.page(query).sections
        print_sections(pageSections, truncLen=truncLen)
        
        return pageSections


# Wiki Factory
class WikiWrapperFactory:

    @classmethod
    def getClient(self):

        self.wikiWrapper = WikiWrapper()

        return self.wikiWrapper


if __name__ == '__main__':

    wiki = WikiWrapperFactory().getClient()
    # res = wiki.extractPageSummary('Kobe Bryant')
    res = wiki.extractPageSummary('Kobe bryant')
    temp = 1
    # res = wiki.extractPageSummary('科比·布莱恩特')

    # translator = Translator()
    # result = translator.translate(res, dest='zh-cn')

    temp = 1

