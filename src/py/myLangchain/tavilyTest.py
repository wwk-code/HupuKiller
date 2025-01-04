from langchain_community.tools import TavilySearchResults
from tavily import TavilyClient


tool = TavilySearchResults(
    max_results=2,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=False,
    include_domains=[],
    # exclude_domains=[...],
    # name="...",            # overwrite default tool name
    # description="...",     # overwrite default tool description
    # args_schema=...,       # overwrite default args_schema: BaseModel
)

# query = 'NBA中勒布朗-詹姆斯有多少个总冠军'
query = 'How mant championships did Lebron-James win in NBA'
res = tool.invoke({'query' : query})
print(res)

temp = 1