from openai import OpenAI
import json,os,sys,logging


# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

from common.myFile import * 


total_items = 100  # 目标总生成数
num_per_call = 100  # 每次对话目标生成数
num_calls = total_items // num_per_call  # 循环调用对话轮数

# tag_0
# loggingFileName = 'generate_tag0_0.log'
# tag_1
# loggingFileName = 'generate_tag1_0.log'
# tag_2
# loggingFileName = 'generate_tag2_0.log'
# tag_3
# loggingFileName = 'generate_tag3_0.log'
# test data
loggingFileName = 'generate_test.log'

# tag_0
# outputFilePath = os.path.join(project_root_dir, 'assets', 'distilBert','datas', 'tag0_0.txt')
# tag_1
# outputFilePath = os.path.join(project_root_dir, 'assets', 'distilBert','datas', 'tag1_0.txt')
# tag_2
# outputFilePath = os.path.join(project_root_dir, 'assets', 'distilBert','datas', 'tag2_0.txt')
# tag_3
# outputFilePath = os.path.join(project_root_dir, 'assets', 'distilBert','datas', 'tag3_0.txt')
# test data
outputFilePath = os.path.join(project_root_dir, 'assets', 'distilBert','datas', 'testDataItems.txt')

loggingFilePath = os.path.join(project_root_dir,'outputs','DistilBert','logging',loggingFileName)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(loggingFilePath),  # 输出到日志文件
        logging.StreamHandler()  # 输出到终端
    ]
)


client = OpenAI(
    api_key=os.getenv("QWEN_API_KEY"), 
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


base_prompt = '''
    现在我在做一款NBA-Agent应用,在构造针对用户不同类型问题文本多类型任务的微调数据集。你是一个数据生成助手，专门生成对应的中文问题文本和对应的分类标签。
    文本内容可以是关于用户对于NBA-Agent的提问，标签是 0 到 3 之间的数字。 请根据我的命令每次生成n条数据，格式为 '文本: 标签'。多个文本对间用；隔开，
    同时记住，生成的回复中出现的符号都需要是英文符号，而非中文格式符号。同时请你记住我们的对话上下文，我需要你确保生成的数据没有重复的文本，同时你生成的
    数据文本应该尽可能出现NBA字样，且你生成的每条文本字数应该在5-50之间，不要太固定。同时注意,让你生成多少对你就生成多少对,不要擅自增加或减少.
'''

prompt_tag_0 = f'''
    首先请你生成{{num}}条关于'NBA领域主观意识讨论类问题'的文本:标签对,这类问题的标签是0，这类问题属于在虎扑等论坛的帖子上不同用户之间讨论的主观问题，这时你应该生成的内容和形式如下: 
    詹姆斯和乔丹谁是NBA的Gota? : 0 ; NBA中詹姆斯和库里谁更好 ? : 0
    现在请你生成额外的{{num}}条,注意上述例子中有2对示例,但只是例子,你要根据我让你生成的条数来实际生成
'''
prompt_tag_0 = prompt_tag_0.format(num=num_per_call).strip()


prompt_tag_1 = f'''
    首先请你生成{{num}}条关于'NBA领域百科类问题'的文本:标签对,这类问题的标签是1，这类问题属于用户在wikipedia上搜索某个NBA领域的常识性问题，这时你应该生成的内容和形式如下: 
    请向我简要介绍詹姆斯 : 1 ; 请向我简要介绍NBA中2016年的金州勇士队 : 1 
    注意,只能是'请向我介绍某位球员/某个NBA领域专有名词'这种形式的,能直接在wikipedia上搜到的问题.不能是具体的某场比赛,也不能是开放式问题(如:请向我简要介绍NBA历史上的最佳防守阵容).
    同时对于'2020年NBA总冠军是哪支球队' '请告诉我NBA历史上有哪些传奇控卫 ' 这种问题,不能将它们视作可以在wikipedia上搜到的,而应该是属于tag_2,用AI搜索工具进行广泛搜索的.现在
    请你生成额外的{{num}}条.同时注意,你生成的内容中的球员名字应该尽可能全,例如不要说乔丹,而应该是迈克尔乔丹.注意上述例子中有2对示例,但只是例子,你要根据我让你生成的条数来实际生成
'''
prompt_tag_1 = prompt_tag_1.format(num=num_per_call).strip()


prompt_tag_2 = f'''
    首先请你生成{{num}}条关于'NBA领域较为细化的问题'的文本:标签对,这类问题的标签是2，这类问题属于用户调用AI搜索工具进行搜索某个NBA领域的较为细节的问题，这时你应该生成的内容和形式如下: 
    请告诉我2016年库里的MVP赛季表现 : 2 ; 请告诉我近期NBA常规赛程 : 2; 请告诉我2016年NBA总决赛勇士和骑士两队球员的场均数据 : 2 ; 詹姆斯的第一个MVP赛季数据是怎样的? ; 请向我介绍2016年NBA总决赛勇士队球员的阵容 : 2
    注意,请发挥的你想象,这类问题的种类非常非常多,不仅仅局限于上述例子中类型,同时注意与其它标签对应的问题类型区分开.tag1:'请向我介绍xx(球员);请向我简要介绍NBA中1995年的休斯顿火箭队'这种用户直接在wikipedia上可以
    搜到的.也不应该是tag0:'NBA的Gota是谁' 这种没有明确答案的主观开放探讨类问题.同时注意,你生成的内容中的球员名字应该尽可能全,例如不要说乔丹,而应该是迈克尔乔丹.注意上述例子中有2对示例,但只是例子,你要根据我让你生成的条数来实际生成
'''
prompt_tag_2 = prompt_tag_2.format(num=num_per_call).strip()


prompt_tag_3 = f'''
    首先请你生成{{num}}条关于'引导NBA-Agent输出NBA领域图片'的文本:标签对,这类问题的标签是3，这类问题属于用户引导NBA-Agent发送图像类NBA领域的回复(例如某位球员的图片,或动图GIF等),，这时你应该生成的内容和形式如下: 
    来点詹姆斯的图片 : 3 ; 斯蒂芬库里长什么样? : 3 ; 来点阴阳怪气表的表情包 : 3 ; 来个詹姆斯笑的图片 : 3
    注意,请发挥的你想象,这类问题的种类非常非常多,不仅仅局限于上述例子中类型,但注意一定要是能体现出引导NBA-Agent发NBA相关的图片的文本,文本中至少需要包含有图片、GIF、动图、表情包这种类型的一个字样.同时注意,多回顾上下文,不要生成重复的文本内容,
    你生成的内容中的球员名字应该尽可能全,例如不要说乔丹,而应该是迈克尔乔丹.注意上述例子中有2对示例,但只是例子,你要根据我让你生成的条数来实际生成
'''
prompt_tag_3 = prompt_tag_3.format(num=num_per_call).strip()


# 初始化对话历史
messages = [
    {'role': 'system', 'content': base_prompt}
]


def generate_data(tag_prompt):
    
    messages.append({'role': 'user', 'content': tag_prompt})
    
    completion = client.chat.completions.create(
        model="qwen-turbo", 
        messages=messages,
        max_tokens=8192,  
        temperature=0.9,  # 控制生成多样性
    )
    
    # 获取生成的文本
    response = completion.choices[0].message.content
    
    # 移除当前的 user-tag_prompt，避免下一轮对话重复回答
    messages.pop(-1)  
    
    # 将生成的文本添加到对话历史中，避免生成重复的数据样本
    # messages.append({'role': 'assistant', 'content': f"you have generate above contents: {response}"})
    
    usage = completion.usage
    total_tokens = usage.total_tokens
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    
    logging.info(f"总 Token 消耗: {total_tokens}")
    logging.info(f"输入 Token: {prompt_tokens}")
    logging.info(f"输出 Token: {completion_tokens}")
    
    dataIems = list(map(str.strip,response.split(';')))
    
    return dataIems



for i in range(num_calls):
    logging.info(f"######## QWEN_Turbo LLM API: 第 {i + 1} 次生成数据 ########")
    # tag_0
    # data_items = generate_data(tag_prompt=prompt_tag_0)
    # tag_1
    # data_items = generate_data(tag_prompt=prompt_tag_1)
    # tag_2
    # data_items = generate_data(tag_prompt=prompt_tag_2)
    # tag_3
    # data_items = generate_data(tag_prompt=prompt_tag_3)

    # test data
    data_items = generate_data(tag_prompt=prompt_tag_3)
    writeIterableToFile(outputFilePath,data_items,mode='a')
    