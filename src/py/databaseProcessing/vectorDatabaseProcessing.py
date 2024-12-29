from sentence_transformers import SentenceTransformer
import faiss,os,sys,random
import numpy as np
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility


# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)

from .mysqlProcessing import *
from common.myCustomPath import *
from common.myFile import *



# Global Variable 
mysqlNBAFinalAverageDatasHandler = None
sentenceTransformer = None

# Faiss 无法直接存储元数据信息,只专注于做高效的向量检索,适合结合Redis使用
class FaissHandler():

    def __init__(self):
        # 初始化向量模型和数据库
        self.model = SentenceTransformer(sentence_transformer_path) if sentenceTransformer is None else sentenceTransformer
        self.dimension = 384  # 目前选用的 Embbeding 模型的向量维度为 384
        self.base_index = faiss.IndexFlatL2(self.dimension)  
        self.faiss_index = faiss.IndexIDMap(self.base_index)  # 封装为支持 ID 的IndexIDMap索引类型
        self.metadata_store = {}  # 用于存储元数据，例如 {向量ID: (数据库名, 表名, 数据ID)}
        self.vectorId = 0
        self.queryNum = None  # 默认查询所有数据



class NBAFinalAverageDatasFaissHandler(FaissHandler):


    def __init__(self):
        super().__init__()
        self.mysqlNBAFinalAverageDatasHandler = MysqlNBAFinalAverageDatasHandler()
        self.queryNum = 33   
        self.questionTemplates = [
            f'{{year}}年NBA总决赛{{player}}的场均数据是多少？',
            f'请告诉我{{year}}年NBA总决赛{{player}}的场均数据',
            f'{{year}}年NBA总决赛{{player}}的场均数据如何？'
        ]

    def constructFaissIndex(self):  
        # 在myslq数据库对应的表中查询NBA历年总决赛各球员场均数据
        queryDatas = self.mysqlNBAFinalAverageDatasHandler.exportNBAFinalAverageDatasDicts(self.queryNum)
        for queryData in queryDatas:
            question = f"{queryData['year']}年NBA总决赛{queryData['player']}的场均数据是多少？"
            questionEmbedding = self.model.encode(question)


class MilvusHandler:
    def __init__(self, host="localhost", port="19530"):
        self.host = host
        self.port = port
        self.connect()  # 连接 milvus 服务
        

    def connect(self):
        # print("""连接到 Milvus 服务""")
        connections.connect("default", host=self.host, port=self.port)

    """创建集合（表）"""
    def create_collection(self, collection_name, fields: list[FieldSchema], description=""):
        
        if utility.has_collection(collection_name):
            # print(f"Collection '{collection_name}' already exists.")
            if not self.is_collection_loaded(collection_name):
                Collection(name=collection_name).load()
            return Collection(name=collection_name)
        schema = CollectionSchema(fields, description=description)
        collection = Collection(name=collection_name, schema=schema)
        self.my_create_index(collection)
        print(f"Collection '{collection_name}' created.")
        collection.load()
        return collection

    # 为collection中的字段field_name创建索引
    def my_create_index(self,collection, field_name="vector", index_type="IVF_FLAT", metric_type="L2"):
        index_params = {
            "index_type": index_type,
            "metric_type": metric_type,
            "params": {"nlist": 128}  # nlist 决定索引分桶数量，根据数据量设置
        }
        collection.create_index(field_name, index_params)
        print(f"Index created for field '{field_name}' with params: {index_params}")
    

    def insert_datas(self, collection: Collection, embeddings: list, metadatas: list):
        assert len(embeddings) == len(metadatas), "Embeddings and metadatas length must match"
        vector_ids = [None] * len(embeddings)  # 自动生成 ID
        embeddings_list = [embedding.tolist() for embedding in embeddings]
        data = [
            # vector_ids,   # ID 列
            embeddings_list,   # 向量列
            metadatas,    # 元数据列
        ]
        insert_result = collection.insert(data)
        print(f"Inserted {len(insert_result.primary_keys)} records into collection '{collection.name}'")

        collection.load()  # 更新内存上的milvus对应collection快照


    # 根据嵌入向量查找结果,注意 search 和 query不同，search返回 SearchResult类，query返回list
    def search_datas(self, collection:Collection, query_vectors:list[list[float]], top_k: int,output_fields: list[str] = ["dbId"]):
        """检索向量"""
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=query_vectors,
            anns_field="vector",
            param=search_params,
            limit=top_k,
            output_fields=output_fields,
        )
        return results
    
    # query返回list
    def query_by_conditions(self, collection: Collection, conditionsExpr: str, output_fields: list[str] = ["dbId"]):
        """
            条件查询数据
            :param collection: Collection 对象
            :param conditionsExpr: 查询条件，形如 "id in [454647945772706224, 454647945772706241]"
            :param output_fields: 需要返回的字段列表
            :return: 查询结果
        """
        results = collection.query(expr=conditionsExpr, output_fields=output_fields)
        return results
    
    
    def query_all_data(self, collection: Collection, output_fields: list[str] = ["dbId"], limit: int = 10000):
        """
            获取集合中的所有数据
            :param collection: Collection 对象
            :param output_fields: 需要返回的字段列表
            :param limit: 最大返回记录数
            :return: 数据列表
        """
        alwaysTrueExpr = 'id >= 0'  #恒真条件表达式
        results = collection.query(expr=alwaysTrueExpr, output_fields = ["dbId"], limit=limit)
        print(f"Retrieved {len(results)} records from collection '{collection.name}'.")
        return results


    # results 的类型应该是 pymilvus.orm.search.SearchResults
    def extract_db_ids_from_Result(self,results):
        db_ids = []
        for hits in results:  # 每个 `hits` 是一个查询向量的匹配结果
            for hit in hits:  # 每个 `hit` 是一个匹配的向量
                db_id = hit.entity.get("dbId")  # 提取字段 "dbId"
                db_ids.append(db_id)
        return db_ids
    
    # 按条件删除表中数据
    def delete_by_conditions(self, collection: Collection, conditionExpr: str):
        """
            条件删除数据
            :param collection: Collection 对象
            :param conditionExpr: 删除条件，形如 "id in [454647945772706224, 454647945772706241]"
            :return: 无
        """
        collection.delete(expr=conditionExpr)

    # 删除表中的所有数据
    def delete_all_datas(self, collection: Collection):
        results = self.query_all_data(collection)
        ids = [dataItem['id'] for dataItem in results]
        collection.delete(expr=f'id in {ids}')
    
    
    # 根据给定的 collection_name 删除对应 collection
    def my_drop_collection(self, collection_name: str):
        """删除集合"""
        if utility.has_collection(collection_name):
            Collection(collection_name).release()
            utility.drop_collection(collection_name)
            print(f"Collection '{collection_name}' dropped.")

    # 判断名为collection_name的collection是否已经load到了内存中
    def is_collection_loaded(self,collection_name: str):
        try:
            # 查询集合的段信息
            segments = utility.get_query_segment_info(collection_name)
            # 如果存在 segments 信息，则表示集合已加载到内存
            if segments:
                return True
            else:
                return False
        except Exception as e:
            # 如果抛出异常，说明集合未加载
            print(f"Error: {e}")
            print(f"Collection '{collection_name}' is not loaded.")
            return False


class NBAFinalAverageDatasMilvusHandler():

    def __init__(self, host="localhost", port="19530"):
        self.host = host
        self.port = port
        self.milvusHandler = MilvusHandler(host,port)
        self.vectorizeModel = SentenceTransformer(sentence_transformer_path) if sentenceTransformer is None else sentenceTransformer
        self.dim = 384
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
            FieldSchema(name="dbId", dtype=DataType.INT64)
        ]
        self.collection_name = "test_collection"
        self.collection = self.milvusHandler.create_collection(self.collection_name, fields)
        self.mysqlNBAFinalAverageDatasHandler =  \
            MysqlNBAFinalAverageDatasHandler(os.getenv('HupuKiller_MYSQL_USERNAME'),os.getenv('HupuKiller_MYSQL_PASSWORD') , self.host, 3306, MYSQL_DATABASE_NAME) if mysqlNBAFinalAverageDatasHandler is None else mysqlNBAFinalAverageDatasHandler


    # 从mysql数据库中查询对应数据并构造出对应 Quesion-DtaItem 对,向量化问题后插入到 Milvus 中
    def extractMysqlNBAFinalAverageDatasAndExport2Milvus(self):
        # 在myslq数据库对应的表中查询NBA历年总决赛各球员场均数据
        # queryMysqlNum = 10
        queryMysqlNum = None # 查询全部
        queryDatas = self.mysqlNBAFinalAverageDatasHandler.exportNBAFinalAverageDatasDicts(queryMysqlNum)
        questionEmbeddings,dbIds = [],[]
        questions = []
        output_fields = ["dbId"]
        questionTemplates = [
            f'{{year}}年NBA总决赛{{player}}的场均数据是多少？',
            # f'请告诉我{{year}}年NBA总决赛{{player}}的场均数据',
            # f'{{year}}年NBA总决赛{{player}}的场均数据如何？'
        ]
        for queryData in queryDatas:
            for questionTemplate in questionTemplates:
                fillDatas = {
                    'year' : queryData['year'],
                    'player' : queryData['player']
                }
                question = questionTemplate.format(**fillDatas)
                questions.append(question)
                questionEmbedding = self.vectorizeModel.encode(question)
                questionEmbeddings.append(questionEmbedding)
                dbId = int(queryData['id'])
                dbIds.append(dbId)
        
        self.milvusHandler.insert_datas(self.collection, questionEmbeddings, dbIds)

        # self.milvusHandler.query_all_data(self.collection)  # 查询当前表中所有数据
        # self.milvusHandler.delete_all_datas(self.collection)  # 删除当前表中所有数据

    
    # 从给定问题到查询出对应的mysql中数据项并做返回 
    def queryFromMysqlForQuestions(self,questions:list[str],top_k = 1) -> list:    # QA对列表
        questionEmbbedings = []
        for question in questions:
                testQuestionEmbbeding = self.vectorizeModel.encode(question)    
                questionEmbbedings.append(testQuestionEmbbeding)
        
        searchResults = self.milvusHandler.search_datas(self.collection,questionEmbbedings,top_k=top_k)
        searchIds = self.milvusHandler.extract_db_ids_from_Result(searchResults)
        queryResults = self.mysqlNBAFinalAverageDatasHandler.query_batch_by_ids(MYSQL_TABLE_NAMES[0],searchIds)
        # print(searchIds)
        return queryResults
        

def testMilvusHandler():
    collection_name = "test_collection"
    milvusHandler = MilvusHandler()
    vectorizeModel = SentenceTransformer(sentence_transformer_path) if sentenceTransformer is None else sentenceTransformer
    dim = 384
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="dbId", dtype=DataType.INT64)
    ]
    collection = milvusHandler.create_collection(collection_name, fields)
    mysqlNBAFinalAverageDatasHandler = MysqlNBAFinalAverageDatasHandler()

    temp = 1
    # milvusHandler.insert_datas(collection, questionEmbeddings, dbIds)
    # milvusHandler.delete_collection(collection_name)
    # print(milvusHandler.extract_db_ids(milvusHandler.query_by_conditions(collection,{'dbId' : 0},output_fields=["dbId"])))
    

def testNBAFinalAverageDatasMilvusHandler():
    nbaFinalAverageDatasMilvusHandler = NBAFinalAverageDatasMilvusHandler()
    qaOutputFilePath = os.path.join(project_root_dir,'outputs','Check','dbPipelineOutputs.txt')  # QA输出文件路径
    queryMysqlNum = 816 
    queryDatas = nbaFinalAverageDatasMilvusHandler.mysqlNBAFinalAverageDatasHandler.exportNBAFinalAverageDatasDicts(queryMysqlNum)
    questionTemplates = [
        f'{{year}}年NBA总决赛{{player}}的场均数据是多少？',
        f'请告诉我{{year}}年NBA总决赛{{player}}的场均数据',
        f'{{year}}年NBA总决赛{{player}}的场均数据如何？'
    ]
    questions = []
    questionEmbbedings = []
    for queryData in queryDatas:
        for questionTemplate in questionTemplates:
            fillDatas = {
                'year' : queryData['year'],
                'player' : queryData['player']
            }
            question = questionTemplate.format(**fillDatas)
            questions.append(question)
    
    queryResults = nbaFinalAverageDatasMilvusHandler.queryFromMysqlForQuestions(questions)

    assert len(questions) == len(questionTemplates) * len(queryResults)
    
    e2eQAs = []
    for i in range(len(questions)):
        question,queryResult_str = questions[i],str(queryResults[int(i / len(questionTemplates))])
        e2eQAs.append(f"Q: {question}\n A:{queryResult_str}")

    writeIterableToFile(qaOutputFilePath,e2eQAs)

    print('finished!')

    


if __name__ == '__main__':
    # testMilvusHandler()
    testNBAFinalAverageDatasMilvusHandler()
    # nbaFinalAverageDatasHandler = NBAFinalAverageDatasMilvusHandler()
    # nbaFinalAverageDatasHandler.extractMysqlNBAFinalAverageDatasAndExport2Milvus()