from transformers import AutoTokenizer, AutoModelForCausalLM,LlamaForCausalLM
from peft import PeftModel, PeftConfig
import torch
import os,sys
from tqdm import tqdm
from sqlalchemy import create_engine, Table, MetaData, select, and_
from sqlalchemy.orm import sessionmaker
from typing import Union


# 项目根目录
project_root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# 项目python源码目录
py_src_root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(py_src_root_dir)
sys.path.append(project_root_dir)

from common.myFile import *

MYSQL_USERNAME = os.getenv('HupuKiller_MYSQL_USERNAME')
MYSQL_PASSWORD = os.getenv('HupuKiller_MYSQL_PASSWORD')
MYSQL_DATABASE_NAME = 'HupuKillerDatabase'
MYSQL_TABLE_NAMES = ['NBAFinalAverageDatasTable']

class MySQLDatabaseHandler:
    def __init__(self, username, password, host, port, database):
        self.database_url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        self.engine = create_engine(self.database_url, echo=False)
        self.metadata = MetaData()
        self.metadata.reflect(bind=self.engine)
        self.Session = sessionmaker(bind=self.engine)
        self.session = self.Session()
        self.tables = {} 
        self.tables[MYSQL_TABLE_NAMES[0]] = self.create_table_object(MYSQL_TABLE_NAMES[0])

    def create_table_object(self, table_name):
        """
        Dynamically load the table object based on the database schema.
        """
        if table_name in self.metadata.tables:
            return self.metadata.tables[table_name]
        else:
            raise ValueError(f"Table '{table_name}' does not exist in the database.")

    def insertDatas(self, table_name, datas: Union[list[dict],dict]):
        """
        Insert a single record into the specified table.
        :param table_name: The name of the table.
        :param data: A dictionary containing the data to insert.
        """
        if isinstance(datas,dict):
            insert_query = self.tables[table_name].insert().values(**datas)
            with self.engine.connect() as connection:
                connection.execute(insert_query)
                connection.commit()   # commit transaction
        elif isinstance(datas,list):
            with self.engine.connect() as connection:
                for data in datas:
                    insert_query = self.tables[table_name].insert().values(**data)
                    connection.execute(insert_query)
                connection.commit()   # commit transaction
        else:
            raise Exception('Error Parameter for insertData')
        
    def query_one(self, table_name, filter_condition):
        """
        Query a single record from the specified table.
        :param table_name: The name of the table.
        :param filter_condition: A SQLAlchemy filter condition.
        """
        query = select(self.tables[table_name]).where(filter_condition).limit(1)
        with self.engine.connect() as connection:
            result = connection.execute(query).fetchone()
        return result

    def query_batch(self, table_name:str = None,num = None):
        """
        Query all records from the specified table.
        :param table_name: The name of the table.
        """
        if num is None:
            query = select(self.tables[table_name])
            with self.engine.connect() as connection:
                result = connection.execute(query).fetchall()
        else:
            """
            Query the top 'num' records from the specified table.
            :param table_name: The name of the table.
            :param num: The number of records to retrieve.
            """
            query = select(self.tables[table_name]).limit(num)
            with self.engine.connect() as connection:
                result = connection.execute(query).fetchall()
        return result
    

    def query_batch_by_ids(self, table_name:str = None, id_list: list = None):
        """
        Query records where the ID column matches any value in the provided list.
        :param table_name: The name of the table.
        :param id_column_name: The name of the ID column to filter on.
        :param id_list: A list of IDs to query.
        :return: A list of matching records.
        """
        # Ensure the table and column exist
        table = self.tables[table_name]
        id_column = table.c['id']  # Access the column dynamically

        # Build the query with the IN clause
        query = select(table).where(id_column.in_(id_list))
        # Execute the query
        with self.engine.connect() as connection:
            result = connection.execute(query).fetchall()
        return result

    def deleteDatas(self, table_name, filter_condition=None):
        """
        Delete records from the specified table.
        :param table_name: The name of the table.
        :param filter_condition: A SQLAlchemy filter condition.
        """
        table = self.create_table_object(table_name)
        delete_query = table.delete()  
        if filter_condition is not None:  # 添加条件，仅在必要时
            delete_query = delete_query.where(filter_condition)
        with self.engine.connect() as connection:
            connection.execute(delete_query)
            connection.commit()
            
    def close_connection(self):
        """Close the database session."""
        self.session.close()


def testMySQLDatabaseHandler():
    db_handler = MySQLDatabaseHandler(username=MYSQL_USERNAME, password=MYSQL_PASSWORD, host="localhost", port=3306, database=MYSQL_DATABASE_NAME)
    data_1 = {
        "id": "1",
        "player": "LeBron James",
        "minutes": 40.2,
        "age": 36,
        "score": 30.5,
        "rebound": 8.0,
        "assist": 7.5,
        "steal": 1.2,
        "block": 0.8
    }
    data_2 = {
        "id": "2",
        "player": "LeBron James",
        "minutes": 40.2,
        "age": 36,
        "score": 30.5,
        "rebound": 8.0,
        "assist": 7.5,
        "steal": 1.2,
        "block": 0.8
    }
    datas = [data_1,data_2]
    # db_handler.insertDatas(MYSQL_TABLE_NAMES[0], data_1)
    db_handler.insertDatas(MYSQL_TABLE_NAMES[0], datas)
    # result = db_handler.query_one(MYSQL_TABLE_NAMES[0], filter_condition=(db_handler.tables[MYSQL_TABLE_NAMES[0]].c.id == "1"))
    # print(result)
    # results = db_handler.query_all(MYSQL_TABLE_NAMES[0])
    # print(results)
    # db_handler.deleteDatas(MYSQL_TABLE_NAMES[0], filter_condition=(db_handler.tables[MYSQL_TABLE_NAMES[0]].c.id == "1"))
    db_handler.deleteDatas(MYSQL_TABLE_NAMES[0], filter_condition=None)
    db_handler.close_connection()


class MysqlNBAFinalAverageDatasHandler(MySQLDatabaseHandler):
    def __init__(self, username, password, host, port, database):
        super().__init__(username, password, host, port, database)
        self.tableName = MYSQL_TABLE_NAMES[0]
        # self.mysqlHandler = MySQLDatabaseHandler(username=MYSQL_USERNAME, password=MYSQL_PASSWORD, host="localhost", port=3306, database=MYSQL_DATABASE_NAME)
    
    def exportNBAFinalAverageDatasDicts(self,queryNum = None):
        allDataDicts = []
        queryDatas = self.query_batch(self.tableName,queryNum)
        # queryDatas = self.mysqlHandler.query_n(self.tableName,3)
        for queryData in queryDatas:
            dataDict = {
                'id' : queryData[0],
                'year' : queryData[1],
                'player' : queryData[2],
                'minutes' : queryData[3],
                'age' : queryData[4],
                'score' : queryData[6],
                'rebound' : queryData[7],
                'assist' : queryData[8]
            }
            allDataDicts.append(dataDict)
        return allDataDicts
    

def testMysqlNBAFinalAverageDatasHandler():
    handler = MysqlNBAFinalAverageDatasHandler(username=MYSQL_USERNAME, password=MYSQL_PASSWORD, host="localhost", port=3306, database=MYSQL_DATABASE_NAME)
    outputfile = '/data/workspace/projects/HupuKiller/outputs/Check/mysqlNBAFinalAverageDatas.txt'
    # handler.exportNBAFinalAverageDatasDicts()
    # queryResults = handler.query_batch_by_ids(MYSQL_TABLE_NAMES[0],[0,1,2,3,4,5])
    queryResults = handler.query_batch(handler.tableName)
    questionStrs = []
    for queryResult in queryResults:
        questionStr = str(queryResult)
        questionStrs.append(questionStr)
    writeIterableToFile(outputfile,questionStrs)

    temp = 1




if __name__ == "__main__":
    testMysqlNBAFinalAverageDatasHandler()


