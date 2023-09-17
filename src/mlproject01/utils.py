import os
import sys
from src.mlproject01.exception import CustomException
from src.mlproject01.logger import logging
import pandas as pd
import pymysql
import pickle
import numpy as np
from dotenv import load_dotenv

load_dotenv()

host=os.getenv("host")
user=os.getenv("user")
password=os.getenv("password")
db=os.getenv("db")

def read_sql_data():
    logging.info("Reading sql database started..")

    try:
        db_conn=pymysql.connect(
            host=host,
            user=user,
            password=password,
            database=db
        )
        logging.info("Connection establised...")
        df=pd.read_sql_query("select * from student",db_conn)

        print(df.head())

        return df

    except Exception as e:
        raise CustomException(e, sys)

def save_object(filepath,obj):
    try:
        dir_path=os.path.dirname(filepath)

        os.makedirs(dir_path,exist_ok=True)

        with open(filepath,"wb") as file_obj:
            pickle.dump(obj,file_obj)


    except Exception as e:
        raise CustomException(e,sys)

