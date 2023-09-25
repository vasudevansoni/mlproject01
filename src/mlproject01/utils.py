import os
import sys
from src.mlproject01.exception import CustomException
from src.mlproject01.logger import logging
import pandas as pd
import pymysql
import pickle
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

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

def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        report={}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            param = params[list(models.keys())[i]]

            gs = GridSearchCV(model,param,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            y_train_predict = model.predict(X_train)
            y_test_predict = model.predict(X_test)

            train_model_score = r2_score(y_train,y_train_predict)
            test_model_score = r2_score(y_test,y_test_predict)

            report[list(models.keys())[i]] = test_model_score

        return report


    except Exception as e:
        raise CustomException(e,sys)

    

