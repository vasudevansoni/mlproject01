import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import (AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor)
from src.mlproject01.exception import CustomException
from src.mlproject01.logger import logging
from src.mlproject01.utils import save_object, evaluate_models
from urllib.parse import urlparse
import numpy as np
import mlflow
import dagshub
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error


@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()
    
    def eval_metrics(self,actual,pred):
        rmse = np.sqrt(mean_squared_error(actual,pred))
        mae = mean_absolute_error(actual,pred)
        r2 = r2_score(actual,pred)

        return rmse,mae,r2

    
    def initiate_model_trainer(self,train_array,test_array):
        
        try:
            logging.info("Spliting training and testing array..")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            # List of Models 
            models={
                "CatBoosting Regressor":CatBoostRegressor(verbose=False),
                "Linear Regression":LinearRegression(),
                "Decision Tree":DecisionTreeRegressor(),
                "XGB Regressor":XGBRegressor(),
                "ADA Boost Regressor":AdaBoostRegressor(),
                "Random Forest":RandomForestRegressor(),
                "Gradient Boosting":GradientBoostingRegressor()
            }

            params={
                "CatBoosting Regressor":{
                    'depth':[6,8,10],
                    'learning_rate':[0.1,0.01,0.05],
                    'iterations':[30,50,100]
                },
                "Linear Regression":{},
                "Decision Tree":{
                    'criterion':['squared_error','poisson','friedman_mse','absolute_error'],
                    #'splitter':['best','random'],
                    #'max_features':['sqrt','log2']
                },
                "XGB Regressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "ADA Boost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Random Forest":{
                    # 'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    # 'max_features':['sqrt','log2'],
                    'n_estimators':[8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators':[8,16,32,64,128,256],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'loss':['squared_error','huber','absolute_error','quantile'],

                }
            }

            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test,models,params)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model = models[best_model_name]

            print(f"This is the best model --> {best_model}")

            model_names = list(params.keys())

            actual_model=""

            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model
            
            best_params = params[actual_model]

            # mlfow

            mlflow.set_registry_uri("https://dagshub.com/vasudevansoni/mlproject01.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
            
            dagshub.init("mlproject01", "vasudevansoni", mlflow=True)
            with mlflow.start_run():

                predicted_qualities = best_model.predict(X_test)

                rsme,mae,r2 = self.eval_metrics(y_test,predicted_qualities)

                mlflow.log_params(best_params)

                mlflow.log_metric("rsme",rsme)
                mlflow.log_metric("mae",mae)
                mlflow.log_metric("r2",r2)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(best_model,"model",registered_model_name=actual_model)
                
                else:
                    mlflow.sklearn.log_model(best_model,"model")



            if best_model_score<0.6:
                raise CustomException("No best model found..")
            logging.info(f"Best model found on both training and testing datasets..")

            save_object(
                self.model_trainer_config.trained_model_file_path,
                best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e,sys)
