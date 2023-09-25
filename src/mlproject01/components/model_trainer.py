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

@dataclass
class ModelTrainingConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainingConfig()
    
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
