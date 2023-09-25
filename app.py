from src.mlproject01.logger import logging
from src.mlproject01.exception import CustomException
import sys
from src.mlproject01.components.data_ingestion import DataIngestion
from src.mlproject01.components.data_transformation import DataTransformation
from src.mlproject01.components.model_trainer import ModelTrainer,ModelTrainingConfig


if __name__=="__main__":
    logging.info("The execution has started...")

    try:
        data_ingestion = DataIngestion()
        train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()

        data_transformation=DataTransformation()
        train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        #Model Training 
        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr,test_arr))


    except Exception as e:
        raise CustomException(e, sys)