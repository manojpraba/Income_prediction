import pandas as pd
import numpy as np
from src.Adult_income_prediction.logger import logging
from src.Adult_income_prediction.exception import customexception

import os
import sys
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

class DataIngestionConfig:
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")

class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config=DataIngestionConfig()
    def initiate_data_ingestion(self):
        logging.info("Data Ingestion started")

        try:
            data=pd.read_csv(Path(os.path.join('notebooks/data','adult.csv')))
            logging.info("dataset is readed successfully")
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            logging.info("artifact folder is created")
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            train_data,test_data=train_test_split(data,test_size=0.25,random_state=44)
            logging.info("data splitted successfully")
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info("train and test data spilted successfuly")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Exception occured in dataingestion")
            raise customexception(e,sys)

if __name__ == '__main__':
    dd=DataIngestion()
    dd.initiate_data_ingestion()
    print(dd)
        


