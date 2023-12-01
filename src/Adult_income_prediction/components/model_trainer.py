import sys
import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.svm import SVC
from src.Adult_income_prediction.utils.utils import save_object,load_object
from src.Adult_income_prediction.exception import customexception
from src.Adult_income_prediction.logger import logging
from src.Adult_income_prediction.components.data_file_location import Data_File_Station
from src.Adult_income_prediction.utils.utils import evaluate_model

class Model_Trainer:
    def __init__(self) -> None:
        self.model_trainer_config=Data_File_Station()
    
    def model_trainer(self,train_array,test_array):

        logging.info("Model training is started")

        try:
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1] )
             
            model =XGBClassifier(colsample_bytree=0.8, 
                                  learning_rate=0.2, 
                                  max_depth=5, 
                                  min_child_weight=1, 
                                  n_estimators=100, 
                                  subsample =1.0)
            model_report = evaluate_model(X_train,y_train,X_test,y_test,model)
            logging.info(f"==="*50)
            
            logging.info(f"MODEL ACCURACY \n {model_report}")
            
            logging.info(f"==="*50)
             
            save_object(
                self.model_trainer_config.trained_model_file_path,
                model)
        except Exception as e:
            logging.info("there may be error in model trainer primary")
            raise customexception(e,sys) 
            
     

             
        



