import os
import sys
import pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from src.Adult_income_prediction.exception import customexception
from src.Adult_income_prediction.logger import logging
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, classification_report,ConfusionMatrixDisplay, \
                            precision_score, recall_score, f1_score, roc_auc_score,roc_curve,confusion_matrix

import mlflow
import mlflow.sklearn





def save_object(file_path,obj) -> None:
    try:
        dir_name=os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        logging.info("Error in save object")
        raise customexception(e,sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info("Error in loading object")
        raise customexception(e,sys)
    
def evaluate_clf(true, predicted):
    '''
    This function takes in true values and predicted values
    Returns: Accuracy, F1-Score, Precision, Recall, Roc-auc Score
    '''
    acc = accuracy_score(true, predicted) # Calculate Accuracy
    f1 = f1_score(true, predicted) # Calculate F1-score
    precision = precision_score(true, predicted) # Calculate Precision
    recall = recall_score(true, predicted)  # Calculate Recall
    roc_auc = roc_auc_score(true, predicted) #Calculate Roc
    return acc, f1 , precision, recall, roc_auc

def evaluate_model(X_train,y_train,X_test,y_test,model):
    try:
        with mlflow.start_run():
            report = {}
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            y_train_pred=model.predict(X_train)
            scores=accuracy_score(y_test,y_pred)
            # Training set performance
            model_train_accuracy, model_train_f1,model_train_precision,\
            model_train_recall,model_train_rocauc_score=evaluate_clf(y_train ,y_train_pred)
            print("------------ Best model Found--------------")
            print('Model performance for Training set')
            print("- Accuracy: {:.4f}".format(model_train_accuracy))
            #training_accuracy_list.append(model_train_accuracy)
            print('- F1 score: {:.4f}'.format(model_train_f1)) 
            print('- Precision: {:.4f}'.format(model_train_precision))
            print('- Recall: {:.4f}'.format(model_train_recall))
            print('- Roc Auc Score: {:.4f}'.format(model_train_rocauc_score))
        # print(f'- COST: {train_cost}.')

            print('----------------------------------')
            model_test_accuracy,model_test_f1,model_test_precision,\
            model_test_recall,model_test_rocauc_score=evaluate_clf(y_test, y_pred)
            print('Model performance for Test set')
            print('- Accuracy: {:.4f}'.format(model_test_accuracy))
            print('- F1 score: {:.4f}'.format(model_test_f1))
            print('- Precision: {:.4f}'.format(model_test_precision))
            print('- Recall: {:.4f}'.format(model_test_recall))
            print('- Roc Auc Score: {:.4f}'.format(model_test_rocauc_score))
            report[model]=scores
            acc, f1 , precision, recall, roc_auc=evaluate_clf(y_test,y_pred)
            metrics={"accuracy":acc,"precision":precision,"recall":recall,"roc_auc":roc_auc}
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(model,"XGBOOST")
            return(report)
        
    except Exception as e:
        logging.info("there may be error in evaluate model")
        raise customexception(e,sys)