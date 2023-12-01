import pandas as pd
import numpy as np
import os
import sys
import warnings
warnings.filterwarnings('ignore')
from src.Adult_income_prediction.exception import customexception
from src.Adult_income_prediction.logger import logging
from src.Adult_income_prediction.components.data_file_location import Data_File_Station
from src.Adult_income_prediction.utils.utils import save_object
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler


class Data_Transformation:

    """
   Data_Transformation :- Class handle data and return preprocess data as a pickle file
    Data_Transformation contain methods :-
                                            Data_Transformation_primary         :- Segregatting depedent and indepedent data and futher process are done by the other methods 
                                            Data_Transformation_secondary       :- Replacing and handlling missing values ,outliers and return clean data or we can say cleaning process is done inside this method
                                            Data_Transformation_preprocessed    :- Setting up Pipeline of the data                                       
    """
    def __init__(self):
        self.file_station_config=Data_File_Station()
    
    def Data_Transformation_preprocessed(self):
        logging.info("Data preprocession has started")

        try:
            cate_col= ['workclass', 'occupation', 'race', 'sex', 'country']
            num_col = ['age', 'education_num', 'capital_gain', 'capital_loss','hours_per_week']
            
            report={
                    'workclass': ['State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov','Local-gov', 'Self-emp-inc', 'Without-pay'],
                    'occupation': ['Adm-clerical', 'Exec-managerial', 'Handlers-cleaners','Prof-specialty', 'Other-service', 'Sales', 'Craft-repair','Transport-moving', 'Farming-fishing', 'Machine-op-inspct','Tech-support', 'Protective-serv', 'Armed-Forces','Priv-house-serv'],
                    'race': ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo','Other'],
                    'sex': ['Male', 'Female'],
                    'country': ['United-States', 'Cuba', 'Jamaica', 'India', 'Mexico','Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran','Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand','Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal','Dominican-Republic', 'El-Salvador', 'France', 'Guatemala', 'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru','Scotland', 'Trinadad&Tobago', 'Greece', 'Nicaragua', 'Vietnam','Hong', 'Ireland', 'Hungary', 'Holand-Netherlands']
                    }
            
            num_pipeline = Pipeline(
                                    steps=[
                                        ("imputer",SimpleImputer(strategy='median')),
                                        ("scaler",StandardScaler())
                                        ]
                                    )
            cate_pipeline = Pipeline(
                                    steps=[
                                        ("imputer",SimpleImputer(strategy='most_frequent')),
                                        ("ordinal_encoder",OrdinalEncoder(categories=[report['workclass'],report['occupation'],report['race'],report['sex'],report['country']])),
                                        ("scaler",StandardScaler())
                                        ]
                                    )
            preprocessor = ColumnTransformer([
                                        ("num_pipeline",num_pipeline,num_col),
                                        ("cate_pipeline",cate_pipeline,cate_col)
                                        ]
                                    )
            logging.info("data preprocessed has completed")
            return (preprocessor)
        except Exception as e:
            logging.info(("there may be error in data transformation preprocessed"))
            raise customexception(e,sys)   
    
    def Data_Transformation_secondary(self,data):
        logging.info("Data transformation has started")

        try:
            for req in data:
                data[req].replace(" ?",np.NaN,inplace=True)

            logging.info("Dropping all the null values in workclass and occupation columns")
            data.dropna(subset=['workclass','occupation'],inplace=True)
            data.rename(columns={'education-num':"education_num",
                   "marital-status":"marital_status",
                   "capital-gain":"capital_gain",
                   "capital-loss":"capital_loss",
                   "hours-per-week":"hours_per_week"
                   },inplace=True)
            val = str(data['country'].mode())
            data['country'].fillna(val,inplace=True)

            data['country']=data['country'].replace({"United-States":" United-States",
                                     "0     United-States\nName: country, dtype: object":" United-States",
                                     " Outlying-US(Guam-USVI-etc)":" United-States"})
            
            q1= data['age'].quantile(0.10)
            q3= data["age"].quantile(0.75)
            IQR=q3-q1
            upper_limit = q3 + 1.5 * IQR
            lower_limit = q1 - 1.5 * IQR
            data.loc[(data["age"] > upper_limit), "age"]=upper_limit
            data.loc[(data["age"] < lower_limit), "age"]=lower_limit
            
            for i in data:
                if data[i].dtypes == "O":
                    data[i]=data[i].str.replace(" ","")
            data['salary'].replace({'<=50K':0, '>50K':1},inplace=True)
            data.drop_duplicates(inplace=True)
            
            return data
        except Exception as e:
            logging("Error in Data_Transformation_secondary")
            raise customexception(e,sys)

    def Data_Transformation_primary(self,train_path,test_path):
        logging.info("Data transformation Primary has started")

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            train_df=self.Data_Transformation_secondary(train_df)
            test_df=self.Data_Transformation_secondary(test_df)
            target_col = 'salary'
            drop_col = [target_col,
                        'education',
                        'fnlwgt',
                        'marital_status',
                        'relationship']
            logging.info("Data splitting is started")
            input_x_train = train_df.drop(columns=drop_col,
                                          axis=1)
            input_y_train = train_df[target_col]
            input_x_test = test_df.drop(columns=drop_col,
                                          axis=1)
            input_y_test = test_df[target_col]
            logging.info("train and test data is spliited successfully")

            preprocessor_obj=self.Data_Transformation_preprocessed()
            input_x_train_arr = preprocessor_obj.fit_transform(input_x_train)
            input_x_test_arr = preprocessor_obj.transform(input_x_test)
            train_array = np.c_[input_x_train_arr,np.array(input_y_train)]
            test_array = np.c_[input_x_test_arr,np.array(input_y_test)]

            save_object(
                file_path=self.file_station_config.preprocessor_file,
                obj=preprocessor_obj
            )
            
            logging.info(f'data saved inside the preprocessed object')
            logging.info(f'preproceeing finished')
            
            return (
                train_array,
                test_array,
                self.file_station_config.preprocessor_file
            )
            
        except Exception as e:
            logging.info("there may be some error in data transforamtion primary")
            raise customexception(e,sys)
        



            






            


    

            
        
