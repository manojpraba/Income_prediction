# Adult Census Income-Prediction


## Problem Statement:
The Goal is to predict whether a person has an income of more than 50K a year or not.
This is basically a binary classification problem where a person is classified into the **>50K group** or **<=50K group**.


## End To End Project 

This is a classification problem where we need to predict whether a person earns more than a sum of 50,000 k anuually or not. This classification task is accomplished by using a XGB Classifier trained on the dataset extracted by Barry Becker from the 1994 Census database. The dataset contains about 33k records and 15 features which after all the implementation of all standard techniques like Data Cleaning, Feature Engineering, Feature Selection, Outlier Treatment, etc was feeded to our Classifier which after training and testing, was deployed in the form of a web application.

## Input
![alt tag](https://github.com/manojpraba/Income_prediction/blob/main/images/Index%20page.png)
![alt tag](https://github.com/manojpraba/Income_prediction/blob/main/images/prediction%20page.png)

## Output
![alt tag](https://github.com/manojpraba/Income_prediction/blob/main/images/result%20page.png)

## MLFLOW UI
![alt tag](https://github.com/manojpraba/Income_prediction/blob/main/images/mlflow%20ui.png)

### Model performance for Training set
- Accuracy: 0.8676
- F1 score: 0.6874
- Precision: 0.8261
- Recall: 0.5885
- Roc Auc Score: 0.7739
----------------------------------
### Model performance for Test set
- Accuracy: 0.8473
- F1 score: 0.6496
- Precision: 0.7810
- Recall: 0.5560
- Roc Auc Score: 0.7514


## Libraries used
#### Flask
#### Sklearn
#### Pandas
#### Numpy

## Tech Stack
#### Front-End: HTML and CSS
#### Back-End: Flask
#### IDE: Jupyter notebook, VScode

## How to Run:
#### 1. create new enviroment with python=3.8
#### 2. install requirements.txt file.
#### 3. Run app.py file.
#### 4. Run mlflow ui

## Workflow
![alt tag](https://github.com/manojpraba/Income_prediction/blob/main/images/Architecture.jpg)

