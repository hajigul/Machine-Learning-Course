# data_import.py

import pandas as pd
import pandas_profiling
from sklearn.model_selection import train_test_split

def load_data(path='D:/Preparation_for_Github/5. Machine Learning Using Python/data.csv'):
    
    df = pd.read_csv(path)
    return df

def preprocess_data(df):
    # One-hot encode categorical variables
    cat_vars = ['department', 'salary']
    for var in cat_vars:
        cat_list = pd.get_dummies(df[var], prefix=var)
        df = df.join(cat_list)
    df.drop(columns=cat_vars, axis=1, inplace=True)
    
    # Split features and target
    X = df.loc[:, df.columns != 'quit']
    y = df['quit']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test