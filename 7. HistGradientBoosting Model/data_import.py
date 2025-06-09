# data_import.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def load_data(path='D:/Preparation_for_Github/5. Machine Learning Using Python/data.csv'):
    df = pd.read_csv(path)
    return df



def preprocess_data(df):
    # Separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_vars = ['department', 'salary']

    # One-hot encode categorical variables
    for var in cat_vars:
        cat_list = pd.get_dummies(df[var], prefix=var)
        df = df.join(cat_list)
    df.drop(columns=cat_vars, axis=1, inplace=True)

    # Remove 'quit' from numeric columns if present
    if 'quit' in numeric_cols:
        numeric_cols.remove('quit')

    # Impute missing values only on numeric columns
    imputer = SimpleImputer(strategy='mean')
    df_numeric = df[numeric_cols]  # Extract numeric columns
    imputer.fit(df_numeric)  # ⬅️ Now we fit the imputer
    df[numeric_cols] = imputer.transform(df_numeric)  # Then transform

    # Split features and target
    X = df.loc[:, df.columns != 'quit']
    y = df['quit']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test