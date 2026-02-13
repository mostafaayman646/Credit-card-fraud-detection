import pandas as pd

def load_data():
    data_train = pd.read_csv('../data/split/train.csv')
    data_val = pd.read_csv('../data/split/val.csv')
    
    X_train = data_train.drop(columns=['Class']).to_numpy()
    y_train = data_train['Class'].to_numpy()
    X_val = data_val.drop(columns=['Class']).to_numpy()
    y_val = data_val['Class'].to_numpy()

    return X_train,y_train,X_val,y_val