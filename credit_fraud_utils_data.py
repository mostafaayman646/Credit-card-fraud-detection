import pandas as pd
from imblearn.under_sampling import RandomUnderSampler,NearMiss
from imblearn.over_sampling import SMOTE,RandomOverSampler

def load_data(SEED,Over_sampling_factor = None,Under_sampling_factor=None):
    data_train = pd.read_csv('../data/split/train.csv')
    data_val = pd.read_csv('../data/split/val.csv')
    
    X_train = data_train.drop(columns=['Class']).to_numpy()
    y_train = data_train['Class'].to_numpy()
    X_val = data_val.drop(columns=['Class']).to_numpy()
    y_val = data_val['Class'].to_numpy()

    if Under_sampling_factor:
        minority_size = (y_train == 1).sum()
        # rus = RandomUnderSampler(random_state=SEED,sampling_strategy={0: Under_sampling_factor*minority_size})
        nm = NearMiss(sampling_strategy={0: Under_sampling_factor*minority_size})
        X_train,y_train = nm.fit_resample(X_train,y_train)
        print("After UnderSampling")
        print(f"Class 1:{(y_train == 1).sum()} and Class 0: {(y_train == 0).sum()}")
    
    elif Over_sampling_factor:
        minority_size = (y_train == 1).sum()
        # smt = SMOTE(random_state=SEED,sampling_strategy={1:Over_sampling_factor*minority_size},k_neighbors=5)
        ros = RandomOverSampler(random_state=SEED,sampling_strategy={1:Over_sampling_factor*minority_size})
        X_train,y_train = ros.fit_resample(X_train,y_train)
        print("After OverSampling")
        print(f"Class 1:{(y_train == 1).sum()} and Class 0: {(y_train == 0).sum()}")
    return X_train,y_train,X_val,y_val