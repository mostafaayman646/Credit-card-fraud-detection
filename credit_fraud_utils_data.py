import pandas as pd
from imblearn.under_sampling import RandomUnderSampler,NearMiss
from imblearn.over_sampling import SMOTE,RandomOverSampler

def load_data(args):
    if not args.CV:
        data_train = pd.read_csv(args.dataset+'/train.csv')
        data_val = pd.read_csv(args.dataset+'/val.csv')
        X_train = data_train.drop(columns=['Class']).to_numpy()
        y_train = data_train['Class'].to_numpy()
        X_val = data_val.drop(columns=['Class']).to_numpy()
        y_val = data_val['Class'].to_numpy()
    
    if args.UnderSamplingFactor != None:
        minority_size = (y_train == 1).sum()
        
        if args.UnderSampling == 'NearMiss':
            us = NearMiss(sampling_strategy={0: args.UnderSamplingFactor*minority_size})
        else:
            us = RandomUnderSampler(random_state=args.SEED,
                                    sampling_strategy={0: args.UnderSamplingFactor*minority_size})
        
        X_train,y_train = us.fit_resample(X_train,y_train)
        print("After UnderSampling")
        print(f"Class 1:{(y_train == 1).sum()} and Class 0: {(y_train == 0).sum()}")
    
    elif args.OverSamplingFactor != None:
        minority_size = (y_train == 1).sum()
        
        if args.OverSampling == 'SMOTE':
            os = SMOTE(random_state=args.SEED,
                        sampling_strategy={1:args.OverSamplingFactor*minority_size},
                        k_neighbors=args.SMOTE_k_neighbors)
        else:
            os = RandomOverSampler(random_state=args.SEED,
                                    sampling_strategy={1:args.OverSamplingFactor*minority_size})
        
        X_train,y_train = os.fit_resample(X_train,y_train)
        print("After OverSampling")
        print(f"Class 1:{(y_train == 1).sum()} and Class 0: {(y_train == 0).sum()}")
    
    return X_train,y_train,X_val,y_val