from credit_fraud_utils_data import load_data
from credit_fraud_utils_eval import evaluate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler,PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import argparse
import pickle

parser = argparse.ArgumentParser(description='Credit_Card_Parser')

parser.add_argument('--SEED', type=int, default=17)
#DATA Args
parser.add_argument('--CV',type = bool, default=False)
parser.add_argument('--dataset', type=str, default='../data/split')
parser.add_argument('--UnderSampling', type=str, default='NearMiss',
                    help='NearMiss or RandomUnderSampling')
parser.add_argument('--UnderSamplingFactor', type=int, default=None)
parser.add_argument('--OverSampling', type=str, default='SMOTE',
                    help='SMOTE or RandomOverSampling')
parser.add_argument('--OverSamplingFactor', type=int, default=10)
parser.add_argument('--SMOTE_k_neighbors', type=int, default=5)

#Training Args
parser.add_argument('--fit_intercept', type=bool, default=True)
parser.add_argument('--RFC_n_estimators', type=int, default=50)
parser.add_argument('--RFC_max_depth', type=int, default=10)
parser.add_argument('--NN_max_iter', type=int, default=3000)
parser.add_argument('--voting_type', type=str, default='soft',
                    help='soft or hard')
parser.add_argument('--description', type=str, default=None,
                    help='Write your comment for this training before saving in json')
parser.add_argument('--SaveResults', type=bool, default=False)
parser.add_argument('--SaveModel', type=bool, default=False)
parser.add_argument('--ModelPath', type=str, default='Model/VotingClassifier.pkl')

args = parser.parse_args()


def train(X_train,t_train,X_val):
    #Initialize models
    lr = LogisticRegression(solver='lbfgs',random_state=args.SEED,fit_intercept=args.fit_intercept)
    rfc = RandomForestClassifier(n_estimators=args.RFC_n_estimators,max_depth=args.RFC_max_depth)
    nn = MLPClassifier(random_state=args.SEED,max_iter=args.NN_max_iter,hidden_layer_sizes=
                        (10,10,10))
    
    #Voting
    vcf = VotingClassifier([
        ('lr',lr),('rfc',rfc)
    ],voting=args.voting_type,weights=[1,2])
    
    pip = Pipeline([
        ('Poly',PolynomialFeatures(degree=1,include_bias=args.fit_intercept)),
        ('Scaler',MinMaxScaler()),
        ('vcf',vcf)
    ],)

    pip.fit(X_train,t_train)

    y_train = pip.predict(X_train)
    y_val = pip.predict(X_val)
    
    return y_train,y_val,pip

def save_model_pkl(model, filename=args.ModelPath):
    """
    Saves a model object to a .pkl file using the pickle library.
    """
    try:
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        print(f"Success! Model saved to {filename}")
    except Exception as e:
        print(f"An error occurred while saving: {e}")

if __name__ == '__main__':
    np.random.seed(args.SEED)  # If you want to fix the results
    
    X_train,t_train,X_val,t_val = load_data(args)
    
    y_train,y_val,model = train(X_train,t_train,X_val)
    
    evaluate(t_train,y_train,t_val,y_val,description=args.description,save_results=args.SaveResults)
    
    if args.SaveModel:
        save_model_pkl(model)