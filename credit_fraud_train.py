from credit_fraud_utils_data import load_data
from credit_fraud_utils_eval import evaluate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler,PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier,RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np

#Constants:
SEED = 17

def train(X_train,t_train,X_val,add_intercept):
    #Initialize models
    lr = LogisticRegression(solver='lbfgs',random_state=SEED,fit_intercept=add_intercept)
    rfc = RandomForestClassifier(n_estimators=50,max_depth=10)
    nn = MLPClassifier(random_state=SEED,max_iter=300)
    
    #Voting
    vcf = VotingClassifier([
        ('lr',lr),('rfc',rfc),('nn',nn)
    ],voting='soft',weights=[1,2,1])
    
    pip = Pipeline([
        ('Poly',PolynomialFeatures(degree=1,include_bias=add_intercept)),
        ('Scaler',MinMaxScaler()),
        ('vcf',vcf)
    ],)

    pip.fit(X_train,t_train)

    y_train = pip.predict(X_train)
    y_val = pip.predict(X_val)
    
    return y_train,y_val

if __name__ == '__main__':
    np.random.seed(SEED)  # If you want to fix the results
    
    X_train,t_train,X_val,t_val = load_data()
    
    y_train,y_val = train(X_train,t_train,X_val,add_intercept=True)
    
    desc = "Using voting classifier between random forrest and nn and logistic regression and minmax scale on data only"
    
    evaluate(t_train,y_train,t_val,y_val,description=desc,save_results=True)
