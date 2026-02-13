from credit_fraud_utils_data import load_data
from credit_fraud_utils_eval import evaluate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler,PolynomialFeatures
from sklearn.linear_model import LogisticRegression
import numpy as np

#Constants:
SEED = 17

def train(X_train,t_train,X_val,add_intercept):
    pip = Pipeline([
        ('Poly',PolynomialFeatures(degree=1,include_bias=add_intercept)),
        ('Scaler',MinMaxScaler()),
        ('Logistic',LogisticRegression(solver='lbfgs',random_state=SEED,fit_intercept=add_intercept))
    ],)

    pip.fit(X_train,t_train)

    y_train = pip.predict(X_train)
    y_val = pip.predict(X_val)
    
    return y_train,y_val

if __name__ == '__main__':
    np.random.seed(SEED)  # If you want to fix the results
    
    X_train,t_train,X_val,t_val = load_data()
    
    y_train,y_val = train(X_train,t_train,X_val,add_intercept=True)
    
    # desc = "Logistic regression with default parameters and minmax scale on data only"
    
    evaluate(t_train,y_train,t_val,y_val)
