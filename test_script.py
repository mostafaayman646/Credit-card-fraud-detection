import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report

# Load the model
with open('Model/VotingClassifier.pkl', 'rb') as f:
    model = pickle.load(f)

#Load Test Data
Test = pd.read_csv('../data/split/test.csv')
X_test = Test.drop(columns=['Class']).to_numpy()
y_test = Test['Class'].to_numpy()

#Predict
y_pred = model.predict(X_test)

#Evaluation
cm = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred,output_dict=False)

tn,fp,fn,tp = cm.ravel()

print("Results______________________________")
print(f"TN:{tn},FP:{fp},FN:{fn},TP:{tp}")
print(cr)
print("____________________________")