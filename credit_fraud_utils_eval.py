from sklearn.metrics import confusion_matrix,classification_report
import json,os


def evaluate(t_train,y_train,t_val,y_val,description = None,save_results = False):
    cm = confusion_matrix(t_train,y_train)
    cr_train = classification_report(t_train,y_train,output_dict=save_results)

    tn_train,fp_train,fn_train,tp_train = cm.ravel()

    print("Training results______________________________")
    print(f"TN:{tn_train},FP:{fp_train},FN:{fn_train},TP:{tp_train}")
    print(cr_train)
    print("____________________________")

    if save_results:
        new_entry = {
        "DescriptionL: ": description,
        "Type": "Training Results",
        "confusion_matrix": {
            "TN": int(tn_train),
            "FP": int(fp_train),
            "FN": int(fn_train),
            "TP": int(tp_train)
        },
        "metrics_report": cr_train
        }

    cm = confusion_matrix(t_val,y_val)
    cr_val = classification_report(t_val,y_val,output_dict=save_results)

    tn_val,fp_val,fn_val,tp_val = cm.ravel()

    print("Validation results______________________________")
    print(f"TN:{tn_val},FP:{fp_val},FN:{fn_val},TP:{tp_val}")
    print(cr_val)
    print("____________________________")

    if save_results:
        new_entry_2 = {
        "Type": "Validation Results",
        "confusion_matrix": {
            "TN": int(tn_val),
            "FP": int(fp_val),
            "FN": int(fn_val),
            "TP": int(tp_val)
        },
        "metrics_report": cr_val
        }
    
    
        filename = 'Training_Results/results.json'
        
        current_data = []
        
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                current_data = json.load(f)

        current_data.append(new_entry)
        current_data.append(new_entry_2)
        current_data.append("_______________________________________")

        with open(filename, 'w') as f:
            json.dump(current_data, f, indent=4)