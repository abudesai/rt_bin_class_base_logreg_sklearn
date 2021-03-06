import os, shutil
import sys
import time
import numpy as np
import pandas as pd
import pprint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, './../app')
import algorithm.utils as utils 
import algorithm.model_trainer as model_trainer
import algorithm.model_server as model_server
import algorithm.model_tuner as model_tuner
import algorithm.preprocessing.pipeline as pipeline
import algorithm.model.logistic_regression as logistic_regression


inputs_path = "./ml_vol/inputs/"

data_schema_path = os.path.join(inputs_path, "data_config")

data_path = os.path.join(inputs_path, "data")
train_data_path = os.path.join(data_path, "training", "binaryClassificationBaseMainInput")
test_data_path = os.path.join(data_path, "testing", "binaryClassificationBaseMainInput")

model_path = "./ml_vol/model/"
model_access_path = os.path.join(model_path, "model.save")
hyper_param_path = os.path.join(model_path, "model_config")
model_artifacts_path = os.path.join(model_path, "artifacts")

output_path = "./ml_vol/outputs"
hpt_results_path = os.path.join(output_path, "hpt_outputs")
testing_outputs_path = os.path.join(output_path, "testing_outputs")
errors_path = os.path.join(output_path, "errors")

test_results_path = "test_results"
if not os.path.exists(test_results_path): os.mkdir(test_results_path)

'''
this script is useful for doing the algorithm testing locally without needing 
to build the docker image and run the container.
make sure you create your virtual environment, install the dependencies
from requirements.txt file, and then use that virtual env to do your testing. 
This isnt foolproof. You can still have host os, or python-version related issues, so beware. 
'''

model_name = "logistic_regression_sklearn"


def create_ml_vol():    
    dir_tree = {
        "ml_vol": {
            "inputs": {
                "data_config": None,
                "data": {
                    "training": {
                        "binaryClassificationBaseMainInput": None
                    },
                    "testing": {
                        "binaryClassificationBaseMainInput": None
                    }
                }
            },
            "model": {
                "model_config": None,
                "artifacts": None,
            }, 
            
            "outputs": {
                "hpt_outputs": None,
                "testing_outputs": None,
                "errors": None,                
            }
        }
    }    
    def create_dir(curr_path, dir_dict): 
        for k in dir_dict: 
            dir_path = os.path.join(curr_path, k)
            if os.path.exists(dir_path): shutil.rmtree(dir_path)
            os.mkdir(dir_path)
            if dir_dict[k] != None: 
                create_dir(dir_path, dir_dict[k])

    create_dir("", dir_tree)


def copy_example_files(dataset_name):     
    # data schema
    shutil.copyfile(f"./examples/{dataset_name}_schema.json", os.path.join(data_schema_path, f"{dataset_name}_schema.json"))
    # train data    
    shutil.copyfile(f"./examples/{dataset_name}_train.csv", os.path.join(train_data_path, f"{dataset_name}_train.csv"))    
    # test data     
    shutil.copyfile(f"./examples/{dataset_name}_test.csv", os.path.join(test_data_path, f"{dataset_name}_test.csv"))    
    # hyperparameters
    shutil.copyfile("./examples/hyperparameters.json", os.path.join(hyper_param_path, "hyperparameters.json"))


def run_HPT(num_hpt_trials): 
    # Read data
    train_data = utils.get_data(train_data_path)    
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)  
    # run hyper-parameter tuning. This saves results in each trial, so nothing is returned
    model_tuner.tune_hyperparameters(train_data, data_schema, num_hpt_trials, hyper_param_path, hpt_results_path)


def train_and_save_algo():        
    # Read hyperparameters 
    hyper_parameters = utils.get_hyperparameters(hyper_param_path)    
    # Read data
    train_data = utils.get_data(train_data_path)    
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)  
    # get trained preprocessor, model, training history 
    preprocessor, model = model_trainer.get_trained_model(train_data, data_schema, hyper_parameters)            
    # Save the processing pipeline   
    pipeline.save_preprocessor(preprocessor, model_artifacts_path)
    # Save the model 
    logistic_regression.save_model(model, model_artifacts_path)
    print("done with training")


def load_and_test_algo(): 
    # Read data
    test_data = utils.get_data(test_data_path)   
    # read data config
    data_schema = utils.get_data_schema(data_schema_path)    
    # instantiate the trained model 
    predictor = model_server.ModelServer(model_artifacts_path)
    # make predictions
    predictions = predictor.predict_proba(test_data, data_schema)
    # save predictions
    predictions.to_csv(os.path.join(testing_outputs_path, "test_predictions.csv"), index=False)
    # score the results
    results = score(test_data, predictions)  
    print("done with predictions")
    return results


def set_scoring_vars(dataset_name):
    global target_class, other_class, target_field
    if dataset_name == "cancer": 
        target_class = "M"; other_class = "B"; target_field = "diagnosis"
    elif dataset_name == "credit_card": 
        target_class = "negative"; other_class = "positive"; target_field = "class"
    elif dataset_name == "mushroom": 
        target_class = "p"; other_class = "e"; target_field = "class"
    elif dataset_name == "segment": 
        target_class = "P"; other_class = "N"; target_field = "binaryClass"
    elif dataset_name == "spam": 
        target_class = 1; other_class = 0; target_field = "class"
    elif dataset_name == "telco_churn": 
        target_class = "Yes"; other_class = "No"; target_field = "Churn"
    elif dataset_name == "titanic": 
        target_class = 1; other_class = 0; target_field = "Survived"
    else: raise Exception(f"Error: Cannot find dataset = {dataset_name}")


def score(test_data, predictions): 
    predictions["pred_class"] = predictions.apply(lambda row: 
        target_class if row[target_class] >= 0.5 else other_class, axis=1)    
    
    accu = accuracy_score(test_data[target_field], predictions['pred_class'])    
    f1 = f1_score(test_data[target_field], predictions['pred_class'], pos_label=target_class)    
    precision = precision_score(test_data[target_field], predictions['pred_class'], pos_label=target_class)    
    recall = recall_score(test_data[target_field], predictions['pred_class'], pos_label=target_class)    
    y_true = np.where(test_data[target_field] == target_class, 1., 0.)
    auc_score = roc_auc_score(y_true, predictions[target_class])
    
    results = { 
               "accuracy": np.round(accu,4), 
               "f1_score": np.round(f1, 4), 
               "precision": np.round(precision, 4), 
               "recall": np.round(recall, 4), 
               "auc_score": np.round(auc_score, 4), 
               }
    return results


def save_test_outputs(results, run_hpt, dataset_name):    
    df = pd.DataFrame(results) if dataset_name is None else pd.DataFrame([results])        
    df = df[["model", "dataset_name", "run_hpt", "num_hpt_trials", 
             "accuracy", "f1_score", "precision", "recall", "auc_score",
             "elapsed_time_in_minutes"]]
    
    file_path_and_name = get_file_path_and_name(run_hpt, dataset_name)
    df.to_csv(file_path_and_name, index=False)
    

def get_file_path_and_name(run_hpt, dataset_name): 
    if dataset_name is None: 
        fname = f"_{model_name}_results_with_hpt.csv" if run_hpt else f"_{model_name}_results_no_hpt.csv"
    else: 
        fname = f"{model_name}_{dataset_name}_results_with_hpt.csv" if run_hpt else f"{model_name}_{dataset_name}_results_no_hpt.csv"
    full_path = os.path.join(test_results_path, fname)
    return full_path


def run_train_and_test(dataset_name, run_hpt, num_hpt_trials):
    start = time.time()
    
    create_ml_vol()   # create the directory which imitates the bind mount on container
    copy_example_files(dataset_name)   # copy the required files for model training    
    if run_hpt: run_HPT(num_hpt_trials)               # run HPT and save tuned hyperparameters
    train_and_save_algo()        # train the model and save
    
    set_scoring_vars(dataset_name=dataset_name)
    results = load_and_test_algo()        # load the trained model and get predictions on test data
    
    end = time.time()
    elapsed_time_in_minutes = np.round((end - start)/60.0, 2)
    
    results = { **results, 
               "model": model_name, 
               "dataset_name": dataset_name, 
               "run_hpt": run_hpt, 
               "num_hpt_trials": num_hpt_trials if run_hpt else None, 
               "elapsed_time_in_minutes": elapsed_time_in_minutes 
               }
    
    print(f"Done with dataset in {elapsed_time_in_minutes} minutes.")
    return results 


if __name__ == "__main__": 
    
    num_hpt_trials = 30
    run_hpt_list = [False, True]
    # run_hpt_list = [False]
    
    datasets = ["cancer", "credit_card", "mushroom", "segment", "spam", "telco_churn", "titanic"]
    # datasets = ["cancer"]
    
    for run_hpt in run_hpt_list:
        all_results = []
        for dataset_name in datasets:        
            print("-"*60)
            print(f"Running dataset {dataset_name}")
            results = run_train_and_test(dataset_name, run_hpt, num_hpt_trials)
            save_test_outputs(results, run_hpt, dataset_name)            
            all_results.append(results)
            print("-"*60)
        
        save_test_outputs(all_results, run_hpt, dataset_name=None)