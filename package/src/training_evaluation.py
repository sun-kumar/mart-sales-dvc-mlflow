# loading the training and testing data
# training algorithm
# saving the evaluation metrices and hyperparameter
from operator import index
import os
import warnings
import sys
import pandas as pd
import numpy as np
import argparse
import joblib
import json
from urllib.parse import urlparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from getting_data import read_params
from preprocessing_transformations import preprocessing,OutletTypeEncoder
from sklearn.model_selection import GridSearchCV
import mlflow



def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    

    max_depth = config["estimators"]["random_forest_regressor"]["params"]["max_depth"]
    min_samples_leaf = config["estimators"]["random_forest_regressor"]["params"]["min_samples_leaf"]
    parameteres = {"random_forest__max_depth":max_depth, "random_forest__min_samples_leaf":min_samples_leaf}

    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path)
    test = pd.read_csv(test_data_path)

    train_y = train[target].squeeze()
    test_y = test[target].squeeze()

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    pre_process = preprocessing(config_path)
    
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])
    
    
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
    
        rfr = RandomForestRegressor()

        model_pipeline = Pipeline(steps=[('get_outlet_binary_columns', OutletTypeEncoder()),('pre_processing',pre_process),
                                            ('random_forest', rfr)
                                            ])
        grid = GridSearchCV(model_pipeline, param_grid=parameteres, cv=5)

        
        grid.fit(train_x, train_y)

        predicted_consumption = grid.predict(test_x)
        
        (rmse, mae, r2) = eval_metrics(test_y, predicted_consumption)
        best_max_depth=grid.best_params_['random_forest__max_depth']
        best_min_samples = grid.best_params_['random_forest__min_samples_leaf']

        print("Random Forest Model (max_depth=%f, min_samples_leaf=%f):" % (best_max_depth,
                                                                            best_min_samples))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        
        mlflow.log_param("max_depth",best_max_depth)
        mlflow.log_param("min_samples_leaf", best_min_samples)

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)
        mlflow.end_run()
        
        

    #####################################################
        scores_file = config["reports"]["scores"]
        params_file = config["reports"]["params"]

        with open(scores_file, "w") as f:
            scores = {
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            json.dump(scores, f, indent=4)

        with open(params_file, "w") as f:
            params = {
                "max_depth": best_max_depth,
                "min_samples_leaf": best_min_samples,
            }
            json.dump(params, f, indent=4)
    #####################################################


        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.joblib")

        joblib.dump(grid, model_path)
        
        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(grid, "model", 
                    registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(grid, "model")
        
        #####################################################

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
                                                                                                                                         
# Why DVC Remote file name is changed and is not conserved after DVC Push?
## Dvc remote is a content-based storage, so names are not preserved.
#  Dvc creates metafiles (*.dvc files) in your workspace that contain names and those files are usually tracked by git,
#  so you need to use git remote and dvc remote together to have both filenames and their contents.
#  Here is a more detailed explanation about the format of local and remote storage:
#  https://dvc.org/doc/user-guide/project-structure/internal-files#structure-of-the-cache-directory .
#  Also, checkout https://dvc.org/doc/use-cases/sharing-data-and-model-files


# To Do


# Post Demo Learning:
# https://discuss.dvc.org/t/dvc-integration-with-azure-ml-pipeline-and-versioning-iot-data/364
# https://towardsdatascience.com/large-data-versioning-with-dvc-and-azure-blob-storage-a-complete-guide-b97344827c81
# https://github.com/iterative/dvc/issues/2200
# https://anno-ai.medium.com/mlops-and-data-managing-large-ml-datasets-with-dvc-and-s3-part-1-d5b8f2fb8280
# https://azure.microsoft.com/en-in/services/storage/blobs/
# https://www.analyticsvidhya.com/blog/2021/06/mlops-tracking-ml-experiments-with-data-version-control/
# https://neptune.ai/blog/azure-ml-alternatives-for-mlops


#################################################################

