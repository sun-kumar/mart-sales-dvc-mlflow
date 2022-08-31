# importing packages
import os
import argparse
import yaml
import logging
import pandas as pd

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_data(config_path):
    config = read_params(config_path)
    # data_path = config["data_source"]["az_source"] # in real case scenario
    data_path = config["load_data"]["raw_dataset_csv"] # replace with above in real case scenario
    clientname = config["base"]["clientname"]
    df = pd.read_csv(data_path + "/" + clientname +"/"+"raw.csv")
    return df

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    data = get_data(config_path=parsed_args.config)
