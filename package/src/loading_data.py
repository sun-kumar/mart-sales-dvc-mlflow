# reading the data from data source
# save it in the data/raw for further process
import os
from getting_data import read_params, get_data
import argparse

def load_and_save(config_path):
    config = read_params(config_path)
    df = get_data(config_path)
    # eg: new_cols = [col.replace(" ", "_") for col in df.columns] 
    """Any other operation  on column name if required 
    needed on training data coming each cycle will be be done here"""
    raw_data_path = config["load_data"]["raw_dataset_csv"]
    df.to_csv(raw_data_path,index=False) # attribute header is a must incase of column name transformation in case above eg: header=new_cols


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_save(config_path=parsed_args.config)  
    
