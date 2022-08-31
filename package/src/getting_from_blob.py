import os
import uuid
import sys
import argparse
from getting_data import read_params
import os
from azure.storage.blob import  BlobServiceClient
import datetime

def download_data(config_path):
   config = read_params(config_path)
   storageaccountname = config['access_control']['account_name'] 
   connect_str = config['access_control']['connection_string']
   cont_name = config['access_control']['container_name']
   
   raw_data_path = config['load_data']['raw_dataset_csv']
   blob_service_client = BlobServiceClient.from_connection_string(conn_str=connect_str)
   # create container client
   container_client = blob_service_client.get_container_client(cont_name)
   # code below lists all the blobs in the container and downloads them one after another
   blob_list = container_client.list_blobs()
   for blob in blob_list:
       
       # check if the path contains a folder structure, create the folder structure
       if "/" in "{}".format(blob.name):
           #extract the folder path and check if that folder exists locally, and if not create it
           head, tail = os.path.split("{}".format(blob.name))
           if not (os.path.isdir(raw_data_path+ "/" + head)):
               #create the diretcory and download the file to it
               """ This logic needed development bases on what criteria we want to create new folder if is is not already present"""
               os.makedirs(raw_data_path+ "/" + head, exist_ok=True)
           else:
               print("Dumping data into existing directory")
               blob_client = container_client.get_blob_client(blob.name)
               with open(raw_data_path+ "/" + head + "/" + tail, "wb") as my_blob:
                   my_blob.write(container_client.download_blob(blob.name).readall())
                   my_blob.close()
   return


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    download_data(config_path=parsed_args.config) 
