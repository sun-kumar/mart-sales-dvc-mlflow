#importing packages
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer
import category_encoders as ce
from sklearn.preprocessing import StandardScaler
from getting_data import read_params
import argparse
import pandas as pd
from sklearn.model_selection import GridSearchCV


"""pre-processsing step
Dropping unwanted the columns - 
"""
class OutletTypeEncoder(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, documents, y=None):
        return self

    def transform(self, x_dataset):
        x_dataset['outlet_grocery_store'] = (x_dataset['Outlet_Type'] == 'Grocery Store')*1
        x_dataset['outlet_supermarket_3'] = (x_dataset['Outlet_Type'] == 'Supermarket Type3')*1
        x_dataset['outlet_identifier_OUT027'] = (x_dataset['Outlet_Identifier'] == 'OUT027')*1
        
        return x_dataset

def preprocessing(config_path):


    """" Above logic is not correct because it exposed testing data, also any misjudgement can lead to data leakage
    ideal approach is to pass testing data after prediction in training_and_evaluate function
    in that case we need to figure out other logic to pass test data. 
    After splitting, testing data should be touched only after
    predict function so this function should accept dataframe and not only config and it should return also dataframe"""

    """Imputing the missing values in column Item_Weight by mean
    Scaling the data in the column Item_MRP"""
    
    pre_process = ColumnTransformer(remainder='passthrough',
                                    transformers=[('drop_columns', 'drop', ['Item_Identifier',
                                                                            'Outlet_Identifier',
                                                                            'Item_Fat_Content',
                                                                            'Item_Type',
                                                                            'Outlet_Identifier',
                                                                            'Outlet_Size',
                                                                            'Outlet_Location_Type',
                                                                            'Outlet_Type'
                                                                        ]),
                                                ('impute_item_weight', SimpleImputer(strategy='mean'), ['Item_Weight']),
                                                ('scale_data', StandardScaler(),['Item_MRP'])])
    return pre_process


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    preprocessing(config_path=parsed_args.config)