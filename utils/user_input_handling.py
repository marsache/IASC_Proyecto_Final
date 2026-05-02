from config import getSettings 
import os
import pandas as pd

def find_dataset():
    dataset_found = False
    while (not dataset_found):

        dataset_path = input("CSV path: ")
        dataset_path = getSettings().base_dataset_path + dataset_path

        dataset_found = os.path.exists(dataset_path)
        if(not dataset_found):
            print("Path not found")

    return dataset_path

def find_target_feature_in_dataset(dataset: pd.DataFrame) -> str:
    target_found = False
    
    while (not target_found):

        target = input("Target feature: ") #"over_limit"
        
        target_found = target in dataset
        if(not target_found):
            print("Feature not found")
    
    return target