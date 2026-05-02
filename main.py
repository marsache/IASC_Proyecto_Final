# IMPORTS
from config import getSettings 

import pandas as pd
import numpy as np
import os

# from langchain_ollama import ChatOllama
# from langchain.agents import create_agent
# from langchain.tools import tool

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

model = None
dataset = None

#@tool
def describe_dataset() -> str:
    """
    Describe the dataset content

    Returns:
        str: Description of the dataset
    """    
    global dataset
    desc = f"""
Instancias totales : {dataset.shape[0]:,}.
Características          : {dataset.shape[1] - 1}
Valores ausentes: {dataset.isnull().sum()[dataset.isnull().sum() > 0]}
Estadísticas descriptivas por característica: 
{dataset.describe()}
    """

    return desc

def preprocess_dataset(target : str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Cleans the dataset, one-hot-encode the categorical features and split the dataset into train and test.

    Args:
        target (str): Target feature for the ML model.

    Returns:
        - X_train: Training data
        - X_test: Test data
        - Y_train: Training labels
        - Y_test: Test labels
    """
    global dataset

    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop = True)

    is_numerical = np.vectorize(lambda x : np.issubdtype(x, np.number))
    numericals = is_numerical(dataset.dtypes)

    enc = OneHotEncoder()
    for i in range(len(numericals)):
        name = dataset.iloc[:, i].name
        if target is not name and not numericals[i]:
            OHE = enc.fit_transform(dataset[name]).toarray()
            dataset = dataset.drop(name, axis = 1)
            dataset = dataset.reset_index(drop = True)
            dataset = pd.concat([pd.DataFrame(OHE, columns=enc.get_feature_names_out()).reset_index(drop = True), dataset], axis = 1)
        
    X = dataset.drop(target, axis = 1).to_numpy()
    y = dataset[target].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test

def main():
    global model
    global dataset

    settings = getSettings()

    dataset_found = False
    
    while (not dataset_found):

        dataset_path = input("CSV path: ")

        dataset_path = settings.base_dataset_path + dataset_path

        dataset_found = os.path.exists(dataset_path)
        if(not dataset_found):
            print("Path not found")

    dataset = pd.read_csv(dataset_path)

    print(f"Dataset head:\n{dataset.head()}")

    target_found = False
    
    while (not target_found):

        target = input("Target feature: ") #"over_limit"
        
        target_found = target in dataset
        if(not target_found):
            print("Feature not found")

    x_train, x_test, y_train, y_test = preprocess_dataset(target)

    print(f"""
    Train shapes:
        {x_train.shape}
        {y_train.shape}
    Test shapes:
        {x_test.shape}
        {y_test.shape}
""")

    # llm = ChatOllama(model = settings.ollama_model, temperature = settings.temperature)

    # system_prompt = ""

    # try:
    #     with open("system_prompt.txt", "r") as f:
    #         system_prompt = f.read()
    # except IOError as e:
    #     print("Can't find system prompt!")
    #     return
    
    # agent = create_agent(
    #     model = llm,
    #     tools = []
    # )

    # messages = {"messages" : []}

    # exit = False

    # while(not exit):
    #     query = input("- ")
    #     messages["messages"].append({"role" : "user", "content" : query})

    #     response = agent.invoke(messages)["messages"][-1].content

    #     messages["messages"].append({"role" : "assistant", "content" : response})

    #     print(response)

    #     exit = query == "exit"


if __name__ == "__main__":
    main()
