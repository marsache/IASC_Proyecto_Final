# import pandas as pd
# import numpy as np

# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.model_selection import train_test_split

# def preprocess_dataset(dataset : pd.DataFrame, target : str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     """
#     Cleans the dataset, one-hot-encode the categorical features and split the dataset into train and test.

#     Args:
#         target (str): Target feature for the ML model.

#     Returns:
#         - X_train: Training data
#         - X_test: Test data
#         - Y_train: Training labels
#         - Y_test: Test labels
#     """
#     dataset = dataset.dropna()
#     dataset = dataset.reset_index(drop = True)

#     is_numerical = np.vectorize(lambda x : np.issubdtype(x, np.number))
#     numericals = is_numerical(dataset.dtypes)

#     enc = OneHotEncoder()
#     for i in range(len(numericals)):
#         name = dataset.iloc[:, i].name
#         if target is not name and not numericals[i]:
#             OHE = enc.fit_transform(dataset[name]).toarray()
#             dataset = dataset.drop(name, axis = 1)
#             dataset = dataset.reset_index(drop = True)
#             dataset = pd.concat([pd.DataFrame(OHE, columns=enc.get_feature_names_out()).reset_index(drop = True), dataset], axis = 1)
        
#     X = dataset.drop(target, axis = 1).to_numpy()
#     y = dataset[target].to_numpy()

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     return X_train, X_test, y_train, y_test

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np
import json
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def generate_dataset_profile(df: pd.DataFrame, target: str) -> str:
    """
    Agente Perfilador: Extrae estadísticas crudas y usa Llama 3 en Ollama 
    para generar un diccionario de metadatos semántico en JSON.
    """
    # 1. Extraer estadísticas básicas con Pandas
    numericals_stats = df.select_dtypes(include=[np.number]).describe().to_dict()
    
    categoricals_stats = {}
    for col in df.select_dtypes(exclude=[np.number]).columns:
        if col != target:
            categoricals_stats[col] = df[col].unique().tolist()
            
    target_info = {
        "name": target,
        "type": str(df[target].dtype),
        "unique_values": df[target].unique().tolist()
    }
    # 2. Configurar el Output Parser para forzar y parsear el JSON
    parser = JsonOutputParser()

    # 3. Crear el Prompt Template estilo LangChain
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Eres un Analista de Datos experto.
        Tu tarea es analizar las estadísticas de un dataset y generar un diccionario de datos semántico.
        
        {format_instructions}"""),
        ("human", """ESTADÍSTICAS DEL DATASET:
        - Target (Variable a predecir): {target_info}
        - Variables Numéricas (Resumen): {numericals_stats}
        - Variables Categóricas (Ejemplos): {categoricals_stats}""")
    ])

    # 4. Instanciar el modelo LLM mediante ChatOllama
    # Nota: Le pasamos format="json" para activar el modo JSON nativo de Ollama
    llm = ChatOllama(model="llama3", temperature=0, format="json")

    # 5. Componer la cadena (Pipeline)
    chain = prompt | llm | parser

    # 6. Ejecutar la cadena
    try:
        response_dict = chain.invoke({
            "target_info": json.dumps(target_info),
            "numericals_stats": json.dumps(numericals_stats),
            "categoricals_stats": json.dumps(categoricals_stats),
            "format_instructions": """Devuelve ÚNICAMENTE un objeto JSON válido con esta estructura exacta:
            {
                "dataset_description": "Breve descripción inferida",
                "target_description": "Qué significa la variable objetivo",
                "features": {
                    "nombre_de_columna": "Descripción semántica y valores típicos"
                }
            }"""
        })
        # LangChain devuelve un diccionario de Python gracias al JsonOutputParser. 
        # Lo convertimos a string JSON para mantener la compatibilidad con el resto de tu código.
        return json.dumps(response_dict) 
        
    except Exception as e:
        print(f"Error ejecutando la cadena de LangChain con Ollama: {e}")
        return "{}"
    
def preprocess_dataset(dataset: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Cleans the dataset, generates semantic profile, one-hot-encodes categorical features, 
    and splits the dataset into train and test.
    """
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)

    # --- NUEVO: Generar el perfil antes de transformar los datos ---
    dataset_metadata_json = generate_dataset_profile(dataset, target)

    print(dataset_metadata_json)
    # ---------------------------------------------------------------

    is_numerical = np.vectorize(lambda x: np.issubdtype(x, np.number))
    numericals = is_numerical(dataset.dtypes)

    enc = OneHotEncoder(sparse_output=False) # sparse_output=False reemplaza toarray()
    
    # Usamos list(dataset.columns) para evitar problemas al mutar el df en el bucle
    columns = list(dataset.columns) 
    for i, name in enumerate(columns):
        if target != name and not numericals[i]: # Corrección: '!=' en lugar de 'is not'
            # OneHotEncoder requiere un array 2D, usamos dataset[[name]]
            OHE = enc.fit_transform(dataset[[name]])
            dataset = dataset.drop(name, axis=1)
            ohe_df = pd.DataFrame(OHE, columns=enc.get_feature_names_out([name]))
            dataset = pd.concat([dataset, ohe_df], axis=1)
            
    X = dataset.drop(target, axis=1).to_numpy()
    y = dataset[target].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Devolvemos el metadato como 5to elemento para guardarlo o pasarlo al otro agente
    return X_train, X_test, y_train, y_test, dataset_metadata_json