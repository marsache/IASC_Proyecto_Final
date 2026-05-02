from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

import os
import pandas as pd
import numpy as np
import json

from agents.dataset_profiler_agent import generate_dataset_profile

def preprocess_dataset(dataset: pd.DataFrame, dataset_path: str, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Cleans the dataset, generates semantic profile, one-hot-encodes categorical features, 
    and splits the dataset into train and test.
    """
    dataset = dataset.dropna()
    dataset = dataset.reset_index(drop=True)

    # --- NUEVO: Generar el perfil antes de transformar los datos ---
    # dataset_metadata_json = generate_dataset_profile(dataset, target)

    # print(dataset_metadata_json)
    # ---------------------------------------------------------------
    # 1. Extraer el directorio y el nombre base del archivo
    dataset_dir = os.path.dirname(dataset_path)
    dataset_filename = os.path.basename(dataset_path)
    dataset_name = os.path.splitext(dataset_filename)[0] # Quita la extensión (.csv, .xlsx, etc.)
    
    # 2. Construir la ruta esperada para el JSON
    metadata_file_path = os.path.join(dataset_dir, f"{dataset_name}_metadata.json")
    
    # 3. Comprobar si existe o generarlo
    if os.path.exists(metadata_file_path):
        print(f"[*] Cargando metadatos cacheados desde: {metadata_file_path}")
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            dataset_metadata_json = f.read()
    else:
        print("[*] Metadatos no encontrados. Generando perfil con el Agente Perfilador...")
        dataset_metadata_json = generate_dataset_profile(dataset, target)
        
        # --- NUEVO: Añadir 'dataset_name' al JSON generado ---
        try:
            # Convertimos el string a diccionario
            metadata_dict = json.loads(dataset_metadata_json)
            # Inyectamos el nombre del dataset
            metadata_dict["dataset_name"] = dataset_name
            # Volvemos a convertirlo a string JSON formateado
            dataset_metadata_json = json.dumps(metadata_dict, indent=4, ensure_ascii=False)
        except json.JSONDecodeError:
            print("[!] Advertencia: El Agente Perfilador no devolvió un JSON válido. Guardando como texto crudo.")
        # -----------------------------------------------------

        # 4. Guardar el resultado en la misma carpeta para la próxima vez
        with open(metadata_file_path, 'w', encoding='utf-8') as f:
            f.write(dataset_metadata_json)
        print(f"[*] Metadatos guardados exitosamente en: {metadata_file_path}")
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
    return X_train, X_test, y_train, y_test, dataset_metadata_json, scaler

def descale_x(row: np.ndarray, scaler, feature_names: list) -> dict:
    """
    Toma el array estandarizado del cliente, revierte el escalado a sus valores 
    originales de negocio y lo convierte en un diccionario legible.
    """
    # inverse_transform espera un array 2D, así que hacemos un reshape temporal
    descaled_x = scaler.inverse_transform(row.reshape(1, -1))[0]
    
    # Redondeamos a 2 decimales para que el LLM no reciba números con 15 decimales
    descaled_x = np.round(descaled_x, 2)
    
    # Emparejamos los nombres de las columnas con sus valores reales
    descaled_x_dict = dict(zip(feature_names, descaled_x))
    
    return descaled_x_dict