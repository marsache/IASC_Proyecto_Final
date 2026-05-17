# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.model_selection import train_test_split

# import os
# import pandas as pd
# import numpy as np
# import json
# import random

# from agents.dataset_profiler_agent import generate_dataset_profile

# def preprocess_dataset(dataset: pd.DataFrame, dataset_path: str, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
#     """
#     Cleans the dataset, generates semantic profile, one-hot-encodes categorical features, 
#     and splits the dataset into train and test.
#     """
#     dataset = dataset.dropna()
#     dataset = dataset.reset_index(drop=True)

#     # --- NUEVO: Generar el perfil antes de transformar los datos ---
#     # dataset_metadata_json = generate_dataset_profile(dataset, target)

#     # print(dataset_metadata_json)
#     # ---------------------------------------------------------------
#     # 1. Extraer el directorio y el nombre base del archivo
#     dataset_dir = os.path.dirname(dataset_path)
#     dataset_filename = os.path.basename(dataset_path)
#     dataset_name = os.path.splitext(dataset_filename)[0] # Quita la extensión (.csv, .xlsx, etc.)
    
#     # 2. Construir la ruta esperada para el JSON
#     metadata_file_path = os.path.join(dataset_dir, f"{dataset_name}_metadata.json")
    
#     # 3. Comprobar si existe o generarlo
#     if os.path.exists(metadata_file_path):
#         print(f"[*] Cargando metadatos cacheados desde: {metadata_file_path}")
#         with open(metadata_file_path, 'r', encoding='utf-8') as f:
#             dataset_metadata_json = f.read()
#     else:
#         print("[*] Metadatos no encontrados. Generando perfil con el Agente Perfilador...")
#         dataset_metadata_json = generate_dataset_profile(dataset, target)
        
#         # --- NUEVO: Añadir 'dataset_name' al JSON generado ---
#         try:
#             # Convertimos el string a diccionario
#             metadata_dict = json.loads(dataset_metadata_json)
#             # Inyectamos el nombre del dataset
#             metadata_dict["dataset_name"] = dataset_name
#             # Volvemos a convertirlo a string JSON formateado
#             dataset_metadata_json = json.dumps(metadata_dict, indent=4, ensure_ascii=False)
#         except json.JSONDecodeError:
#             print("[!] Advertencia: El Agente Perfilador no devolvió un JSON válido. Guardando como texto crudo.")
#         # -----------------------------------------------------

#         # 4. Guardar el resultado en la misma carpeta para la próxima vez
#         with open(metadata_file_path, 'w', encoding='utf-8') as f:
#             f.write(dataset_metadata_json)
#         print(f"[*] Metadatos guardados exitosamente en: {metadata_file_path}")
#     # ---------------------------------------------------------------

#     is_numerical = np.vectorize(lambda x: np.issubdtype(x, np.number))
#     numericals = is_numerical(dataset.dtypes)

#     enc = OneHotEncoder(sparse_output=False) # sparse_output=False reemplaza toarray()
    
#     # Usamos list(dataset.columns) para evitar problemas al mutar el df en el bucle
#     columns = list(dataset.columns) 
#     for i, name in enumerate(columns):
#         if target != name and not numericals[i]: # Corrección: '!=' en lugar de 'is not'
#             # OneHotEncoder requiere un array 2D, usamos dataset[[name]]
#             OHE = enc.fit_transform(dataset[[name]])
#             dataset = dataset.drop(name, axis=1)
#             ohe_df = pd.DataFrame(OHE, columns=enc.get_feature_names_out([name]))
#             dataset = pd.concat([dataset, ohe_df], axis=1)
            
#     X = dataset.drop(target, axis=1).to_numpy()
#     y = dataset[target].to_numpy()

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     # Devolvemos el metadato como 5to elemento para guardarlo o pasarlo al otro agente
#     return X_train, X_test, y_train, y_test, dataset_metadata_json, scaler, dataset

# def get_random_row(dataset: np.ndarray, dataset_metadata, target):
#     """
#     Selecciona una fila aleatoria de un array de NumPy (ej. x_test) y la mapea
#     a un diccionario usando los nombres de las features del metadata.
#     """
#     # 1. Seleccionamos una fila aleatoria del array de test
#     #sample = random.choice(dataset)
#     i = random.randrange(0, dataset.shape[0])
#     sample = dataset[i]

    
#     # 2. Extraemos los nombres de las columnas
#     labels = list(dataset_metadata['features'].keys())
    
#     # 3. zip() empareja automáticamente (label1, valor1), (label2, valor2)...
#     sample_test = {}

#     sample_test["sample"] = dict(zip(labels, sample))

#     sample_test["target"] = target[i]

#     return sample_test

# def descale_x(row: np.ndarray, scaler) -> dict:
#     """
#     Toma un diccionario con los valores estandarizados del cliente, revierte 
#     el escalado a sus valores originales de negocio y devuelve un nuevo diccionario.
#     """
#     # 1. Extraemos las etiquetas y los valores estandarizados del diccionario
#     feature_names = list(row.keys())
#     standardized_values = np.array(list(row.values()))
    
#     # 2. inverse_transform espera un array 2D, así que hacemos un reshape temporal
#     descaled_values = scaler.inverse_transform(standardized_values.reshape(1, -1))[0]
    
#     # 3. Redondeamos a 2 decimales para que el LLM reciba números limpios
#     descaled_values = np.round(descaled_values, 2)
    
#     # 4. Volvemos a emparejar los nombres de las columnas con sus valores reales
#     descaled_x_dict = dict(zip(feature_names, descaled_values))
    
#     return descaled_x_dict

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import os
import pandas as pd
import numpy as np
import json
import random
from pathlib import Path

from agents.dataset_profiler_agent import generate_dataset_profile
from utils.user_input_handling import find_target_feature_in_dataset

from PIL import Image


def load_any_dataset(path_str: str):
    path = Path(path_str)
    
    # 1. Verificar si la ruta existe
    if not path.exists():
        raise FileNotFoundError(f"La ruta no existe: {path_str}")

    # --- ESCENARIO A: ES UN DIRECTORIO (Dataset de Imágenes) ---
    if path.is_dir():
        print(f"[*] Detectado directorio. Procesando como dataset de imágenes: {path.name}")
        image_data = []
        # Extensiones comunes de imagen
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
        
        # Iterar por subcarpetas (cada una es una clase)
        for class_folder in path.iterdir():
            if class_folder.is_dir():
                label = class_folder.name
                for img_file in class_folder.iterdir():
                    if img_file.suffix.lower() in valid_extensions:
                        image_data.append({
                            "file_path": str(img_file),
                            "label": label
                        })
        
        if not image_data:
            return None, "Directorio vacío o sin imágenes válidas."
            
        df_images = pd.DataFrame(image_data)
        print(f"[V] Dataset de imágenes creado. Clases encontradas: {df_images['label'].unique()}")
        return df_images, "image"

    # --- ESCENARIO B: ES UN ARCHIVO (Dataset Tabular) ---
    elif path.is_file():
        print(f"[*] Detectado archivo. Identificando formato: {path.suffix}")
        ext = path.suffix.lower()
        
        try:
            if ext == '.csv':
                df = pd.read_csv(path)
            elif ext in ['.xls', '.xlsx']:
                df = pd.read_excel(path)
            elif ext == '.json':
                df = pd.read_json(path)
            elif ext == '.parquet':
                df = pd.read_parquet(path)
            elif ext in ['.txt', '.tsv']:
                df = pd.read_csv(path, sep=None, engine='python') # Detecta separador automáticamente
            else:
                return None, f"Formato de archivo no soportado: {ext}"
            
            print(f"[V] Archivo tabular cargado exitosamente. Shape: {df.shape}")
            return df, "tabular"
            
        except Exception as e:
            return None, f"Error al leer el archivo: {str(e)}"

    return None, "Tipo de ruta desconocido"

def preprocess_dataset(dataset_path: str, target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Cleans the dataset, generates semantic profile, one-hot-encodes categorical features, 
    and splits the dataset into train and test.
    """

    dataset, dataset_type = load_any_dataset(dataset_path)

    if dataset_type == "tabular":
            
        print(dataset.head(5))

        dataset = dataset.dropna()
        dataset = dataset.reset_index(drop=True)

        # ---------------------------------------------------------------
        # 1. Extraer el directorio y el nombre base del archivo
        dataset_dir = os.path.dirname(dataset_path)
        dataset_filename = os.path.basename(dataset_path)
        dataset_name = os.path.splitext(dataset_filename)[0] # Quita la extensión (.csv, .xlsx, etc.)
        
        # 2. Construir la ruta esperada para el JSON
        metadata_file_path = os.path.join(dataset_dir, f"{dataset_name}_metadata.json")
        
        metadata_dict = {}
        # 3. Comprobar si existe o generarlo
        if os.path.exists(metadata_file_path):
            print(f"[*] Cargando metadatos cacheados desde: {metadata_file_path}")
            with open(metadata_file_path, 'r', encoding='utf-8') as f:
                dataset_metadata_json = f.read()
            metadata_dict = json.loads(dataset_metadata_json)
        else:
            print("[*] Metadatos no encontrados. Generando perfil con el Agente Perfilador...")
            dataset_metadata_json = generate_dataset_profile(dataset, target)
            metadata_dict = json.loads(dataset_metadata_json)
            
            # --- NUEVO: Añadir 'dataset_name' al JSON generado ---
            try:
                # Convertimos el string a diccionario
                # Inyectamos el nombre del dataset
                metadata_dict["dataset_name"] = dataset_name
                # Volvemos a convertirlo a string JSON formateado
            except json.JSONDecodeError:
                print("[!] Advertencia: El Agente Perfilador no devolvió un JSON válido. Guardando como texto crudo.")
            # -----------------------------------------------------

            # 4. Guardar el resultado en la misma carpeta para la próxima vez
            with open(metadata_file_path, 'w', encoding='utf-8') as f:
                f.write(dataset_metadata_json)
            print(f"[*] Metadatos guardados exitosamente en: {metadata_file_path}")
        # ---------------------------------------------------------------

        metadata_dict["dataset_type"] = "tabular"
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
        
        dataset_metadata_json = json.dumps(metadata_dict, indent=4, ensure_ascii=False)

        # Devolvemos el metadato como 5to elemento para guardarlo o pasarlo al otro agente
        return X_train, X_test, y_train, y_test, dataset_metadata_json, scaler, target, dataset
    elif dataset_type == "image":
        print("[*] Procesando dataset de imágenes...")
        
        # 1. Extraer rutas (X) y etiquetas en texto
        X_paths = dataset['file_path'].to_numpy()
        y_strings = dataset['label'].to_numpy()

        # 2. 🔥 NUEVO: Convertir strings de las clases a enteros (0, 1, 2...)
        label_encoder = LabelEncoder()
        y_labels = label_encoder.fit_transform(y_strings)
        
        # Guardamos el mapeo de clases (ej. ['gato', 'perro']) para los metadatos
        detected_classes = list(label_encoder.classes_)

        # 3. Dividir rutas y etiquetas ya numéricas en Train y Test
        X_train_paths, X_test_paths, y_train, y_test = train_test_split(
            X_paths, y_labels, test_size=0.2, random_state=42, stratify=y_labels
        )

        # 4. Función auxiliar para cargar, redimensionar y normalizar
        target_size = (128, 128) 
        print(f"[*] Cargando y redimensionando imágenes a {target_size}...")
        
        def load_images_from_paths(paths):
            images_list = []
            for path in paths:
                img = Image.open(path).convert('RGB')
                img = img.resize(target_size)
                img_array = np.array(img, dtype=np.float32) / 255.0
                images_list.append(img_array)
            return np.array(images_list)

        X_train = load_images_from_paths(X_train_paths)
        X_test = load_images_from_paths(X_test_paths)

        # 5. Ajustar variables de retorno con las clases reales
        metadata_dict = {
            "dataset_type": "image",
            "classes": detected_classes,  # Guardamos los nombres reales de las carpetas
            "image_shape": target_size + (3,), 
            "num_train": len(X_train),
            "num_test": len(X_test)
        }
        dataset_metadata_json = json.dumps(metadata_dict, indent=4, ensure_ascii=False)
        
        scaler = None 
        target = "label"

        return X_train, X_test, y_train, y_test, dataset_metadata_json, scaler, target, dataset

    else:
        raise ValueError(f"[!] Tipo de dataset no soportado: {dataset_type}")

def get_random_row(dataset: np.ndarray, dataset_metadata):
    """
    Selecciona una fila aleatoria de un array de NumPy (ej. x_test) y la mapea
    a un diccionario usando los nombres de las features del metadata.
    """
    # 1. Seleccionamos una fila aleatoria del array de test
    sample = random.choice(dataset)
    
    # 2. Extraemos los nombres de las columnas
    labels = list(dataset_metadata['features'].keys())
    
    # 3. zip() empareja automáticamente (label1, valor1), (label2, valor2)...
    sample_test = dict(zip(labels, sample))

    return sample_test

def descale_x(row: np.ndarray, scaler) -> dict:
    """
    Toma un diccionario con los valores estandarizados del cliente, revierte 
    el escalado a sus valores originales de negocio y devuelve un nuevo diccionario.
    """
    # 1. Extraemos las etiquetas y los valores estandarizados del diccionario
    feature_names = list(row.keys())
    standardized_values = np.array(list(row.values()))
    
    # 2. inverse_transform espera un array 2D, así que hacemos un reshape temporal
    descaled_values = scaler.inverse_transform(standardized_values.reshape(1, -1))[0]
    
    # 3. Redondeamos a 2 decimales para que el LLM reciba números limpios
    descaled_values = np.round(descaled_values, 2)
    
    # 4. Volvemos a emparejar los nombres de las columnas con sus valores reales
    descaled_x_dict = dict(zip(feature_names, descaled_values))
    
    return descaled_x_dict