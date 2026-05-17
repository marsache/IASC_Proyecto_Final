import json
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score

# --- Importaciones de Modelos de Clasificación ---
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# --- Importaciones de Modelos de Regresión ---
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

# Importes de Keras para Transfer Learning
from keras.applications import MobileNetV2, ResNet50, VGG16, EfficientNetB0
import tensorflow as tf

from skops.io import load as skops_load, dump
import tensorflow as tf


# Mapea el string que devuelve el LLM a la clase constructora correspondiente
MODEL_REGISTRY = {
    # --- TABULARES: Clasificación ---
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "SVC": SVC,
    
    # --- TABULARES: Regresión ---
    "RandomForestRegressor": RandomForestRegressor,
    "LinearRegression": LinearRegression,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVR": SVR,

    # --- IMÁGENES: Keras / Deep Learning ---
    # Nota: En tu función de entrenamiento, estas clases necesitarán ser tratadas 
    # añadiendo un 'GlobalAveragePooling2D' y una capa 'Dense' con softmax.
    "MobileNetV2": MobileNetV2,
    "ResNet50": ResNet50,
    "VGG16": VGG16,
    "EfficientNetB0": EfficientNetB0,
    "SimpleCNN": "SimpleCNN" # Flag para construir una red convolucional secuencial personalizada
}

def build_and_train_recommended_models(recommendations_json: str, X_train, y_train):
    """
    Lee el JSON del agente, instancia los modelos de Scikit-Learn con los 
    hiperparámetros sugeridos y los entrena.
    """
    try:
        recommendations = json.loads(recommendations_json)
    except json.JSONDecodeError:
        print("Error: El agente no devolvió un JSON válido.")
        return []

    trained_models = []
    
    # Extraemos la lista de modelos del JSON
    models_data = recommendations.get("models", [])
    
    for model_info in models_data:
        model_name = model_info.get("name")
        params = model_info.get("hyperparameters", {})
        
        print(f"\nIntentando instanciar: {model_name}")
        print(f"Parámetros sugeridos por el LLM: {params}")
        
        # 2. Verificar que el modelo sugerido está en nuestro registro
        if model_name not in MODEL_REGISTRY:
            print(f"Advertencia: El modelo '{model_name}' no está soportado o fue alucinado por el LLM. Saltando...")
            continue
            
        model_class = MODEL_REGISTRY[model_name]
        task_type = recommendations.get("task_type", "")
        # Ejemplo mental de cómo deberás procesarlo luego
        if task_type == "image_classification":
            base_model = MODEL_REGISTRY[model_name](weights='imagenet', include_top=False, input_shape=(128, 128, 3))
            
            base_model.trainable = False 
            
            num_classes = len(np.unique(y_train))

            # Construir cabecera
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                
                tf.keras.layers.Dense(num_classes, activation='softmax')
            ])
            
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=params.get('learning_rate', 0.001)),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
            
            model.fit(X_train, y_train, epochs=params.get('epochs', 10), batch_size=params.get('batch_size', 32))
        
            trained_models.append({
                    "name": model_name,
                    "model_object": model,
                    "reasoning": model_info.get("reasoning", "Sin justificación provista.")
                })
        else:
            # Excepción específica para SVC para asegurar que funcione con explicadores locales (XAI)
            if model_name == "SVC":
                params["probability"] = True
                
            # 3. Instanciar y Entrenar
            try:
                # El operador ** desempaqueta el diccionario en argumentos nombrados
                # Equivalente a: RandomForestClassifier(n_estimators=100, max_depth=5)
                model_instance = model_class(**params)
                
                print(f"Entrenando {model_name}...")
                model_instance.fit(X_train, y_train)
                
                # Guardamos un diccionario con el modelo entrenado y su información
                trained_models.append({
                    "name": model_name,
                    "model_object": model_instance,
                    "reasoning": model_info.get("reasoning", "Sin justificación provista.")
                })
                print("Entrenamiento completado.")
                
            except Exception as e:
                # Mecanismo de seguridad: Si el LLM alucina un parámetro que la clase no acepta
                # (ej. sugiere 'learning_rate' para un RandomForest), Scikit-Learn lanzará un error.
                print(f"Error al instanciar {model_name} con parámetros {params}.")
                print(f"Detalle del error: {e}")
                
                print(f"Intentando entrenar {model_name} con parámetros por defecto (Fallback)...")
                try:
                    # Fallback: instanciar sin parámetros
                    model_fallback = model_class()
                    # Forzar probability=True en SVC incluso en el fallback
                    if model_name == "SVC":
                        model_fallback = model_class(probability=True)
                        
                    model_fallback.fit(X_train, y_train)
                    trained_models.append({
                        "name": f"{model_name} (Default Params)",
                        "model_object": model_fallback,
                        "reasoning": "Fallback tras error de parámetros del LLM."
                    })
                    print("Entrenamiento de fallback completado.")
                except Exception as e_fallback:
                    print(f"Fallo crítico en fallback: {e_fallback}")
                    
    return trained_models

def generate_model_info(model, X_test: np.ndarray, y_test: np.ndarray, task_type: str = "classification") -> str:
    """
    Evalúa el modelo entrenado y genera un resumen en texto plano
    con su nombre y métricas clave para inyectar en el LLM.
    """
    # 1. Extraer el nombre del algoritmo automáticamente
    model_name = type(model).__name__
    
    # 2. Generar predicciones sobre el conjunto de prueba
    y_pred = model.predict(X_test)
    
    # 3. Construir el texto de salida
    info_lines = [f"Tipo de Modelo Base: {model_name}"]
    info_lines.append("Métricas de Rendimiento (Evaluación en Test):")
    
    # 4. Calcular métricas según el tipo de problema
    if task_type == "classification":
        acc = accuracy_score(y_test, y_pred)
        # Usamos average='weighted' por si hay desbalanceo multiclase
        f1 = f1_score(y_test, y_pred, average='weighted')
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec = recall_score(y_test, y_pred, average='weighted')
        
        info_lines.append(f"- Exactitud (Accuracy): {acc:.2f} (Rango 0-1, donde 1 es perfecto)")
        info_lines.append(f"- F1-Score (Ponderado): {f1:.2f}")
        info_lines.append(f"- Precisión: {prec:.2f}")
        info_lines.append(f"- Recall: {rec:.2f}")
        
    elif task_type == "regression":
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        info_lines.append(f"- Error Cuadrático Medio (MSE): {mse:.2f}")
        info_lines.append(f"- R2 Score: {r2:.2f} (Rango de 0-1, donde 1 es ajuste perfecto)")
        
    else:
        info_lines.append("- Tipo de tarea no reconocida para calcular métricas.")

    # Convertimos la lista en un solo string con saltos de línea
    return "\n".join(info_lines), y_pred

def try_load_model(dataset_path: str):
    dataset_dir = os.path.dirname(dataset_path)
    dataset_filename = os.path.basename(dataset_path)
    dataset_name = os.path.splitext(dataset_filename)[0] # Quita la extensión
    
    metadata_file_path = os.path.join(dataset_dir, f"{dataset_name}_metadata_model.json")
    
    # 1. Si no existe el JSON de metadatos, no podemos saber qué modelo es
    if not os.path.exists(metadata_file_path):
        return False, None, ""
        
    # 2. Leer primero los metadatos
    try:
        with open(metadata_file_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error al leer los metadatos del modelo: {e}")
        return False, None, ""
        
    task_type = metadata.get("task_type", "")
    
    # 3. Determinar el nombre del archivo del modelo.
    # Usamos la clave 'model_file' que guardamos en la función save_model.
    # Si por compatibilidad con versiones anteriores no existe, lo deducimos.
    model_filename = metadata.get("model_file")
    if not model_filename:
        extension = ".keras" if task_type == "image_classification" else ".skops"
        model_filename = f"{dataset_name}_model{extension}"
        
    model_file_path = os.path.join(dataset_dir, model_filename)
    
    # 4. Comprobar que el archivo del modelo realmente existe
    if not os.path.exists(model_file_path):
        return False, None, ""
        
    # 5. Cargar el modelo con la librería correcta según el task_type
    try:
        if task_type == "image_classification":
            # Modelo Keras / Deep Learning
            model = tf.keras.models.load_model(model_file_path)
        else:
            # Modelo Scikit-Learn tabular clásico
            # Nota: 'trusted' debe configurarse según tus necesidades de seguridad
            model = skops_load(model_file_path, trusted=[]) 
            
        return True, model, task_type
        
    except Exception as e:
        print(f"Error al cargar el archivo del modelo: {e}")
        return False, None, ""

def save_model(dataset_path: str, task_type: str, model):
    dataset_dir = os.path.dirname(dataset_path)
    dataset_filename = os.path.basename(dataset_path)
    dataset_name = os.path.splitext(dataset_filename)[0] # Quita la extensión (.csv, .zip, etc.)
    
    # 1. Determinar si es un modelo de imágenes (Keras) o tabular (Scikit-Learn)
    is_image_task = (task_type == "image_classification")
    
    # 2. Asignar la extensión correcta
    extension = ".keras" if is_image_task else ".skops"
    model_filename = f"{dataset_name}_model{extension}"
    
    model_file_path = os.path.join(dataset_dir, model_filename)
    metadata_file_path = os.path.join(dataset_dir, f"{dataset_name}_metadata_model.json")

    # 3. Ampliar los metadatos para incluir el nombre del archivo del modelo
    # (Esto facilitará mucho la carga posterior)
    model_metadata = {
        "task_type": task_type,
        "model_type": str(type(model)),
        "model_file": model_filename 
    }

    # 4. Guardar los metadatos JSON
    with open(metadata_file_path, 'w', encoding='utf-8') as f:
        json.dump(model_metadata, f, indent=4, ensure_ascii=False)

    # 5. Guardar el modelo usando la librería correspondiente
    if is_image_task:
        # Keras tiene su propio método integrado para guardarse
        model.save(model_file_path)
    else:
        # Scikit-Learn usa skops (o joblib/pickle)
        dump(model, model_file_path)

# # ==========================================
# # FUNCIONES DE ENTRENAMIENTO: CLASIFICACIÓN
# # ==========================================

# def train_random_forest_classifier(x_train, y_train):
#     model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
#     model.fit(x_train, y_train)
#     return model

# def train_logistic_regression(x_train, y_train):
#     # max_iter se aumenta para asegurar convergencia en datasets complejos
#     model = LogisticRegression(max_iter=1000, random_state=42)
#     model.fit(x_train, y_train)
#     return model

# def train_gradient_boosting_classifier(x_train, y_train):
#     model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
#     model.fit(x_train, y_train)
#     return model

# def train_svc_classifier(x_train, y_train):
#     # probability=True es indispensable para herramientas XAI que usan predict_proba()
#     model = SVC(kernel='rbf', probability=True, random_state=42)
#     model.fit(x_train, y_train)
#     return model


# # ==========================================
# # FUNCIONES DE ENTRENAMIENTO: REGRESIÓN
# # ==========================================

# def train_random_forest_regressor(x_train, y_train):
#     model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
#     model.fit(x_train, y_train)
#     return model

# def train_linear_regression(x_train, y_train):
#     model = LinearRegression()
#     model.fit(x_train, y_train)
#     return model

# def train_gradient_boosting_regressor(x_train, y_train):
#     model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
#     model.fit(x_train, y_train)
#     return model

# def train_svr_regressor(x_train, y_train):
#     model = SVR(kernel='rbf')
#     model.fit(x_train, y_train)
#     return model