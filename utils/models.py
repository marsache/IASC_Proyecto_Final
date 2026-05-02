import json
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

# 1. Crear el Registro de Modelos Permitidos
# Mapea el string que devuelve el LLM a la clase constructora de Scikit-Learn
MODEL_REGISTRY = {
    # Clasificación
    "RandomForestClassifier": RandomForestClassifier,
    "LogisticRegression": LogisticRegression,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "SVC": SVC,
    # Regresión
    "RandomForestRegressor": RandomForestRegressor,
    "LinearRegression": LinearRegression,
    "GradientBoostingRegressor": GradientBoostingRegressor,
    "SVR": SVR
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
    return "\n".join(info_lines)


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