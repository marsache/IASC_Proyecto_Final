import json

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Mapea el string que devuelve el LLM a la clase constructora correspondiente
TABULAR_REGISTRY = [
    # --- TABULARES: Clasificación ---
    "RandomForestClassifier",
    "LogisticRegression",
    "GradientBoostingClassifier",
    "SVC",
    
    # --- TABULARES: Regresión ---
    "RandomForestRegressor",
    "LinearRegression",
    "GradientBoostingRegressor",
    "SVR",
]

IMAGE_REGISTRY = [
        # --- IMÁGENES: Keras / Deep Learning ---
    # Nota: En tu función de entrenamiento, estas clases necesitarán ser tratadas 
    # añadiendo un 'GlobalAveragePooling2D' y una capa 'Dense' con softmax.
    "MobileNetV2",
    "ResNet50",
    "VGG16",
    "EfficientNetB0",
    "SimpleCNN"
]

def recommend_best_models(dataset_metadata_json: str) -> str:
    """
    Agente Enrutador y Selectores: Analiza los metadatos del dataset, 
    detecta el tipo y deriva la tarea a un agente especialista (Tabular o Imagen)
    para devolver los 2 mejores modelos/arquitecturas.
    """
    parser = JsonOutputParser()
    llm = ChatOllama(model="llama3", temperature=0, format="json")

    # 1. Leer los metadatos para tomar la decisión de enrutamiento
    metadata = json.loads(dataset_metadata_json)
    dataset_type = metadata.get("dataset_type", "tabular") # Por defecto tabular si no existe

    # 2. Definir los Prompts de los Agentes Especialistas
    
    # --- AGENTE EXPERTO EN DATOS TABULARES ---
    if dataset_type == "tabular":
        system_prompt = """Eres un Científico de Datos experto especializado en Scikit-Learn (Machine Learning clásico).
        Tu tarea es analizar los metadatos de un dataset tabular y seleccionar los 2 mejores algoritmos para entrenar.

        INSTRUCCIONES CRÍTICAS:
        1. Determina si la tarea es 'classification' o 'regression' basándote en los metadatos.
        2. Selecciona 2 algoritmos clásicos de Scikit-Learn (ej. RandomForestClassifier, SVC, GradientBoostingRegressor, etc.).
        3. Sugiere 2 o 3 hiperparámetros clásicos (ej. n_estimators, max_depth, C).

        REGISTRO DE MODELOS:
        {TABULAR_REGISTRY}

        FORMATO DE SALIDA ESTRICTO:
        Devuelve ÚNICAMENTE un JSON válido siguiendo exactamente esta estructura, sin texto adicional:
        {{
            "task_type": "classification o regression",
            "models": [
                {{
                    "name": "NombreClaseExacto",
                    "reasoning": "Explicación breve",
                    "hyperparameters": {{ "parametro": "valor" }}
                }},
                {{
                    "name": "NombreClaseExacto2",
                    "reasoning": "Explicación breve",
                    "hyperparameters": {{ "parametro": "valor" }}
                }}
            ]
        }}
        """

    # --- AGENTE EXPERTO EN IMÁGENES ---
    elif dataset_type == "image":
        system_prompt = """Eres un Ingeniero de Machine Learning experto especializado en TensorFlow/Keras (Visión por Computador y Deep Learning).
        Tu tarea es analizar los metadatos de un dataset de imágenes y seleccionar las 2 mejores arquitecturas para entrenar.

        INSTRUCCIONES CRÍTICAS:
        1. La tarea será obligatoriamente 'image_classification'.
        2. Selecciona 2 arquitecturas de Keras apropiadas para Transfer Learning o desde cero (ej. MobileNetV2, ResNet50, EfficientNetB0, SimpleCNN).
        3. Sugiere 2 o 3 hiperparámetros de entrenamiento de Deep Learning (ej. learning_rate, batch_size, epochs).

        REGISTRO DE MODELOS:
        {IMAGE_REGISTRY}

        FORMATO DE SALIDA ESTRICTO:
        Devuelve ÚNICAMENTE un JSON válido siguiendo exactamente esta estructura, sin texto adicional:
        {{
            "task_type": "image_classification",
            "models": [
                {{
                    "name": "NombreClaseExacto",
                    "reasoning": "Explicación breve",
                    "hyperparameters": {{ "parametro": "valor" }}
                }},
                {{
                    "name": "NombreClaseExacto2",
                    "reasoning": "Explicación breve",
                    "hyperparameters": {{ "parametro": "valor" }}
                }}
            ]
        }}
        """
    else:
        raise ValueError(f"Tipo de dataset no reconocido: {dataset_type}")

    # 3. Construir y ejecutar la cadena para el agente seleccionado
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "METADATOS DEL DATASET:\n{metadata}")
    ])

    chain = prompt | llm | parser

    try:
        response_dict = chain.invoke({
            "metadata": dataset_metadata_json,
            "IMAGE_REGISTRY": IMAGE_REGISTRY,
            "TABULAR_REGISTRY": TABULAR_REGISTRY
        })
        return json.dumps(response_dict, indent=4, ensure_ascii=False)
        
    except Exception as e:
        print(f"Error ejecutando el Agente Selector de Modelos ({dataset_type}): {e}")
        # Fallbacks específicos según el tipo de fallo
        if dataset_type == "tabular":
            fallback = {
                "task_type": "classification",
                "models": [{"name": "RandomForestClassifier", "reasoning": "Fallback seguro.", "hyperparameters": {"n_estimators": 100}}]
            }
        else:
            fallback = {
                "task_type": "image_classification",
                "models": [{"name": "SimpleCNN", "reasoning": "Fallback seguro.", "hyperparameters": {"learning_rate": 0.001, "epochs": 10}}]
            }
        return json.dumps(fallback)