import json

from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def recommend_best_models(dataset_metadata_json: str) -> str:
    """
    Agente Selector de Modelos: Analiza los metadatos del dataset y devuelve
    los 2 mejores modelos de Machine Learning para entrenar, incluyendo parámetros.
    """
    # 1. Configurar el parser de JSON
    parser = JsonOutputParser()

    # 2. Definir el System Prompt con instrucciones muy estrictas para Llama 3 8B
    # 2. Definir el System Prompt con instrucciones muy estrictas para Llama 3 8B
    system_prompt = """Eres un Científico de Datos experto (Data Scientist) especializado en Scikit-Learn.
    Tu tarea es analizar los metadatos de un dataset y seleccionar los 2 mejores algoritmos clásicos de Machine Learning para entrenar.

    INSTRUCCIONES:
    1. Determina si el problema es de 'classification' o 'regression' basándote en la 'target_description'.
    2. Elige los 2 algoritmos de Scikit-Learn más apropiados dadas las características (lineales, basados en árboles, SVM, etc.).
    3. Para cada modelo, sugiere 2 o 3 hiperparámetros clave con valores recomendados y seguros.
    
    FORMATO DE SALIDA ESTRICTO:
    Devuelve ÚNICAMENTE un JSON válido siguiendo exactamente esta estructura:
    {{
        "task_type": "classification o regression",
        "models": [
            {{
                "name": "NombreClaseScikitLearn (ej. RandomForestClassifier, SVR, LogisticRegression)",
                "reasoning": "Explicación breve de por qué es adecuado",
                "hyperparameters": {{
                    "nombre_parametro": "valor recomendado (número o string)"
                }}
            }},
            {{
                "name": "NombreClaseScikitLearn2",
                "reasoning": "Explicación breve",
                "hyperparameters": {{
                    "nombre_parametro": "valor recomendado (número o string)"
                }}
            }}
        ]
    }}
    """

    # 3. Crear el Prompt Template
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "METADATOS DEL DATASET:\n{metadata}")
    ])

    # 4. Instanciar el modelo LLM local (Ollama)
    # temperature=0 para asegurar que recomiende lo más lógico y no alucine parámetros
    llm = ChatOllama(model="llama3", temperature=0, format="json")

    # 5. Componer la cadena (Pipeline LCEL)
    chain = prompt | llm | parser

    # 6. Ejecutar la cadena
    try:
        response_dict = chain.invoke({
            "metadata": dataset_metadata_json
        })
        # Devolvemos el string JSON formateado
        return json.dumps(response_dict, indent=4, ensure_ascii=False)
        
    except Exception as e:
        print(f"Error ejecutando el Agente Selector de Modelos: {e}")
        return "{}"