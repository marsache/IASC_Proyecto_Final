import json
import pandas as pd
import numpy as np
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
    