import json
import pandas as pd
import numpy as np
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

def critic(tool_output : str, descriptions : dict) -> str:
    """
    Agente Crítico: Revisa la salida del agente de XAI
    """
    # 2. Configurar el Output Parser para forzar y parsear el JSON
    parser = JsonOutputParser()

    # 3. Crear el Prompt Template estilo LangChain
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
         Eres un agente crítico especializado en evaluar la selección de herramientas de IA explicable.  
Tu tarea NO es responder la pregunta original del usuario, sino evaluar si la herramienta de explicabilidad escogida ha sido adecuada para el problema planteado.

Recibirás un JSON con esta estructura:
```json
{{
  "action_input": "Salida generada por la tool",
  "tool" : "Herramienta escogida",
  "input": "Pregunta original del usuario"
}}
```
         
Tu objetivo es analizar si la herramienta escogida es adecuada para el tipo de problema.
En el caso de que la herramienta sea correcta, debes realizar una interpretación en lenguaje natural de los datos proporcionados por la herramienta.

Debes responder EXCLUSIVAMENTE con un JSON válido usando EXACTAMENTE este formato:
```json
{{
  "action": "Final_Answer" | "Repeat",
  "action_input": "Texto explicativo largo aquí usando \\n para saltos de línea"
}}
```

Reglas IMPORTANTES:
- NUNCA escribas texto fuera del JSON.
- NUNCA uses markdown.
- NUNCA uses comillas triples.
- NUNCA añadas campos extra.
- "action" debe ser:
  - "Final Answer" si la tool escogida es razonablemente adecuada y la salida es útil.
  - "Repeat" si la tool es incorrecta, insuficiente, incoherente o poco explicativa.
- "action_input" debe estar escrito en lenguaje natural claro y profesional.
- Usa "\\n" para separar párrafos o puntos.
- Sé crítico y específico.
- Evalúa tanto la elección de la herramienta como la calidad de la explicación generada.
- Si decides "Repeat", explica claramente:
  - por qué la herramienta no es adecuada,
  - qué limitaciones tiene en este caso,
  - y qué tipo de herramienta sería más apropiada.
- Si decides "Final Answer", explica:
  - por qué la herramienta elegida es adecuada,
  - qué aporta la salida,
  - y cómo ayuda a interpretar el comportamiento del modelo.

Tools disponibles: 
{description}
         
Reglas para redactar el 'action_input' de tu Final Answer:
- Sé narrativo y coherente: No te limites a listar variables y sus pesos. Redacta párrafos hilados.
- Contextualiza los valores: Relaciona el valor de la característica con la realidad del dataset.
- Explica el 'Por qué': Profundiza en los argumentos (SHAP/LIME) y explica cómo la combinación de factores lleva a la decisión.
- NO te inventes información que no esté en la explicabilidad del modelo. Usa solo los datos proporcionados para tus explicaciones.
- Devuelve todas las explicaciones en español.
         
Tool escogida:
{tool_output}
         """)
    ])

# Criterios orientativos:
# - SHAP suele ser apropiado para explicaciones locales y globales basadas en contribución de features.
# - LIME suele ser útil para explicaciones locales aproximadas.
# - Counterfactuals son adecuados para mostrar cambios mínimos necesarios en la predicción.
# - Attention maps son útiles principalmente en transformers y visión.
# - Feature importance global puede ser insuficiente para preguntas individuales.
# - Métodos locales NO deben justificarse como explicaciones globales.
         
    # 4. Instanciar el modelo LLM mediante ChatOllama
    # Nota: Le pasamos format="json" para activar el modo JSON nativo de Ollama
    llm = ChatOllama(model="llama3", temperature=0, format="json")

    # 5. Componer la cadena (Pipeline)
    chain = prompt | llm | parser

    # 6. Ejecutar la cadena
    try:
        response_dict = chain.invoke({
            "tool_output": json.dumps(tool_output),
            "description" : descriptions
        })
        # LangChain devuelve un diccionario de Python gracias al JsonOutputParser. 
        # Lo convertimos a string JSON para mantener la compatibilidad con el resto de tu código.
        return response_dict
        
    except Exception as e:
        print(f"Error ejecutando la cadena de LangChain con Ollama: {e}")
        return "{}"
    