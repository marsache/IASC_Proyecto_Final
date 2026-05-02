
import json
from langchain_community.chat_models import ChatOllama
from langchain_classic.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool

def setup_xai_agent(metadata, model_info, toolkit):
    # 1. Procesar el metadato generado por el Agente Perfilador
    # Extraemos la información del JSON para el prompt
    dataset_name = metadata.get("dataset_name", "No disponible")
    description = metadata.get("dataset_description", "No disponible")
    target_info = metadata.get("target_description", "No disponible")
    # Convertimos el diccionario de features a un string legible para el LLM
    features_dict = json.dumps(metadata.get("features", {}), indent=2, ensure_ascii=False)

    # 2. Adaptación a LangChain Tools
    tool_global = StructuredTool.from_function(
        func=toolkit.tool_shap_explain_global,
        name="explicar_modelo_global",
        description="Útil para explicar qué variables importan más en todo el modelo general."
    )
    
    tool_local = StructuredTool.from_function(
        func=toolkit.tool_shap_explain_local_prediction,
        name="explicar_prediccion_local",
        description="Útil para explicar por qué se tomó una decisión para un cliente específico."
    )
    
    tools = [tool_global, tool_local]
    tool_names = [tool_global.name, tool_local.name] 
    # 3. Configurar LLM local
    llm = ChatOllama(model="llama3", temperature=0)

    # 4. Definir el System Prompt con los datos del perfilador
    # Nota: Usamos doble llave {{ }} para lo que NO queremos que LangChain intente rellenar ahora (formato JSON)
    system_prompt_template = """
    Eres un asistente experto en Inteligencia Artificial Explicable (XAI). Tu objetivo es analizar y explicar las predicciones de modelos de Machine Learning de forma detallada, comprensible y razonada para un usuario humano.

    [CONTEXTO DEL PROBLEMA]
    - Dataset: {dataset_name}
    - Descripción general: {description}
    - Objetivo (Target): {target_info}
    - Diccionario de características y rangos: 
    {features_dict}
    - Modelo evaluado y métricas: {model_info}

    [HERRAMIENTAS DISPONIBLES]
    Tienes acceso a las siguientes herramientas para extraer información global o local del modelo:
    {tools}
    {tool_names}

    [REGLAS DE OPERACIÓN - ESTRICTAS]
    Para usar una herramienta, debes responder ÚNICAMENTE con un bloque de código JSON válido. No añadas texto introductorio ni explicaciones fuera del JSON. Usa este formato exacto:
    ```json
    {{
        "action": "nombre_de_la_herramienta_aqui",
        "action_input": {{ "nombre_del_parametro": "valor_del_parametro" }}
    }}
    ```

    [INSTRUCCIONES PARA LA EXPLICACIÓN FINAL]
    Cuando hayas usado las herramientas necesarias y estés listo para responder al usuario, debes usar la acción "Final Answer".
    ```json
    {{
        "action": "Final Answer",
        "action_input": "Tu explicación larga aquí. Usa \\n para saltos de línea. NUNCA uses comillas triples."
    }}
    ```

    Reglas para redactar el 'action_input' de tu Final Answer:
    - Sé narrativo y coherente: No te limites a listar variables y sus pesos. Redacta párrafos hilados.
    - Contextualiza los valores: Relaciona el valor de la característica con la realidad del dataset.
    - Explica el 'Por qué': Profundiza en los argumentos (SHAP/LIME) y explica cómo la combinación de factores lleva a la decisión.
    """

    # 5. Crear el Prompt Template y aplicar variables parciales (las que vienen del perfilador)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("human", "Pregunta: {input}\n\nHistorial de acciones y observaciones (Scratchpad):\n{agent_scratchpad}"),
    ])
    
    # Inyectamos los valores del dataset que son estáticos para esta sesión
    prompt = prompt.partial(
        dataset_name=dataset_name,
        description=description,
        target_info=target_info,
        features_dict=features_dict,
        model_info=model_info
    )

    # 6. Crear el Agente y su Ejecutor
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True # Recomendado para modelos 8B
    )

    return agent_executor