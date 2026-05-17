
import json
from langchain_community.chat_models import ChatOllama
from langchain_classic.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_classic.memory import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.callbacks import BaseCallbackHandler

class InstanceMemory:
    def __init__(self):
        self.last_instance = None

class XAIInstanceCaptureCallback(BaseCallbackHandler):
    def __init__(self, memory):
        self.memory = memory

    def on_agent_action(self, action, **kwargs):
        # action.tool_input contiene el JSON exacto
        tool_input = action.tool_input
        if "instance_data" in tool_input:
            self.memory.last_instance = tool_input["instance_data"]

class XAIAgent:
    def __init__(self, agent, memory):
        self.agent = agent
        self.memory = memory

    def invoke(self, input_dict, config = None):
        # Construimos el input final para el agente
        final_input = {
            "input": input_dict["input"] if isinstance(input_dict, dict) else input_dict,
            "previous_instance": json.dumps(self.memory.last_instance, indent=2)
                                 if self.memory.last_instance else "Ninguna"
        }

        return self.agent.invoke(final_input, config)

def setup_xai_agent(metadata, model_info, toolkit, get_history):
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
    
    # tool_local = StructuredTool.from_function(
    #     func=toolkit.tool_shap_explain_local_prediction,
    #     name="explicar_prediccion_local",
    #     description="Útil para explicar por qué se tomó una decisión para un cliente específico."
    # )

    # tool_lime_local = StructuredTool.from_function(
    #     func=toolkit.tool_lime_explain_local_prediction,
    #     name="explicar_prediccion_local_lime",
    #     description="Útil para explicar por qué el modelo tomó una decisión para un cliente específico usando LIME."
    # )

    tool_local = StructuredTool.from_function(
        func=toolkit.tool_shap_lime_explain_local_prediction,
        name="explicar_prediccion_local",
        description="Útil para explicar por qué se tomó una decisión para un cliente específico."
    )

    tool_cf = StructuredTool.from_function(
        func = toolkit.tool_dice_explain,
        name = "generar_contraejemplo",
        description="Útil para dar ejemplos de qué se debe cambiar en una instancia para que cambie el resultado"
    )
    
    tool_proto = StructuredTool.from_function(
        func = toolkit.tool_prototype,
        name = "generar_prototipos",
        description= "Útil para hablar de los datos más representativos del dataset"
    )
    
    
    #tools = [tool_global, tool_local, tool_lime_local]
    tools = [tool_global, tool_local, tool_cf, tool_proto]
    #tool_names = [tool_global.name, tool_local.name, tool_lime_local.name]
    tool_names = [tool_global.name, tool_local.name, tool_cf.name, tool_proto.name]
    # 3. Configurar LLM local
    llm = ChatOllama(model="llama3", temperature=0, format = "json")

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
    - Para usar una herramienta, debes responder ÚNICAMENTE con un bloque de código JSON válido.
    - No añadas texto introductorio ni explicaciones fuera del JSON.
    - Usa este formato exacto:
    ```json
    {{
        "action": "nombre_de_la_herramienta_aqui",
        "action_input": {{ 
        "nombre_del_parametro_1": "valor_del_parametro_1",
        "nombre_del_parametro_2": "valor_del_parametro_2",
        ...,
        "nombre_del_parametro_n": "valor_del_parametro_n"
        }}
    }}
    ``` 
    - Si en el propio prompt te especifica usar un parámetro, usa EXACTAMENTE ese parámetro, no te inventes sus valores.

    [REGLAS CRÍTICAS SOBRE INSTANCIAS]
    - Si el usuario te especifica unos datos, usa EXACTAMENTE esos datos, NO los cambies ni los inventes.
    - Si el usuario hace referencia a una instancia previa, debes reutilizar EXACTAMENTE la misma estructura JSON previamente observada.


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
    - NO te inventes información que no esté en la explicabilidad del modelo. Usa solo los datos proporcionados para tus explicaciones.
    """

    # 5. Crear el Prompt Template y aplicar variables parciales (las que vienen del perfilador)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_template),
        ("human", "Pregunta: {input}\n\nInstancia previa, si existe: {previous_instance}\n\nHistorial de acciones y observaciones (Scratchpad):\n{agent_scratchpad}"),
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
    memory = InstanceMemory()
    callback = XAIInstanceCaptureCallback(memory)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True, # Recomendado para modelos 8B
        callbacks= [callback],
        return_intermediate_steps=True,
        #max_iterations=1
    )

    agent_executor = RunnableWithMessageHistory(
        agent_executor,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
        output_messages_key="output",
    )

    xaiagent = XAIAgent(agent_executor, memory)

    return xaiagent