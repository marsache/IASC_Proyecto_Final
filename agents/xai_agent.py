
import json
from langchain_community.chat_models import ChatOllama
from langchain_classic.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_classic.memory import ChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.callbacks import BaseCallbackHandler

from .critic_agent import critic
from ..tools.FallbackToolkit import FallbackToolkit

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
    def __init__(self, agent, memory, desc):
        self.agent = agent
        self.memory = memory
        self.descriptions = desc

    def invoke(self, input_dict, config = None):
        # Construimos el input final para el agente
        final_input = {
            "input": input_dict["input"] if isinstance(input_dict, dict) else input_dict,
            "previous_instance": json.dumps(self.memory.last_instance, indent=2)
                                 if self.memory.last_instance else "Ninguna"
        }

        stop = False
        max_iters = 3
        iters = 0

        while not stop and iters < max_iters:
            iters += 1
            result =  self.agent.invoke(final_input, config)
            intermediate_steps = result.get("intermediate_steps", [])

            response_text = result.get("output", str(result))
            response = json.loads(response_text)
            response = {
                "action_input" : json.loads(response_text),
                "tool" : intermediate_steps[-1][0].tool,
                "input" : final_input
            }

            c = critic(response, self.descriptions)
            
            stop = c.get("action", "Repeat") == "Final_Answer" or "Final Answer"

            final_input["input"] += f"\n {c.get("action_input", "")}"

        return {"output" : c["action_input"], "intermediate_steps" : intermediate_steps}


def get_agent_tools(toolkit, dataset_type: str):
    """
    Devuelve la lista de herramientas exclusivas y permitidas 
    según la naturaleza del dataset (tabular o imagen).
    """
    tool_fallback = FallbackToolkit()
    tool_greetings = StructuredTool.from_function(
        func=tool_fallback.tool_greetings,
        name="responder_saludo",
        description=(
            "Útil para responder a saludos, presentaciones, despedidas o expresiones de cortesía "
            "del usuario (ej: 'hola', 'buenos días', 'gracias', '¿qué sabes hacer?')."
        ),
        return_direct=True
    )

    tool_out_of_scope = StructuredTool.from_function(
        func=tool_fallback.tool_out_of_scope,
        name="fuera_de_dominio",
        description=(
            "Útil cuando el usuario hace una pregunta que NO está relacionada con "
            "el modelo de Machine Learning, el dataset, la inteligencia artificial o la explicación "
            "de predicciones. Úsalo como salvavidas si la pregunta del usuario no tiene nada que ver con tu propósito."
        ),
        return_direct=True
    )

    tool_global = StructuredTool.from_function(
        func=toolkit.tool_shap_explain_global,
        name="explicar_modelo_global",
        description="Útil para explicar qué variables importan más en todo el modelo general.",
        return_direct=True
    )

    if dataset_type == "tabular":
        tool_local = StructuredTool.from_function(
            func=toolkit.tool_shap_lime_explain_local_prediction,
            name="explicar_prediccion_local",
            description="Útil para explicar por qué se tomó una decisión para un cliente o registro específico.",
            return_direct=True
        )

        tool_cf = StructuredTool.from_function(
            func=toolkit.tool_dice_explain,
            name="generar_contraejemplo",
            description="Útil para dar ejemplos de qué se debe cambiar en una instancia tabular para que cambie el resultado.",
            return_direct=True
        )
        
        tool_proto = StructuredTool.from_function(
            func=toolkit.tool_prototype,
            name="generar_prototipos",
            description="Útil para hablar de los datos más representativos del dataset tabular.",
            return_direct=True
        )
        
        return [tool_greetings, tool_out_of_scope, tool_global, tool_local, tool_cf, tool_proto]

    elif dataset_type == "image":
        tool_gradcam = StructuredTool.from_function(
            func=toolkit.tool_gradcam_explain_local_prediction,
            name="explicar_imagen_gradcam",
            description="Útil para generar mapas de calor y explicar visualmente de forma general en qué áreas gruesas de la imagen se fijó el modelo para tomar su decisión.",
            return_direct=True
        )

        tool_saliency = StructuredTool.from_function(
            func=toolkit.tool_saliency_map_explain,
            name="explicar_imagen_saliencia",
            description="Útil para calcular la sensibilidad de los píxeles. Úsalo cuando el usuario quiera saber qué contornos o píxeles exactos cambiarían más drásticamente la predicción si se alteraran.",
            return_direct=True
        )

        tool_ig = StructuredTool.from_function(
            func=toolkit.tool_integrated_gradients_explain,
            name="explicar_imagen_gradientes_integrados",
            description="Útil para ver una atribución matemática exacta a nivel de píxel. Úsalo para explicar de forma muy fina cómo cada píxel contribuyó desde un estado base (negro) hasta la decisión final.",
            return_direct=True
        )

        tool_occlusion = StructuredTool.from_function(
            func=toolkit.tool_occlusion_sensitivity_explain,
            name="explicar_imagen_oclusion",
            description="Útil para probar la robustez del modelo. Úsalo si se necesita saber qué pasa con la confianza del modelo cuando se tapan u ocultan ciertas partes clave de la imagen.",
            return_direct=True
        )
        
        # (Opcional) Si mantienes LIME para imágenes, la añades aquí
        tool_lime_image = StructuredTool.from_function(
            func=toolkit.tool_lime_explain_local_prediction,
            name="explicar_imagen_lime_superpixeles",
            description="Útil para agrupar la imagen en superpíxeles y ver qué parches visuales actúan a favor o en contra de la predicción.",
            return_direct=True
        )

        return [tool_greetings, tool_out_of_scope, tool_global, tool_gradcam, tool_saliency, tool_ig, tool_occlusion, tool_lime_image]

    else:
        raise ValueError(f"dataset_type no soportado: {dataset_type}. Usa 'tabular' o 'image'.")
    
def setup_xai_agent(metadata, model_info, toolkit, get_history):
    # 1. Procesar el metadato generado por el Agente Perfilador
    # Extraemos la información del JSON para el prompt
    dataset_name = metadata.get("dataset_name", "No disponible")
    description = metadata.get("dataset_description", "No disponible")
    target_info = metadata.get("target_description", "No disponible")
    # Convertimos el diccionario de features a un string legible para el LLM
    features_dict = json.dumps(metadata.get("features", {}), indent=2, ensure_ascii=False)
    
    tools = get_agent_tools(toolkit, metadata.get("dataset_type", "tabular"))
    tool_names = [tool.name for tool in tools]
    # 3. Configurar LLM local
    llm = ChatOllama(model="llama3", temperature=0, format = "json")

    # 4. Definir el System Prompt con los datos del perfilador
    # Nota: Usamos doble llave {{ }} para lo que NO queremos que LangChain intente rellenar ahora (formato JSON)
    system_prompt_template = """
    Eres un asistente experto en Inteligencia Artificial Explicable (XAI). Tu objetivo es analizar y seleccionar qué herramienta es apropiada para la explicación requerida por el usuario.
    SOLO debes llamar a la herramienta.

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
    Cuando hayas usado la herramienta seleccionada, debes usar la acción "Final Answer".
    ```json
    {{
        "action": "Final Answer",
        "action_input": "<salida de la herramienta>"
    }}
    ```

    Reglas para redactar el 'action_input' de tu Final Answer:
    - Debes devolver Final Answer inmediatamente después de recibir el resultado.
    - Llama ÚNICAMENTE a solo una herramienta
    - Utiliza EXACTAMENTE la salida de la herramienta a la que se ha llamado
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
        handle_parsing_errors=True,
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

    xaiagent = XAIAgent(agent_executor, memory, [{t.name : t.description} for t in tools])

    return xaiagent