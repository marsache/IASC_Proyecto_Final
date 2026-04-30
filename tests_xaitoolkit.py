import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import tools.XAIToolkit as xai

# --- IMPORTACIONES DE LANGCHAIN ---
from langchain_core.tools import StructuredTool
from langchain_community.chat_models import ChatOllama
from langchain_classic.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# ==========================================
# 2. CONFIGURACIÓN DE DATOS Y MODELO MOCK
# ==========================================
def preparar_entorno():
    print("📊 Generando datos simulados y entrenando modelo...")
    # Dataset simulado de aprobación de crédito
    np.random.seed(42)
    n = 500
    df = pd.DataFrame({
        'edad': np.random.randint(18, 70, n),
        'ingresos': np.random.randint(15000, 80000, n),
        'deuda_actual': np.random.randint(0, 30000, n),
        'historial_impagos': np.random.randint(0, 3, n)
    })
    # Lógica inventada: aprueban si tienen buenos ingresos, poca deuda y sin impagos
    prob_aprobacion = (df['ingresos'] / 80000) - (df['deuda_actual'] / 30000) - (df['historial_impagos'] * 0.3)
    df['aprobado'] = (prob_aprobacion > 0).astype(int)

    X = df.drop('aprobado', axis=1)
    y = df['aprobado']
    
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X, y)
    
    return model, X

# ==========================================
# 3. EJECUCIÓN SIN LLM (Prueba Unitaria)
# ==========================================
def test_sin_llm(toolkit, cliente_ejemplo):
    print("\n" + "="*50)
    print("🚀 MODO 1: EJECUCIÓN SIN LLM (Raw JSON)")
    print("="*50)
    
    print("\n--- 🌍 Explicación Global ---")
    print(toolkit.tool_shap_explain_global(top_k=3))
    
    print("\n--- 👤 Explicación Local (Cliente Ejemplo) ---")
    print(toolkit.tool_shap_explain_local_prediction(instance_data=cliente_ejemplo))


# ==========================================
# 4. EJECUCIÓN CON LLM (LangChain + Ollama)
# ==========================================
def test_con_llm(toolkit, cliente_ejemplo):
    print("\n" + "="*50)
    print("🤖 MODO 2: EJECUCIÓN CON LLM (Ollama + LangChain)")
    print("="*50)
    
    # 4.1. Adaptación a LangChain Tools
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

    # 4.2. Configurar LLM local (Ollama). Usa 'llama3' o el modelo que tengas descargado
    # Asegúrate de tener Ollama corriendo en segundo plano: `ollama run llama3`
    llm = ChatOllama(model="llama3", temperature=0)

    system_prompt = """Eres un asistente experto en Inteligencia Artificial Explicable.
    Tienes acceso a las siguientes herramientas:
    {tool_names}
    {tools}

    Para usar una herramienta, debes responder EXACTAMENTE con un bloque de código JSON con este formato:
    ```json
    {{
      "action": "nombre_de_la_herramienta",
      "action_input": {{ "parametro": "valor" }}
    }}
    ```

    Si ya tienes la respuesta final para el usuario, responde con:
    ```json
    {{
      "action": "Final Answer",
      "action_input": "Tu explicación final aquí en lenguaje natural"
    }}
    ```
    """

    # 4.3. Crear el Prompt y el Agente
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Pregunta: {input}\n\nHistorial de acciones y observaciones (Scratchpad):\n{agent_scratchpad}"),
    ])
    # Nota: Llama3 mediante ChatOllama soporta tool calling en versiones recientes de Langchain
    agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    # 4.4. Probar las consultas
    print("\n🗣️ Usuario: ¿Cómo funciona el modelo en general? ¿Cuáles son las variables clave?")
    response_global = agent_executor.invoke({"input": "¿Cómo funciona el modelo en general? ¿Cuáles son las variables clave?"})
    print("\n💡 Respuesta LLM:\n", response_global['output'])

    print(f"\n🗣️ Usuario: Explícame la predicción para este cliente: {cliente_ejemplo}")
    response_local = agent_executor.invoke({
        "input": f"Explícame en un lenguaje de negocio por qué el modelo ha tomado esta decisión para el siguiente cliente: {cliente_ejemplo}"
    })
    print("\n💡 Respuesta LLM:\n", response_local['output'])

# ==========================================
# BLOQUE PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # 1. Preparar datos y modelo
    modelo_rf, datos_fondo = preparar_entorno()
    
    # 2. Inicializar Toolkit
    toolkit = xai.XAIToolkit(model=modelo_rf, background_data=datos_fondo)
    
    # Cliente de prueba: Joven, pocos ingresos, mucha deuda y 2 impagos (debería ser rechazado)
    cliente_test = {'edad': 25, 'ingresos': 18000, 'deuda_actual': 25000, 'historial_impagos': 2}
    
    # 3. Probar funciones crudas
    test_sin_llm(toolkit, cliente_test)
    
    # 4. Probar agente (Descomenta la siguiente línea si tienes Ollama instalado y corriendo)
    # test_con_llm(toolkit, cliente_test)