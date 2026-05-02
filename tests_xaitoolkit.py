import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score

# ours
from config import getSettings 
import tools.XAIToolkit as xai
from utils.dataset_cleaning import preprocess_dataset
from xai_agent import setup_xai_agent

# --- IMPORTACIONES DE LANGCHAIN ---
from langchain_core.tools import StructuredTool
from langchain_community.chat_models import ChatOllama
from langchain_classic.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def generate_model_info(model, X_test: np.ndarray, y_test: np.ndarray, task_type: str = "classification") -> str:
    """
    Evalúa el modelo entrenado y genera un resumen en texto plano
    con su nombre y métricas clave para inyectar en el LLM.
    """
    # 1. Extraer el nombre del algoritmo automáticamente
    # Ejemplo: pasará de ser un objeto a un string como "RandomForestClassifier" o "LogisticRegression"
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

# ==========================================
# 2. CONFIGURACIÓN DE DATOS Y MODELO MOCK
# ==========================================
def preparar_entorno():
    print("📊 Generando datos simulados y entrenando modelo...")
    # Dataset simulado de aprobación de crédito

    dataset_found = False
    dataset_name = ""
    while (not dataset_found):

        dataset_path = input("CSV path: ")
        dataset_name = dataset_path
        dataset_path = getSettings().base_dataset_path + dataset_path

        dataset_found = os.path.exists(dataset_path)
        if(not dataset_found):
            print("Path not found")

    dataset = pd.read_csv(dataset_path)

    print(f"Dataset head:\n{dataset.head()}")

    target_found = False
    
    while (not target_found):

        target = input("Target feature: ") #"over_limit"
        
        target_found = target in dataset
        if(not target_found):
            print("Feature not found")

    x_train, x_test, y_train, y_test, dataset_metadata = preprocess_dataset(dataset, target)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(x_train, y_train)

    model_info = generate_model_info(model, x_test, y_test, task_type="classification")

    labels = list(dataset.columns.values)
    labels.remove(target)
    return model, x_train, x_test, y_train, y_test, labels, dataset_metadata, dataset_name, model_info

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
def test_con_llm(toolkit, cliente_ejemplo, dataset_metadata, dataset_name, model_info):
    print("\n" + "="*50)
    print("🤖 MODO 2: EJECUCIÓN CON LLM (Ollama + LangChain)")
    print("="*50)
    
    agent_executor = setup_xai_agent(dataset_metadata_json=dataset_metadata, model_info=model_info, dataset_name=dataset_name, toolkit=toolkit)

    # 4.4. Probar las consultas
    print("\nUsuario: ¿Cómo funciona el modelo en general? ¿Cuáles son las variables clave?")
    response_global = agent_executor.invoke({"input": "¿Cómo funciona el modelo en general? ¿Cuáles son las variables clave?"})
    print("\nRespuesta LLM:\n", response_global['output'])

    print(f"\nUsuario: Explícame la predicción para este cliente: {cliente_ejemplo}")
    response_local = agent_executor.invoke({
        "input": f"Explícame en un lenguaje de negocio por qué el modelo ha tomado esta decisión para el siguiente cliente: {cliente_ejemplo}"
    })
    print("\nRespuesta LLM:\n", response_local['output'])

# ==========================================
# BLOQUE PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # 1. Preparar datos y modelo
    modelo_rf, x_train, x_test, y_train, y_test, labels, dataset_metadata, dataset_name, model_info = preparar_entorno()
    
    print(x_train.shape)
    # 2. Inicializar Toolkit
    toolkit = xai.XAIToolkit(model=modelo_rf, x_test=x_test, labels=labels)
    
    sample = x_test[0]

    i = 0
    sample_test = {}
    for label in labels:
        sample_test[label] = sample[i]
        i += 1

    # print(sample_test)
    # 3. Probar funciones crudas
    # test_sin_llm(toolkit, sample_test)
    
    # 4. Probar agente (Descomenta la siguiente línea si tienes Ollama instalado y corriendo)
    test_con_llm(toolkit, sample_test, dataset_metadata, dataset_name, model_info)