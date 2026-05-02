import pandas as pd
import json

# ours
from utils.user_input_handling import find_dataset, find_target_feature_in_dataset
import tools.XAIToolkit as xai
from utils.dataset_cleaning import preprocess_dataset, descale_x
from agents.xai_agent import setup_xai_agent
from utils.models import generate_model_info, build_and_train_recommended_models
from agents.model_selector_agent import recommend_best_models

def preparar_entorno():

    dataset_path = find_dataset()

    dataset = pd.read_csv(dataset_path)
    target = find_target_feature_in_dataset(dataset)

    x_train, x_test, y_train, y_test, dataset_metadata, scaler = preprocess_dataset(dataset, dataset_path, target)
    
    best_models_json = recommend_best_models(dataset_metadata_json=dataset_metadata)

    recommendations_dict = json.loads(best_models_json)

    # 2. Extraemos el task_type dinámicamente (añadimos fallback a "classification" por seguridad)
    tipo_tarea = recommendations_dict.get("task_type", "classification")

    # 1. Parseamos el JSON del Agente Selector

    models = build_and_train_recommended_models(best_models_json, x_train, y_train)

    for model in models:
        model_info = generate_model_info(model["model_object"], x_test, y_test, task_type=tipo_tarea)
        model["model_info"] = model_info
        print(model_info)

    return models, x_train, x_test, y_train, y_test, dataset_metadata, scaler

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
def test_con_llm(toolkit, cliente_ejemplo, cliente_ejemplo_descaled, model_info, dataset_metadata):
    print("\n" + "="*50)
    print("🤖 MODO 2: EJECUCIÓN CON LLM (Ollama + LangChain)")
    print("="*50)
    
    agent_executor = setup_xai_agent(metadata=dataset_metadata, model_info=model_info, toolkit=toolkit)

    # 4.4. Probar las consultas
    print("\nUsuario: ¿Cómo funciona el modelo en general? ¿Cuáles son las variables clave?")
    response_global = agent_executor.invoke({"input": "¿Cómo funciona el modelo en general? ¿Cuáles son las variables clave?"})
    print("\nRespuesta LLM:\n", response_global['output'])

    print(f"\nUsuario: Explícame la predicción para este cliente: {cliente_ejemplo}")
    response_local = agent_executor.invoke({
        "input": f"Explícame en un lenguaje de negocio por qué el modelo ha tomado esta decisión para el siguiente cliente (valores reales): {cliente_ejemplo_descaled}"
    })
    print("\nRespuesta LLM:\n", response_local['output'])

# ==========================================
# BLOQUE PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # 1. Preparar datos y modelo
    models, x_train, x_test, y_train, y_test, dataset_metadata, scaler = preparar_entorno()
    
    print(x_train.shape)
    # 2. Inicializar Toolkit
    toolkit = xai.XAIToolkit(model=models[0]["model_object"], x_test=x_test, dataset_metadata=dataset_metadata)
    
    sample = x_test[0]
    dataset_metadata = json.loads(dataset_metadata)

    print(dataset_metadata)
    labels = list(dataset_metadata['features'].keys())
    i = 0
    sample_test = {}
    for label in labels:
        sample_test[label] = sample[i]
        i += 1

    # print(sample_test)
    # 3. Probar funciones crudas
    # test_sin_llm(toolkit, sample_test)
    
    # 4. Probar agente (Descomenta la siguiente línea si tienes Ollama instalado y corriendo)
    test_con_llm(toolkit, sample_test, descale_x(sample, scaler, labels), models[0]["model_info"], dataset_metadata)