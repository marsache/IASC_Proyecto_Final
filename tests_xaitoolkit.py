import pandas as pd
import json

# ours
from utils.user_input_handling import find_dataset, find_target_feature_in_dataset
import tools.XAIToolkit as xai
from utils.dataset_utils import preprocess_dataset, descale_x, get_random_row
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

    return models, x_train, x_test, y_train, y_test, json.loads(dataset_metadata), scaler, dataset, target

def test_con_llm(toolkit, cliente_ejemplo, cliente_ejemplo_descaled, cliente_target, model_info, dataset_metadata):
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

    print(f"\nUsuario: Explícame qué debería cambiar para que me de la predicción {(1 - cliente_target)}: {cliente_ejemplo}")
    response_local = agent_executor.invoke({
        "input": f"Explícame en un lenguaje de negocio qué debo cambiar para que cambie la predicción a {(1 - cliente_target)} en el siguiente cliente (valores reales): {cliente_ejemplo_descaled}"
    })
    print("\nRespuesta LLM:\n", response_local['output'])

# ==========================================
# BLOQUE PRINCIPAL
# ==========================================
if __name__ == "__main__":
    # 1. Preparar datos y modelo
    models, x_train, x_test, y_train, y_test, dataset_metadata, scaler, dataset, target = preparar_entorno()
    
    # 2. Inicializar Toolkit con un solo modelo
    toolkit = xai.XAIToolkit(model=models[0]["model_object"], x_test=x_test, dataset_metadata=dataset_metadata, dataset = dataset, target = target)
    
    sample = get_random_row(dataset=x_test, dataset_metadata=dataset_metadata, target = y_test)

    # 4. Probar agente (Descomenta la siguiente línea si tienes Ollama instalado y corriendo)
    test_con_llm(toolkit, sample["sample"], descale_x(sample["sample"], scaler), sample["target"], models[0]["model_info"], dataset_metadata)