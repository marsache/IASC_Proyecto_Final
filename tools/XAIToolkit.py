import os
import uuid
import urllib.parse

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import shap
import json

from shap.maskers import Image as shap_image

import pandas as pd
import numpy as np
from lime import lime_tabular, lime_image

from alibi.explainers import AnchorTabular

from dice_ml import Data, Model, Dice
from PIL import Image

from mmd_critic import MMDCritic
from mmd_critic.kernels import RBFKernel

class XAIToolkit:
    def __init__(self, model, x_test, dataset_metadata, dataset, target, plots_dir: str = "plots"):
        # 1. Asignaciones básicas
        self.model = model
        self.x_test = x_test
        self.dataset_metadata = dataset_metadata
        self.target = target
        self.plots_dir = plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)
        
        # Extraer el tipo de tarea desde los metadatos devueltos por el agente/JSON
        self.task_type = dataset_metadata.get("dataset_type", "tabular")

        # 2. Enrutamiento según el tipo de Dataset
        if self.task_type == "image":
            self._init_image_explainers()
        else:
            self._init_tabular_explainers(dataset)

            
    def _init_tabular_explainers(self, dataset):
        """Inicializa las herramientas XAI exclusivas para datos tabulares"""
        print("[*] Inicializando explicadores XAI para datos tabulares...")
        
        features_df = dataset.drop(columns=[self.target])
        continuous_cols = features_df.select_dtypes(include=['number']).columns.tolist()
        self.labels = features_df.columns.tolist()

        # Inicializar DiCE (Tabular)
        d_dice = Data(dataframe=dataset, continuous_features=continuous_cols, outcome_name=self.target)
        m_dice = Model(model=self.model, backend='sklearn')
        self.dice = Dice(d_dice, m_dice, method='genetic')
        
        # Inicializar SHAP (Tabular)
        model_name = type(self.model).__name__
        if "RandomForest" in model_name or "GradientBoosting" in model_name:
            self.explainer = shap.TreeExplainer(self.model)
        elif "Logistic" in model_name or "Linear" in model_name:
            self.explainer = shap.LinearExplainer(self.model, self.x_test)
        else:
            background_summary = shap.sample(self.x_test, 100)
            self.explainer = shap.Explainer(self.model.predict, background_summary)

        # Inicializar LIME (Tabular)
        lime_mode = "classification" if hasattr(self.model, "predict_proba") else "regression"
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=self.x_test,
            feature_names=self.labels,
            mode=lime_mode,
        )

        # Inicializar Anchor (Tabular)
        self.anchor_explainer = AnchorTabular(predictor=self.model.predict, feature_names=self.labels)
        self.anchor_explainer.fit(self.x_test)

        # Inicializar MMD Critic (Tabular)
        self.critic = MMDCritic(features_df, RBFKernel(sigma=1))

    def _init_image_explainers(self):
        """Inicializa las herramientas XAI exclusivas para visión por computador (Keras)"""
        print("[*] Inicializando explicadores XAI para imágenes...")
        
        # En imágenes, nuestras etiquetas de características son las clases reales detectadas (ej: ['gato', 'perro'])
        self.labels = self.dataset_metadata.get("classes", [])

        # 1. SHAP para Imágenes (Deep Learning)
        # Para redes de Keras, usamos el Explainer genérico pasándole un combinador de imágenes (Partition masker)
        # o directamente el modelo si la versión de SHAP lo gestiona de forma nativa.
        masker = shap_image("inpaint_telea", shape=self.x_test[0].shape)
        self.explainer = shap.Explainer(self.model, masker, output_names=self.labels)

        # 2. LIME para Imágenes
        # Cambiamos completamente el motor: de lime_tabular pasamos a lime_image
        self.lime_explainer = lime_image.LimeImageExplainer()

        # 3. Desactivación de herramientas no compatibles de forma nativa o placeholders
        # DiCE, Anchors tabulares y MMDCritic clásico no aceptan tensores de imágenes directamente.
        self.dice = None
        self.anchor_explainer = None
        self.critic = None

    def _get_lime_predict_fn(self):
        """Devuelve la función de predicción adecuada para el explainer LIME,
        soportando modelos tabulares y redes neuronales de imágenes.
        """
        if self.task_type == "image":
            def image_predict_fn(images):
                # 'images' es un array de NumPy con forma (N, H, W, C) que envía LIME
                
                if hasattr(self.model, "predict"):
                    # Forzamos verbose=0 para evitar que las barras de carga de Keras colapsen la consola
                    preds = self.model.predict(images, verbose=0)
                    
                    # CORRECCIÓN PARA CLASIFICACIÓN BINARIA EN IMÁGENES:
                    # Si la salida de la red es (N, 1), LIME necesita obligatoriamente (N, 2)
                    if len(preds.shape) == 2 and preds.shape[1] == 1:
                        return np.c_[1 - preds, preds]
                        
                    return preds
                    
                raise AttributeError("El modelo de imágenes no tiene un método de predicción compatible.")

            return image_predict_fn
        else:
            # =================================================================
            # RAMA PARA DATOS TABULARES (Tu lógica original optimizada)
            # =================================================================
            if hasattr(self.model, "predict_proba"):
                return self.model.predict_proba
                
            # Si es regresión tabular, LIME Tabular sí espera un array 2D de tipo (N, 1)
            return lambda x: self.model.predict(x).reshape(-1, 1)

    def _save_plot(self, prefix: str) -> str:
        """Guarda la figura matplotlib activa en el directorio de plots y devuelve la ruta."""
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        path = os.path.join(self.plots_dir, filename)
        plt.savefig(path, bbox_inches="tight")
        plt.close()
        return path
    
    def tool_dice_explain(self, instance_data : dict, features_to_vary : list[str], desired_class : int, top_k: int = 5) -> str:
        """
        Genera contraejemplos de un registro.
        Llama a esta función si el individuo pregunta "¿Qué debo cambiar para que salga otro resultado?"

        Args:
            instance_data (dict): Un diccionario con los nombres de las columnas y los valores del individuo al que generar el contraejemplo.
            features_to_vary (list[str]): Lista de las variables del dataset que son posibles de variar de forma sencilla para generar los contraejemplos.
            desired_class (int): Clase a la que desea ser cambiada
            top_k (int, optional): Número de contraejemplos a genearar. Por defectoe es 5.

        Returns:
            str: Un JSON en formato string de los contraejemplos
        """
        df_instance = pd.DataFrame([instance_data])

        dice_exp = self.dice.generate_counterfactuals(
        df_instance,
        total_CFs=top_k,
        desired_class = desired_class, 
        features_to_vary=features_to_vary,
        )
        
        result = {
            "analisis" : f"{top_k} contrajemplos",
            "contraejemplos" : dice_exp.cf_examples_list[0].final_cfs_df.to_dict()
        }

        return json.dumps(result)

    def tool_shap_explain_global(self, top_k: int = 5) -> str:
        """
        Explica el comportamiento global del modelo. Adapta su comportamiento
        si el dataset es tabular o de imágenes para evitar bloqueos de rendimiento.
        """
        try:
            # --- CASO: CLASIFICACIÓN DE IMÁGENES (NUEVO) ---
            if self.task_type == "image":
                print("[*] Ejecutando SHAP Global para imágenes. Optimizando muestra...")
                
                # CRUCIAL: Solo explicamos una muestra diminuta (ej. 3 imágenes) para el comportamiento global
                # Esto reduce el tiempo de 40 minutos a menos de 30 segundos.
                subset_size = min(3, len(self.x_test))
                x_samples = self.x_test[:subset_size]
                
                # Calcular SHAP solo para este subset representativo
                shap_values = self.explainer(x_samples, max_evals=500) # max_evals limita los pases por imagen
                
                result = {
                    "analisis": "Análisis de importancia global de características visuales (Imágenes)",
                    "nota": "En visión por computador, la importancia global se evalúa mediante mapas de calor (heatmaps) sobre regiones de interés (superpíxeles).",
                    "imagenes_evaluadas": subset_size,
                    "clases_detectadas": self.labels
                }
                
                # Generar el gráfico específico de SHAP para imágenes
                try:
                    # shap.plots.image renderiza las regiones que aumentan/disminuyen la probabilidad
                    shap.plots.image(shap_values, show=False)
                    result["plot_path"] = self._save_plot("shap_global_images")
                except Exception as e:
                    print(f"Error al generar gráfico SHAP de imágenes: {e}")
                    pass
                    
                return json.dumps(result, ensure_ascii=False)

            # --- CASO: DATASET TABULAR (TU CÓDIGO ORIGINAL SEGURO) ---
            else:
                # Para datos tabulares podemos usar una muestra más amplia si x_test es gigante
                # Si x_test tiene más de 300 filas, muestreamos para mantener la agilidad
                eval_set = self.x_test if len(self.x_test) <= 300 else shap.sample(self.x_test, 300)
                shap_values = self.explainer(eval_set)
                
                if len(shap_values.shape) == 3:
                    valores_objetivo = shap_values.values[:, :, 1]
                else:
                    valores_objetivo = shap_values.values

                mean_abs_shap = np.abs(valores_objetivo).mean(axis=0)
                feature_names = self.labels

                global_importance = {
                    feature: float(importance) 
                    for feature, importance in zip(feature_names, mean_abs_shap)
                }
                
                sorted_features = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
                
                top_features_summary = {}
                for feature, importance in sorted_features:
                    feature_idx = list(feature_names).index(feature)
                    
                    f_values = eval_set[:, feature_idx] if isinstance(eval_set, np.ndarray) else eval_set.iloc[:, feature_idx].to_numpy()
                    s_values = valores_objetivo[:, feature_idx]
                    
                    correlation = 0
                    if np.std(f_values) > 0 and np.std(s_values) > 0:
                        correlation = np.corrcoef(f_values, s_values)[0, 1]
                    
                    if correlation > 0.3:
                        impacto = "Positivo (A mayor valor de la variable, mayor es la predicción)"
                    elif correlation < -0.3:
                        impacto = "Negativo (A mayor valor de la variable, menor es la predicción)"
                    else:
                        impacto = "Complejo/No lineal (El impacto depende de rangos específicos)"
                    
                    top_features_summary[feature] = {
                        "importancia_media_absoluta": round(importance, 4),
                        "direccion_impacto": impacto
                    }

                if hasattr(shap_values, 'base_values') and shap_values.base_values is not None:
                    b_val = shap_values.base_values[0]
                    base_value = float(b_val[1]) if isinstance(b_val, (np.ndarray, list)) else float(b_val)
                else:
                    base_value = None

                result = {
                    "analisis": f"Top {top_k} variables más importantes a nivel global",
                    "base_value_promedio_dataset": round(base_value, 4) if base_value is not None else "N/A",
                    "variables": top_features_summary
                }

                try:
                    sv_for_plot = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values
                    sv_for_plot.feature_names = self.labels
                    shap.plots.bar(sv_for_plot, show=False)
                    result["plot_path"] = self._save_plot("shap_global_bar")
                except Exception:
                    pass
                
                return json.dumps(result, ensure_ascii=False)
                
        except Exception as e:
            return json.dumps({"error": f"No se pudo calcular la explicación global: {str(e)}"}, ensure_ascii=False)
    
    
    
    def tool_shap_explain_local_prediction(self, instance_data: dict) -> str:
        """
        Útil para explicar por qué el modelo tomó una decisión específica para un solo registro (tabla o imagen).
        Llama a esta función cuando el usuario pregunte "¿Por qué se ha predicho X para este cliente/imagen?"
        
        Args:
            instance_data (dict): Para tablas: diccionario con {columna: valor}.
                                Para imágenes: diccionario con {"file_path": "ruta/a/la/imagen.jpg"}.
        Returns:
            str: Un JSON en formato string con los resultados del análisis y la ruta del gráfico generado.
        """
        try:
            if self.task_type == "image":
                file_path = str(instance_data.get("filepath")).split('filepath=')[1]
                file_path = urllib.parse.unquote(file_path)

                if not file_path or not os.path.exists(file_path):
                    return json.dumps({"error": f"No se proporcionó una ruta de imagen válida o el archivo no existe: {file_path}"})

                # Recuperar el tamaño de imagen objetivo desde los metadatos de la sesión
                img_size = (128, 128)
                if hasattr(self, "dataset_metadata") and self.dataset_metadata:
                    img_size = self.dataset_metadata.get("image_size", (128, 128))

                # Cargar la imagen y normalizarla exactamente como lo hace tu pipeline de entrenamiento
                img = Image.open(file_path).convert("RGB").resize(img_size)
                img_array = np.array(img, dtype=np.float32) / 255.0
                img_batch = np.expand_dims(img_array, axis=0)  # Convertir a lote (1, H, W, C)

                # Calcular los valores SHAP usando tu particionador/explainer de imágenes
                shap_values = self.explainer(img_batch)

                result = {
                    "tipo_dataset": "imagen",
                    "mensaje": "Explicación visual de píxeles generada con éxito.",
                    "file_path_analizado": file_path
                }

                # Intentar extraer nombres de las clases si el explainer las tiene mapeadas
                if hasattr(shap_values, "output_names") and shap_values.output_names:
                    result["clases_analizadas"] = list(shap_values.output_names)

                # RENDERIZAR GRÁFICO DE IMAGEN SHAP
                try:
                    # Limpiar cualquier figura previa para evitar solapamientos
                    plt.clf()
                    
                    # Dependiendo de la versión de SHAP y el tipo de explainer (ej. PartitionExplainer)
                    if hasattr(shap.plots, "image"):
                        # API Moderna: espera el objeto Explanation de la primera muestra [0]
                        shap.plots.image(shap_values[0], show=False)
                    else:
                        # API Clásica fallback
                        shap.image_plot(shap_values, img_batch, show=False)
                    
                    result["plot_path"] = self._save_plot("shap_local_image")
                    plt.close()
                except Exception as plot_err:
                    result["plot_error"] = f"No se pudo renderizar el mapa de calor de la imagen: {str(plot_err)}"

                return json.dumps(result)

            else:
                df_instance = pd.DataFrame([instance_data])
                shap_values = self.explainer(df_instance)
                
                if len(shap_values.shape) == 3:
                    s_values = shap_values.values[0, :, 1]
                else:
                    s_values = shap_values.values[0]

                if hasattr(shap_values, 'base_values') and shap_values.base_values is not None:
                    b_val = shap_values.base_values[0]
                    if isinstance(b_val, (np.ndarray, list)):
                        base_value = float(b_val[1])
                    else:
                        base_value = float(b_val)
                else:
                    base_value = 0.0
                    
                prediction = float(s_values.sum() + base_value)
                
                contributions = {
                    feature: float(value) 
                    for feature, value in zip(df_instance.columns, s_values)
                }
                
                top_contributions = dict(
                    sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
                )
                
                result = {
                    "tipo_dataset": "tabular",
                    "base_value_promedio_dataset": round(base_value, 4),
                    "prediccion_final_modelo": round(prediction, 4),
                    "top_5_variables_impacto": top_contributions
                }

                try:
                    plt.clf()
                    sv_for_plot = shap_values[:, :, 1] if len(shap_values.shape) == 3 else shap_values
                    shap.plots.waterfall(sv_for_plot[0], show=False)
                    result["plot_path"] = self._save_plot("shap_local_waterfall")
                    plt.close()
                except Exception:
                    pass
                
                return json.dumps(result)
                
        except Exception as e:
            return json.dumps({"error": f"No se pudo calcular la explicación local: {str(e)}"})

    def tool_lime_explain_local_prediction(self, instance_data: dict) -> str:
        """
        Útil para explicar por qué el modelo tomó una decisión específica para un solo registro o individuo
        usando el método LIME (Local Interpretable Model-agnostic Explanations).
        Soporta datasets tanto tabulares como de imágenes.
        
        Args:
            instance_data (dict): Para tablas: {nombre_columna: valor}.
                                Para imágenes: {"file_path": "ruta/a/la/imagen.jpg"}.
        Returns:
            str: Un JSON en formato string con los impactos locales y la ruta del gráfico explicativo.
        """
        try:
            if self.task_type == "image":
                from skimage.segmentation import mark_boundaries

                file_path = str(instance_data.get("filepath")).split('filepath=')[1]
                file_path = urllib.parse.unquote(file_path)
                if not file_path or not os.path.exists(file_path):
                    return json.dumps({"error": f"No se proporcionó una ruta de imagen válida o el archivo no existe: {file_path}"})

                # Recuperar el tamaño de imagen objetivo desde los metadatos
                img_size = (128, 128)
                if hasattr(self, "dataset_metadata") and self.dataset_metadata:
                    img_size = self.dataset_metadata.get("image_size", (128, 128))

                # Cargar y redimensionar la imagen
                img = Image.open(file_path).convert("RGB").resize(img_size)
                img_array = np.array(img) # LIME Image trabaja nativamente con arrays enteros (0-255) o floats

                # NOTA: Asegúrate de que tu '_get_lime_predict_fn()' para imágenes acepte 
                # un lote de tensores con forma (N, H, W, C) y devuelva las probabilidades (N, clases)
                predict_fn = self._get_lime_predict_fn()

                # Calcular la explicación (num_samples=200 para evitar congelar la CPU en servidores web)
                explanation = self.lime_explainer.explain_instance(
                    img_array, 
                    predict_fn, 
                    top_labels=5, 
                    hide_color=0, 
                    num_samples=200
                )

                # Obtener el índice de la clase con mayor probabilidad asignada por el modelo
                top_label = explanation.top_labels[0]
                
                # Mapear el ID numérico al nombre real de la clase si está disponible
                clase_predicha = str(top_label)
                if hasattr(self, "dataset_metadata") and self.dataset_metadata:
                    classes = self.dataset_metadata.get("classes", [])
                    if top_label < len(classes):
                        clase_predicha = classes[top_label]

                result = {
                    "tipo_dataset": "imagen",
                    "clase_predicha_index": int(top_label),
                    "clase_predicha_nombre": clase_predicha,
                    "mensaje": "Se han identificado los superpíxeles de la imagen que actúan como pros y contras para esta predicción."
                }

                # Generar el gráfico de superpíxeles con sus fronteras de decisión
                try:
                    # positive_only=False pintará en verde lo que apoya a la clase y en rojo lo que va en contra
                    temp, mask = explanation.get_image_and_mask(
                        top_label, 
                        positive_only=False, 
                        num_features=5, 
                        hide_rest=False
                    )
                    
                    fig, ax = plt.subplots(figsize=(6, 6))
                    ax.imshow(mark_boundaries(temp, mask))
                    ax.axis("off")
                    ax.set_title(f"Explicación LIME: {clase_predicha}")
                    
                    # Guardar el gráfico siguiendo tu estructura exacta de nombres
                    path = os.path.join(self.plots_dir, f"lime_local_image_{uuid.uuid4().hex[:8]}.png")
                    fig.savefig(path, bbox_inches="tight")
                    plt.close(fig)
                    
                    result["plot_path"] = path
                except Exception as plot_err:
                    result["plot_error"] = f"Error al renderizar la máscara de superpíxeles LIME: {str(plot_err)}"

                return json.dumps(result)

            else:
                # =================================================================
                # RAMA B: EXPLICACIÓN LOCAL TABULAR (TU CÓDIGO ORIGINAL)
                # =================================================================
                missing = [f for f in self.labels if f not in instance_data]
                if missing:
                    return json.dumps({"error": f"Faltan las siguientes variables en instance_data: {missing}"})

                instance = np.array([instance_data[feature] for feature in self.labels])

                explanation = self.lime_explainer.explain_instance(
                    data_row=instance,
                    predict_fn=self._get_lime_predict_fn(),
                    num_features=5,
                )

                feature_contributions = {rule: round(float(weight), 4) for rule, weight in explanation.as_list()}

                if hasattr(self.model, "predict_proba"):
                    prediction = float(self.model.predict_proba([instance])[0][1])
                else:
                    prediction = float(self.model.predict([instance])[0])

                result = {
                    "tipo_dataset": "tabular",
                    "prediccion_modelo": round(prediction, 4),
                    "top_5_variables_impacto": feature_contributions,
                }

                try:
                    fig = explanation.as_pyplot_figure()
                    path = os.path.join(self.plots_dir, f"lime_local_{uuid.uuid4().hex[:8]}.png")
                    fig.savefig(path, bbox_inches="tight")
                    plt.close(fig)
                    result["plot_path"] = path
                except Exception:
                    pass

                return json.dumps(result)

        except Exception as e:
            return json.dumps({"error": f"No se pudo calcular la explicación LIME local: {str(e)}"})
        
    def tool_anchor_explain_local_prediction(self, instance_data: dict) -> str:
        """
        Útil para explicar por qué el modelo tomó una decisión específica para un solo registro
        usando el método Anchors (AnchorTabular de alibi). Devuelve reglas del tipo
        "SI variable_A > X Y variable_B <= Y → predicción es Z con un 95 % de precisión".
        Llama a esta función cuando el usuario pregunte "¿Por qué se ha predicho X para este
        cliente?" y quiera una explicación basada en reglas comprensibles.

        Args:
            instance_data (dict): Un diccionario con los nombres de las columnas y los valores
                                para el individuo a predecir.
        Returns:
            str: Un JSON en formato string con las reglas ancla, su precisión y su cobertura.
        """
        try:
            missing = [f for f in self.labels if f not in instance_data]
            if missing:
                return json.dumps({"error": f"Faltan las siguientes variables en instance_data: {missing}"})

            instance = np.array([instance_data[feature] for feature in self.labels])

            # threshold: confianza mínima para aceptar el ancla (por defecto 0.95)
            explanation = self.anchor_explainer.explain(instance)

            if hasattr(self.model, "predict_proba"):
                prediction = float(self.model.predict_proba([instance])[0][1])
            else:
                prediction = float(self.model.predict([instance])[0])

            result = {
                "prediccion_modelo": round(prediction, 4),
                "reglas_ancla": list(explanation.anchor),
                "precision": round(float(explanation.precision), 4),
                "cobertura": round(float(explanation.coverage), 4),
            }

            return json.dumps(result)

        except Exception as e:
            return json.dumps({"error": f"No se pudo calcular la explicación Anchor: {str(e)}"})

    def tool_shap_lime_explain_local_prediction(self, instance_data: dict) -> str:
        shap_result = json.loads(self.tool_shap_explain_local_prediction(instance_data))
        lime_result = json.loads(self.tool_lime_explain_local_prediction(instance_data))
        anchor_result = json.loads(self.tool_anchor_explain_local_prediction(instance_data))
        return json.dumps({"shap": shap_result, "lime": lime_result, "anchor": anchor_result})

    def tool_prototype(self, k : int = 5) -> str:
        """Genera datos bien representados (prototipos) del dataset

        Args:
            k (int, optional): Número de prototipos a crear. Por defecto es 5.

        Returns:
            str: Array de los valores prototípicos
        """
        return str(self.critic.select_prototypes(k))