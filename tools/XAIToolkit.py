import shap
import json

import pandas as pd
import numpy as np
from lime import lime_tabular

from dice_ml import Data, Model, Dice

class XAIToolkit:
    def __init__(self, model, x_test, dataset_metadata, dataset, target):
        # 1. Asignaciones básicas
        self.model = model
        self.x_test = x_test
        self.dataset_metadata = dataset_metadata
        self.target = target
        
        # [Nota] Si quieres los nombres cortos de las variables (ej: "edad", "salario"), 
        # debes usar .keys(). Si usas .values(), obtendrás las descripciones semánticas 
        # largas que generó el agente, lo cual puede romper los gráficos de SHAP.
        self.labels = list(self.dataset_metadata['features'].keys())

        d_dice = Data(
            dataframe=dataset,
            continuous_features=self.labels,
            outcome_name=target
        )

        m_dice = Model(model=model, backend='sklearn')

        self.dice = Dice(d_dice, m_dice, method='genetic')
        
        # 2. Enrutamiento Inteligente del Explainer (SHAP)
        model_name = type(self.model).__name__
        
        if "RandomForest" in model_name or "GradientBoosting" in model_name:
            # TreeExplainer es el más rápido y exacto para algoritmos basados en árboles
            self.explainer = shap.TreeExplainer(self.model)
            
        elif "Logistic" in model_name or "Linear" in model_name:
            # LinearExplainer es específico para modelos lineales, pero EXIGE un dataset de fondo (masker)
            self.explainer = shap.LinearExplainer(self.model, self.x_test)
            
        else:
            # FALLBACK para modelos de caja negra (como SVC, SVR, KNN, etc.)
            # Pasamos explícitamente la función de predicción y un resumen de x_test.
            # Usamos shap.sample para coger un fondo representativo de 100 muestras 
            # y que el cálculo no tarde horas.
            background_summary = shap.sample(self.x_test, 100)
            self.explainer = shap.Explainer(self.model.predict, background_summary)

        # 3. Inicializar el Explainer LIME
        # LIME siempre trabaja como caja negra, por lo que no necesita enrutamiento.
        # Se usa x_test como datos de fondo para estimar la distribución de las perturbaciones.
        lime_mode = "classification" if hasattr(self.model, "predict_proba") else "regression"
        self.lime_explainer = lime_tabular.LimeTabularExplainer(
            training_data=self.x_test,
            feature_names=self.labels,
            mode=lime_mode,
        )

    def _get_lime_predict_fn(self):
        """Devuelve la función de predicción adecuada para el explainer LIME."""
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba
        return lambda x: self.model.predict(x).reshape(-1, 1)
    
    def tool_dice_explain(self, instance_data : dict, features_to_vary : list[str], top_k: int = 5) -> str:
        """
        Genera contraejemplos de un registro.
        Llama a esta función si el individuo pregunta "¿Qué debo cambiar para que salga otro resultado?"

        Args:
            instance_data (dict): Un diccionario con los nombres de las columnas y los valores 
                                para el individuo a predecir.
            features_to_vary (list[str]): Lista de las variables del dataset que son posibles de variar de forma sencilla para generar los contraejemplos.
            top_k (int, optional): Número de contraejemplos a genearar. Por defectoe es 5.

        Returns:
            str: Un JSON en formato string de los contraejemplos
        """
        df_instance = pd.DataFrame([instance_data])

        dice_exp = self.dice.generate_counterfactuals(
        instance_data,
        total_CFs=top_k,
        desired_class = self.target, 
        features_to_vary=features_to_vary,
        )
        
        result = {
            "analisis" : f"{top_k} contrajemplos",
            "contraejemplos" : dice_exp.cf_examples_list[0].final_cfs_df.to_dict()
        }

        return json.dumps(result)
    

    def tool_shap_explain_global(self, top_k: int = 5) -> str:
        """
        Útil para explicar el comportamiento general del modelo y entender qué variables 
        son las más importantes a nivel global para toda la base de datos.
        Llama a esta función cuando el usuario pregunte "¿Cuáles son las variables más importantes del modelo?", 
        "¿Cómo funciona el modelo en general?" o "¿Qué factores influyen más en las predicciones?".
        
        Args:
            top_k (int, opcional): Número de variables más importantes a analizar. Por defecto es 5.
            
        Returns:
            str: Un JSON en formato string con la importancia global de las variables y su dirección de impacto.
        """
        try:
            # Calcular valores SHAP para toda la población de fondo
            shap_values = self.explainer(self.x_test)
            
            # 1. Comprobar las dimensiones de los valores SHAP
            if len(shap_values.shape) == 3:
                # Es clasificación: (muestras, variables, clases). 
                # Nos quedamos con la matriz de la clase 1 (todas las muestras, todas las variables, clase 1)
                valores_objetivo = shap_values.values[:, :, 1]
            else:
                # Es regresión o un modelo que devuelve probabilidades directamente (muestras, variables)
                valores_objetivo = shap_values.values

            # 2. Calcular la media absoluta (Ahora sí devolverá un array 1D)
            mean_abs_shap = np.abs(valores_objetivo).mean(axis=0)

            # 3. Mapear con los nombres de las columnas
            feature_names = self.labels

            global_importance = {
                feature: float(importance) 
                for feature, importance in zip(feature_names, mean_abs_shap)
            }
            
            # Ordenar para obtener el top_k de variables
            sorted_features = sorted(global_importance.items(), key=lambda x: x[1], reverse=True)[:top_k]
            
            # 2. Dirección del impacto: Relación entre el valor de la variable y su valor SHAP
            top_features_summary = {}
            for feature, importance in sorted_features:
                feature_idx = list(feature_names).index(feature)
                
                # Extraer los valores reales de la columna y sus correspondientes valores SHAP
                f_values = self.x_test[:, feature_idx]
                s_values = valores_objetivo[:, feature_idx]
                
                # Calcular correlación para determinar la dirección del impacto
                correlation = 0
                if np.std(f_values) > 0 and np.std(s_values) > 0:
                    correlation = np.corrcoef(f_values, s_values)[0, 1]
                
                # Traducir la matemática a texto que el LLM pueda usar para su narrativa
                if correlation > 0.3:
                    impacto = "Positivo (A mayor valor de la variable, mayor es la predicción)"
                elif correlation < -0.3:
                    impacto = "Negativo (A mayor valor de la variable, menor es la predicción)"
                else:
                    impacto = "Complejo/No lineal (El impacto depende de rangos específicos o interacciones)"
                
                top_features_summary[feature] = {
                    "importancia_media_absoluta": round(importance, 4),
                    "direccion_impacto": impacto
                }

            # Extraer el valor base (predicción media) de forma segura
            if hasattr(shap_values, 'base_values') and shap_values.base_values is not None:
                # Cogemos el valor base correspondiente a la primera fila evaluada
                b_val = shap_values.base_values[0]
                
                # Si b_val es un array/lista (es clasificación, tenemos un valor base por clase)
                if isinstance(b_val, (np.ndarray, list)):
                    base_value = float(b_val[1]) # Nos quedamos con la clase 1 (positiva)
                else:
                    # Es regresión, b_val ya es un número escalar
                    base_value = float(b_val)
            else:
                base_value = None

            result = {
                "analisis": f"Top {top_k} variables más importantes a nivel global",
                "base_value_promedio_dataset": round(base_value, 4) if base_value is not None else "N/A",
                "variables": top_features_summary
            }
            
            return json.dumps(result)
            
        except Exception as e:
            return json.dumps({"error": f"No se pudo calcular la explicación global: {str(e)}"})

    def tool_shap_explain_local_prediction(self, instance_data: dict) -> str:
        """
        Útil para explicar por qué el modelo tomó una decisión específica para un solo registro o individuo.
        Llama a esta función cuando el usuario pregunte "¿Por qué se ha predicho X para este cliente?"
        
        Args:
            instance_data (dict): Un diccionario con los nombres de las columnas y los valores 
                                para el individuo a predecir.
        Returns:
            str: Un JSON en formato string con la contribución de las 5 variables más importantes 
                para esta predicción específica.
        """
        try:
            # Convertir a DataFrame de una fila
            df_instance = pd.DataFrame([instance_data])
            
            # Calcular valores SHAP
            shap_values = self.explainer(df_instance)
            
            # 1. Manejo seguro de shap_values.values (Extraer la fila 0 y la clase 1 si es clasificación)
            if len(shap_values.shape) == 3:
                # Clasificación: (muestras, variables, clases). Tomamos la clase 1 para el primer registro.
                s_values = shap_values.values[0, :, 1]
            else:
                # Regresión o probabilidad directa: (muestras, variables). Tomamos el primer registro.
                s_values = shap_values.values[0]

            # 2. Manejo seguro de shap_values.base_values
            if hasattr(shap_values, 'base_values') and shap_values.base_values is not None:
                b_val = shap_values.base_values[0]
                if isinstance(b_val, (np.ndarray, list)):
                    base_value = float(b_val[1])  # Nos quedamos con la probabilidad base de la clase 1
                else:
                    base_value = float(b_val)
            else:
                base_value = 0.0 # Por seguridad matemática al sumar luego
                
            # Calcular la predicción final sumando el valor base y las contribuciones
            prediction = float(s_values.sum() + base_value)
            
            # 3. Crear un diccionario de variable: contribución usando el array seguro (s_values)
            contributions = {
                feature: float(value) 
                for feature, value in zip(df_instance.columns, s_values)
            }
            
            # Ordenar por el valor absoluto para sacar las más relevantes
            top_contributions = dict(
                sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
            )
            
            # Retornar contexto al LLM
            result = {
                "base_value_promedio_dataset": round(base_value, 4),
                "prediccion_final_modelo": round(prediction, 4),
                "top_5_variables_impacto": top_contributions
            }
            
            return json.dumps(result)
            
        except Exception as e:
            return json.dumps({"error": f"No se pudo calcular la explicación: {str(e)}"})

    def tool_lime_explain_local_prediction(self, instance_data: dict) -> str:
        """
        Útil para explicar por qué el modelo tomó una decisión específica para un solo registro o individuo
        usando el método LIME (Local Interpretable Model-agnostic Explanations).
        Llama a esta función cuando el usuario pregunte "¿Por qué se ha predicho X para este cliente?"
        y quiera una explicación basada en LIME.
        
        Args:
            instance_data (dict): Un diccionario con los nombres de las columnas y los valores 
                                para el individuo a predecir.
        Returns:
            str: Un JSON en formato string con la contribución de las 5 variables más importantes 
                para esta predicción específica según LIME.
        """
        try:
            # Validar que instance_data contiene todas las variables requeridas
            missing = [f for f in self.labels if f not in instance_data]
            if missing:
                return json.dumps({"error": f"Faltan las siguientes variables en instance_data: {missing}"})

            # Convertir el dict a array numpy respetando el orden de columnas del modelo
            instance = np.array([instance_data[feature] for feature in self.labels])

            explanation = self.lime_explainer.explain_instance(
                data_row=instance,
                predict_fn=self._get_lime_predict_fn(),
                num_features=5,
            )

            # as_list() devuelve [(descripcion_regla, contribucion), ...]
            feature_contributions = {rule: round(float(weight), 4) for rule, weight in explanation.as_list()}

            if hasattr(self.model, "predict_proba"):
                prediction = float(self.model.predict_proba([instance])[0][1])
            else:
                prediction = float(self.model.predict([instance])[0])

            result = {
                "prediccion_modelo": round(prediction, 4),
                "top_5_variables_impacto": feature_contributions,
            }

            return json.dumps(result)

        except Exception as e:
            return json.dumps({"error": f"No se pudo calcular la explicación LIME local: {str(e)}"})
        
    def tool_shap_lime_explain_local_prediction(self, instance_data: dict) -> str:
        shap_result = self.tool_shap_explain_local_prediction(instance_data)
        lime_result = self.tool_lime_explain_local_prediction(instance_data)
        return shap_result + lime_result