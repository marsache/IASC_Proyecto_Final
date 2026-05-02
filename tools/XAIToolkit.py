import shap
import json

import pandas as pd
import numpy as np

class XAIToolkit:
    def __init__(self, model, x_test, labels):
        # La parte pesada se ejecuta solo UNA vez al arrancar la app
        self.model = model
        self.explainer = shap.Explainer(self.model)
        self.x_test = x_test
        self.labels = labels

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