import json

class FallbackToolkit:
    def tool_greetings(self, user_message: str = "") -> str:
        """
        Devuelve un mensaje de saludo amigable y explica la función del agente.
        """
        answer = {"result": (
            "¡Hola! Soy tu asistente experto en Inteligencia Artificial Explicable (XAI). "
            "Mi propósito es ayudarte a entender cómo funciona nuestro modelo de Machine Learning, "
            "analizar los datos y explicarte por qué se toman ciertas predicciones, ya sea a nivel "
            "general o para casos concretos. ¿En qué te puedo ayudar hoy?"
        )}
        return json.dumps(answer, ensure_ascii=False)

    def tool_out_of_scope(self, user_message: str = "") -> str:
        """
        Maneja preguntas fuera del dominio de la aplicación (fuera de XAI/ML).
        """
        answer = {"result": ("Mis disculpas, pero soy un asistente diseñado exclusivamente para tareas de "
            "Inteligencia Artificial Explicable (XAI) y análisis de modelos. No estoy capacitado "
            "para conversar o responder preguntas sobre otros temas ajenos a este dominio.\n\n"
            "Sin embargo, estaré encantado de ayudarte con tu modelo. Aquí tienes algunos ejemplos "
            "de lo que puedes preguntarme:\n"
            "- ¿Cuáles son las variables más importantes que usa el modelo?\n"
            "- ¿Por qué el modelo ha tomado esta decisión para este cliente/imagen?\n"
            "- ¿Qué tendría que cambiar en los datos para que la predicción cambie?\n"
            "- ¿Cuáles son los perfiles más representativos de este dataset?"
        )}
        return json.dumps(answer, ensure_ascii=False)
          