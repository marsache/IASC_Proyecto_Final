# IASC_Proyecto_Final

## Autores
**María Sachez Carrasco**

**David Rivera Martínez**

**Pablo Sánchez Martín**

# Guía de Inicio y Uso del Proyecto

Este proyecto está basado en **uv**. A continuación, se detallan los pasos necesarios para instalar, ejecutar e interactuar con la aplicación.

## Instalación y Requisitos

Asegúrate de tener instalado el paquete `uv` en tu máquina. Para configurar el entorno, realiza lo siguiente:

1. Descarga o clona este repositorio en tu equipo local.
2. Abre una consola de comandos en la raíz del proyecto.
3. Ejecuta el siguiente comando para instalar y sincronizar todas las dependencias:
   ```bash
   uv sync

**Ejecución**

Para iniciar la aplicación, ejecuta el siguiente comando en la raíz del proyecto:
Bash

uv run api.py

Una vez que el proceso se haya iniciado correctamente, el servidor estará activo y disponible en tu navegador web a través de la siguiente dirección: http://localhost:7860

**Guía de Uso**
1. Inicialización del Pipeline

Al acceder a la interfaz desde el navegador, se te presentará un menú inicial donde debes configurar los datos de la sesión:

    Adjuntar Dataset: Sube el archivo de datos con el que vas a trabajar.

    Variable Objetivo: Especifica la variable objetivo que se va a analizar.

    Inicializar: Haz clic en el botón "Inicializar pipeline" y espera a que finalice la configuración.

    Importante: El proyecto cuenta con un sistema de persistencia de datos. Si utilizas un dataset que ya ha sido procesado anteriormente, el tiempo de carga será considerablemente menor. En el caso de datasets de imágenes de gran volumen, la configuración inicial puede tardar hasta 30 segundos.

2. Interacción con la Herramienta

Una vez el pipeline esté listo, accederás a una interfaz interactiva y conversacional que te permitirá:

    Preguntar al Agente: Realizar consultas de forma directa mediante lenguaje natural.

    Explicar Instancias: Seleccionar un registro o instancia concreta del dataset cargado para obtener una explicación detallada de su comportamiento.

    Evaluar Nuevos Casos: Dirígete a la pestaña 'Nuevo Caso' (situada en la parte superior derecha de la ventana) para ingresar y analizar datos completamente nuevos que no pertenezcan al dataset original.
