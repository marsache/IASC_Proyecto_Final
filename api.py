"""
FastAPI backend for the XAI Chat Assistant.

Flow
----
1. POST /api/upload   – parse CSV columns (client sends the raw file bytes)
2. POST /api/initialize – start pipeline in a background thread; returns session_id
3. GET  /api/status/{session_id} – poll until status == "ready" | "error"
4. POST /api/chat     – ask a question; returns response text + optional plot URL
5. GET  /api/plot/{filename} – serve a generated XAI plot image
6. DELETE /api/session/{session_id} – clean up session state

Run:  python api.py
      (or: uvicorn api:app --reload --port 7860)
"""

import json
import os
import shutil
import sys
import tempfile
import uuid
import base64
from pathlib import Path
from threading import Lock, Thread
import zipfile
import urllib.parse

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from langchain_community.chat_message_histories import ChatMessageHistory

# Make sure relative imports (config, utils, …) resolve correctly.
sys.path.insert(0, os.path.dirname(__file__))

from agents.model_selector_agent import recommend_best_models
from agents.xai_agent import setup_xai_agent
from config import getSettings          # noqa: F401  (imported for side-effects)
from tools.XAIToolkit import XAIToolkit
from utils.dataset_utils import preprocess_dataset
from utils.models import build_and_train_recommended_models, generate_model_info, try_load_model, save_model

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_PLOTS_DIR = os.path.realpath(os.path.join(_BASE_DIR, "plots"))
_STATIC_DIR = os.path.join(_BASE_DIR, "static")
os.makedirs(_PLOTS_DIR, exist_ok=True)
os.makedirs(_STATIC_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# In-memory session store
# ---------------------------------------------------------------------------
_sessions: dict = {}
_sessions_lock = Lock()

store = {}
def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


def _update_session(session_id: str, **kwargs) -> None:
    with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id].update(kwargs)


# ---------------------------------------------------------------------------
# Pipeline runner (executes in a background thread)
# ---------------------------------------------------------------------------
def _run_full_pipeline(session_id: str, dataset_path: str, target: str, is_image: bool) -> None:
    try:
        _update_session(session_id, progress=0.0, message="Preprocesando datos…")
        
        # ARREGLO: Añadimos 'df_raw' como valor de retorno esperado.
        # Asumimos que preprocess_dataset lee los datos y nos devuelve el DataFrame original 
        # (o None en caso de ser un dataset de imágenes).
        X_train, X_test, y_train, y_test, metadata_json, scaler, actual_target, df_raw = preprocess_dataset(dataset_path, target)
        metadata = json.loads(metadata_json)

        loaded, model, task_type = try_load_model(dataset_path=dataset_path)
        if not loaded:
            _update_session(session_id, progress=0.40, message="Seleccionando los mejores modelos…", dataset_metadata=metadata_json)
            recommendations_json = recommend_best_models(metadata_json)
            task_type = json.loads(recommendations_json).get("task_type", "classification")

            _update_session(session_id, progress=0.60, message="Entrenando modelos…")
            trained_models = build_and_train_recommended_models(recommendations_json, X_train, y_train)
            
            if not trained_models:
                raise ValueError("No se pudo entrenar ningún modelo. Revisa el log de consola.")

            best = trained_models[0]
            model = best["model_object"]

            _update_session(session_id, progress=0.65, message="Guardando modelos…")
            save_model(dataset_path, task_type, model)
            _update_session(session_id, progress=0.70, message="Modelos guardados")

        _update_session(session_id, progress=0.80, message="Inicializando herramientas XAI…")
        model_info, y_pred = generate_model_info(model, X_test, y_test, task_type)

        toolkit = XAIToolkit(
            model=model,
            x_test=X_test,
            dataset_metadata=metadata,
            dataset=df_raw, # Pasará el df limpio para tabular, y None para imágenes
            target=target,
        )

        _update_session(session_id, progress=0.95, message="Configurando agente XAI…")
        agent_executor = setup_xai_agent(metadata, model_info, toolkit, get_history)

        if not is_image:
            df_clean = df_raw
            _update_session(session_id, progress=0.98, message="Generando predicciones para el explorador...")
            
            X_full_scaled = scaler.transform(df_clean.drop(columns=[target])) 
            y_pred = model.predict(X_full_scaled) 
            df_clean['Target_Real'] = df_clean[target]
            df_clean['Target_Predicho'] = y_pred
                        
            with _sessions_lock:
                session = _sessions.get(session_id)
                
            augmented_csv_path = os.path.join(session["_tmp_dir"], "augmented_dataset.csv")
            df_clean.to_csv(augmented_csv_path, index=False)
        else:
            _update_session(session_id, progress=0.95, message="Calculando predicciones de imágenes para el explorador...")
            
            with _sessions_lock:
                session = _sessions.get(session_id)
                
            def keras_predict_fn(keras_model, paths):
                import tensorflow as tf
                import numpy as np
                from PIL import Image
                
                preds = []
                target_size = (128, 128) # Asegúrate que coincide con el tamaño con el que entrenaste
                for p in paths:
                    img = Image.open(p).convert('RGB').resize(target_size)
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    img_array = np.expand_dims(img_array, axis=0) # Crear batch de 1
                    
                    raw_pred = keras_model.predict(img_array, verbose=0)
                    class_idx = np.argmax(raw_pred, axis=1)[0]
                    
                    # Usa los metadatos para transformar el [0] de vuelta a texto ["gato"]
                    class_name = metadata.get("classes", [])[class_idx]
                    preds.append(class_name)
                    
                return preds

            # Ejecutamos nuestra nueva función externalizada (del archivo Python generado)
            run_image_dataset_inference(
                session_id=session_id,
                tmp_dir=session["_tmp_dir"],
                model=model,
                predict_fn=keras_predict_fn,
                update_session_fn=_update_session
            )

        _update_session(
            session_id,
            status="ready", progress=1.0, message="¡Pipeline inicializado correctamente!",
            agent=agent_executor, model_info=model_info,
        )

    except Exception as exc:
        print(exc)
        _update_session(session_id, status="error", message=str(exc), error=str(exc))

from typing import Any, Callable, List

def run_image_dataset_inference(
    session_id: str,
    tmp_dir: str,
    model: Any,
    predict_fn: Callable[[Any, List[str]], List[Any]],
    update_session_fn: Callable[[str, float, str], None]
) -> str:
    """
    Scans the 'dataset_images' directory, runs batch inference using the provided 
    model and prediction function, and generates a unified 'augmented_dataset.csv'.
    
    This avoids doing heavy disk scanning (os.walk) during API pagination calls
    and populates the 'Target_Predicho' column with real model predictions.
    """
    img_folder = os.path.join(tmp_dir, "dataset_images")
    augmented_csv_path = os.path.join(tmp_dir, "augmented_dataset.csv")
    
    if os.path.isfile(augmented_csv_path):
        return augmented_csv_path

    if not os.path.exists(img_folder):
        raise FileNotFoundError(f"La carpeta de imágenes no existe: {img_folder}")
        
    update_session_fn(session_id, progress=0.95, message="Escaneando directorio de imágenes...")
    
    # 1. Gather all image files and infer their real targets from the subfolder names
    image_records = []
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    
    for root, _, files in os.walk(img_folder):
        for file in files:
            if file.lower().endswith(valid_extensions):
                full_path = os.path.join(root, file)
                
                # Deduce the real class/target from the immediate parent directory
                parent_folder = os.path.basename(root)
                target_real = parent_folder if parent_folder != "dataset_images" else "Desconocido"
                
                image_records.append({
                    "file_path": full_path,
                    "Target_Real": target_real
                })
                
    if not image_records:
        raise ValueError(f"No se encontraron imágenes válidas en la carpeta: {img_folder}")
        
    df_images = pd.DataFrame(image_records)
    # Sort paths alphabetically to guarantee stable pagination sequences
    df_images = df_images.sort_values("file_path").reset_index(drop=True)
    
    update_session_fn(session_id, progress=0.96, message=f"Ejecutando inferencia en {len(df_images)} imágenes...")
    
    # 2. Execute model inference using the custom injected prediction function
    # The predict_fn should accept (model, list_of_paths) and return a list of predictions
    try:
        file_paths = df_images["file_path"].tolist()
        predictions = predict_fn(model, file_paths)
        
        if len(predictions) != len(file_paths):
            raise ValueError("La función de predicción debe devolver exactamente el mismo número de elementos que de rutas de entrada.")
            
        df_images["Target_Predicho"] = predictions
        
    except Exception as e:
        df_images["Target_Predicho"] = "Error de Inferencia"
        df_images.to_csv(augmented_csv_path, index=False)
        raise RuntimeError(f"Error crítico durante la inferencia del modelo: {e}")
        
    # 3. Save the pre-computed augmented dataset to disk
    df_images.to_csv(augmented_csv_path, index=False)
    update_session_fn(session_id, progress=0.98, message="Explorador de imágenes e inferencia preparados con éxito.")
    
    return augmented_csv_path


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="XAI Chat API", version="1.0.0")


# ------------------------------------------------------------------
# Root → serve the HTML frontend
# ------------------------------------------------------------------
@app.get("/")
def index() -> FileResponse:
    html_path = os.path.join(_STATIC_DIR, "index.html")
    if not os.path.isfile(html_path):
        raise HTTPException(404, "Frontend not found. Make sure static/index.html exists.")
    return FileResponse(html_path, media_type="text/html")


# ------------------------------------------------------------------
# Upload CSV → return column names + HTML preview
# ------------------------------------------------------------------
@app.post("/api/upload")
async def upload_dataset(file: UploadFile = File(...)) -> JSONResponse:
    filename = (file.filename or "").lower()
    if not (filename.endswith(".csv") or filename.endswith(".zip")):
        raise HTTPException(400, "Solo se aceptan archivos CSV o ZIP.")

    raw = await file.read()

    # --- RAMA ZIP (IMÁGENES) ---
    if filename.endswith(".zip"):
        tmp_dir = tempfile.mkdtemp()
        zip_path = os.path.join(tmp_dir, "temp_upload.zip")
        
        with open(zip_path, "wb") as f:
            f.write(raw)
            
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
                
            # Escanear subcarpetas (clases)
            classes = []
            total_images = 0
            valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
            tmp_dir = tmp_dir +"\\" +filename.split(".")[0]
            for entry in os.listdir(tmp_dir):
                full_path = os.path.join(tmp_dir, entry)
                if os.path.isdir(full_path):
                    # Contar imágenes válidas en la subcarpeta
                    images = [img for img in os.listdir(full_path) if img.lower().endswith(valid_exts)]
                    if images:
                        classes.append(entry)
                        total_images += len(images)
            
            preview_html = (
                f"<div style='padding: 1rem; background: #f8f9fa; border-radius: 8px; text-align: center;'>"
                f"  <strong>🖼️ Dataset de Imágenes Detectado</strong><br><br>"
                f"  <span style='color: #495057;'>Clases encontradas ({len(classes)}):</span> <b>{', '.join(classes)}</b><br>"
                f"  <span style='color: #495057;'>Total de imágenes:</span> <b>{total_images}</b>"
                f"</div>"
            )
            
            return JSONResponse({
                "columns": ["label"], # Columna objetivo por defecto
                "preview_html": preview_html,
                "rows": total_images,
                "cols": 2,
                "is_image": True # Flag para el frontend
            })
            
        except zipfile.BadZipFile:
            raise HTTPException(400, "El archivo ZIP está corrupto o es inválido.")
            
    # --- RAMA CSV (TABULAR) ORIGINAL ---
    else:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".csv")
        try:
            with os.fdopen(tmp_fd, "wb") as fh:
                fh.write(raw)
            df = pd.read_csv(tmp_path)
        except Exception as exc:
            raise HTTPException(400, f"Error al leer el archivo CSV: {exc}") from exc
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        columns = df.columns.tolist()
        preview_html = df.head(5).to_html(classes="preview-table", border=0, index=False, escape=True)

        return JSONResponse({
            "columns": columns,
            "preview_html": preview_html,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "is_image": False
        })

# ------------------------------------------------------------------
# Initialize pipeline
# ------------------------------------------------------------------
@app.post("/api/initialize")
async def initialize(
    file: UploadFile = File(...),
    target: str = Form("label"), # Si viene vacío, por defecto "label"
) -> JSONResponse:
    filename = (file.filename or "").lower()
    if not (filename.endswith(".csv") or filename.endswith(".zip")):
        raise HTTPException(400, "Solo se aceptan archivos CSV o ZIP.")

    raw = await file.read()

    u = uuid.uuid5(uuid.NAMESPACE_DNS, file.filename)
    session_id = base64.b32encode(u.bytes).decode("ascii")[:8]
    tmp_dir = Path(tempfile.gettempdir()) / f"xai_session_{session_id}"
    tmp_dir.mkdir(exist_ok=True)

    dataset_path_to_pass = ""
    is_image_dataset = filename.endswith(".zip")

    if is_image_dataset:
        zip_path = os.path.join(tmp_dir, "dataset.zip")
        with open(zip_path, "wb") as fh:
            fh.write(raw)
            
        extract_dir = os.path.join(tmp_dir, "dataset_images")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        dataset_path_to_pass = extract_dir + "\\" +filename.split('.')[0] # Pasamos el DIRECTORIO al pipeline
        target = "label" # Forzamos el target
    else:
        csv_path = os.path.join(tmp_dir, "dataset.csv")
        with open(csv_path, "wb") as fh:
            fh.write(raw)
        dataset_path_to_pass = csv_path

    with _sessions_lock:
        _sessions[session_id] = {
            "status": "running", "progress": 0.0,
            "message": "Iniciando pipeline…", "agent": None,
            "model_info": None, "error": None, "_tmp_dir": tmp_dir,
        }

    # Le pasamos a la función asíncrona un nuevo parámetro is_image
    thread = Thread(
        target=_run_full_pipeline,
        args=(session_id, dataset_path_to_pass, target.strip(), is_image_dataset),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"session_id": session_id})

# ------------------------------------------------------------------
# Poll pipeline status
# ------------------------------------------------------------------
@app.get("/api/status/{session_id}")
def get_status(session_id: str) -> JSONResponse:
    with _sessions_lock:
        session = _sessions.get(session_id)

    if session is None:
        raise HTTPException(404, "Sesión no encontrada.")

    return JSONResponse(
        {
            "status": session["status"],
            "progress": session["progress"],
            "message": session["message"],
            "dataset_metadata": session.get("dataset_metadata"),
            "model_info": session.get("model_info"),
        }
    )


# ------------------------------------------------------------------
# Chat
# ------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    message: str

def _collect_plot_paths(obj, collector: list[str]) -> None:
    if isinstance(obj, dict):
        plot_path = obj.get("plot_path")
        if isinstance(plot_path, str):
            collector.append(plot_path)
        for value in obj.values():
            _collect_plot_paths(value, collector)
    elif isinstance(obj, list):
        for item in obj:
            _collect_plot_paths(item, collector)


@app.post("/api/chat")
def chat(body: ChatRequest) -> JSONResponse:
    if not body.message.strip():
        raise HTTPException(400, "El mensaje no puede estar vacío.")

    with _sessions_lock:
        session = _sessions.get(body.session_id)

    if session is None:
        raise HTTPException(404, "Sesión no encontrada.")
    if session["status"] != "ready":
        raise HTTPException(
            400,
            "El pipeline aún no está listo. Espera a que la inicialización termine.",
        )

    agent = session["agent"]

    try:
        result = agent.invoke({"input": body.message.strip()}, config = {"configurable": {"session_id": body.session_id}})
        print("RESULT:", result)
        response_text = result.get("output", str(result))

        # Collect all valid plot paths produced during this turn
        plot_urls: list[str] = []
        seen_plot_urls: set[str] = set()
        for _action, observation in result.get("intermediate_steps", []):
            try:
                print("OBSERVATION:", observation)

                # -----------------------------
                # Convertir observation correctamente
                # -----------------------------
                if isinstance(observation, str):
                    obs_dict = json.loads(observation)

                elif isinstance(observation, dict):
                    obs_dict = observation

                else:
                    obs_dict = {}

                print("OBS_DICT:", obs_dict)

                # -----------------------------
                # Buscar plot_path(s)
                # -----------------------------
                raw_paths: list[str] = []
                _collect_plot_paths(obs_dict, raw_paths)

                print("RAW PATHS:", raw_paths)

                # -----------------------------
                # Validar rutas y generar URLs
                # -----------------------------
                for raw_path in raw_paths:

                    safe_path = os.path.realpath(raw_path)

                    print("SAFE PATH:", safe_path)

                    if (
                        safe_path.startswith(_PLOTS_DIR + os.sep)
                        and os.path.isfile(safe_path)
                    ):

                        plot_url = f"/api/plot/{os.path.basename(safe_path)}"

                        print("PLOT URL:", plot_url)

                        if plot_url not in seen_plot_urls:
                            seen_plot_urls.add(plot_url)
                            plot_urls.append(plot_url)

            except Exception as e:
                print("ERROR LEYENDO INTERMEDIATE STEP:", e)

        return JSONResponse(
            {
                "response": response_text,
                "plot_urls": plot_urls,
                "plot_url": plot_urls[0] if plot_urls else None,
            }
        )

    except Exception as exc:
        raise HTTPException(500, f"Error al procesar la consulta: {exc}") from exc


# ------------------------------------------------------------------
# Serve XAI plots
# ------------------------------------------------------------------
@app.get("/api/plot/{filename}")
def get_plot(filename: str) -> FileResponse:
    # Reject any path-traversal attempt in the filename segment
    if os.sep in filename or "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(400, "Nombre de archivo inválido.")

    full_path = os.path.join(_PLOTS_DIR, filename)
    real_path = os.path.realpath(full_path)

    if not real_path.startswith(_PLOTS_DIR + os.sep) or not os.path.isfile(real_path):
        raise HTTPException(404, "Plot no encontrado.")

    return FileResponse(real_path, media_type="image/png")

@app.get("/api/dataset/{session_id}")
def get_dataset(session_id: str, page: int = 1, size: int = 50) -> JSONResponse:
    with _sessions_lock:
        session = _sessions.get(session_id)

    if session is None:
        raise HTTPException(404, "Sesión no encontrada.")
    
    tmp_dir = session.get("_tmp_dir")
    if not tmp_dir:
        raise HTTPException(400, "Carpeta de datos no disponible.")

    # Ahora AMBOS pipelines (tabular e imagen) generan este archivo. ¡Magia!
    csv_path = os.path.join(tmp_dir, "augmented_dataset.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(400, "Dataset exploratorio no está listo aún.")

    try:
        df = pd.read_csv(csv_path)
        
        # --- LÓGICA COMÚN: PAGINACIÓN ---
        total_rows = int(df.shape[0])
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        
        # Extraer solo la página solicitada
        df_page = df.iloc[start_idx:end_idx].copy()
        
        # --- INTERCEPTOR DE IMÁGENES PARA EL FRONTEND ---
        image_keywords = ["image", "image_path", "file_path", "path", "url"]
        image_col = next((col for col in df.columns if col.lower() in image_keywords), None)
        
        records = df_page.to_dict(orient="records")
        
        if image_col:
            for row in records:
                original_path = str(row[image_col])
                
                # Encriptamos la ruta para que pase por nuestro endpoint /api/image/
                if not original_path.startswith("http"):
                    safe_path = urllib.parse.quote(original_path, safe="")
                    row[image_col] = f"/api/image/{session_id}?filepath={safe_path}"

        return JSONResponse({
            "total_rows": total_rows,
            "page": page,
            "size": size,
            "columns": df.columns.tolist(),
            "data": records
        })
        
    except HTTPException:
        raise # Dejar pasar excepciones controladas HTTP
    except Exception as exc:
        raise HTTPException(500, f"Error al leer o procesar el dataset: {exc}")
    
# --- NUEVO ENDPOINT PARA SERVIR LAS IMÁGENES ---
@app.get("/api/image/{session_id}")
def serve_image(session_id: str, filepath: str):
    """
    Recibe la ruta encriptada desde el frontend, la lee del disco duro 
    y se la envía al navegador para que la etiqueta <img> la pueda renderizar.
    """
    # 1. Decodificar la ruta de vuelta a su formato original (ej. C:/... o /var/...)
    real_path = urllib.parse.unquote(filepath)
    
    # 2. Comprobar que realmente existe
    if not os.path.exists(real_path):
        raise HTTPException(404, f"Imagen no encontrada en el disco: {real_path}")
        
    # 3. Devolver el archivo directamente (FastAPI asigna el Content-Type correcto automáticamente)
    return FileResponse(real_path)

# ------------------------------------------------------------------
# Delete / reset session
# ------------------------------------------------------------------
@app.delete("/api/session/{session_id}")
def delete_session(session_id: str) -> JSONResponse:
    with _sessions_lock:
        session = _sessions.pop(session_id, None)

    # Clean up the temporary CSV directory if it still exists
    if session:
        tmp_dir = session.get("_tmp_dir")
        if tmp_dir and os.path.isdir(tmp_dir):
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return JSONResponse({"ok": True})


# ------------------------------------------------------------------
# Mount static files last (so API routes take precedence)
# ------------------------------------------------------------------
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


# ------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
