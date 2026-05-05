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
from threading import Lock, Thread

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Make sure relative imports (config, utils, …) resolve correctly.
sys.path.insert(0, os.path.dirname(__file__))

from agents.model_selector_agent import recommend_best_models
from agents.xai_agent import setup_xai_agent
from config import getSettings          # noqa: F401  (imported for side-effects)
from tools.XAIToolkit import XAIToolkit
from utils.dataset_utils import preprocess_dataset
from utils.models import build_and_train_recommended_models, generate_model_info

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


def _update_session(session_id: str, **kwargs) -> None:
    with _sessions_lock:
        if session_id in _sessions:
            _sessions[session_id].update(kwargs)


# ---------------------------------------------------------------------------
# Pipeline runner (executes in a background thread)
# ---------------------------------------------------------------------------

def _run_full_pipeline(session_id: str, csv_path: str, target: str) -> None:
    try:
        _update_session(session_id, progress=0.05, message="Cargando dataset…")
        df = pd.read_csv(csv_path)

        if target not in df.columns:
            raise ValueError(f"La columna '{target}' no existe en el dataset.")

        _update_session(
            session_id,
            progress=0.15,
            message="Preprocesando datos y generando perfil semántico…",
        )
        X_train, X_test, y_train, y_test, metadata_json, scaler = preprocess_dataset(
            df, csv_path, target
        )
        metadata = json.loads(metadata_json)

        _update_session(
            session_id, progress=0.40, message="Seleccionando los mejores modelos…",
            dataset_metadata=metadata_json
        )
        recommendations_json = recommend_best_models(metadata_json)
        task_type = json.loads(recommendations_json).get("task_type", "classification")

        _update_session(session_id, progress=0.60, message="Entrenando modelos…")
        trained_models = build_and_train_recommended_models(
            recommendations_json, X_train, y_train
        )
        if not trained_models:
            raise ValueError(
                "No se pudo entrenar ningún modelo. Revisa el log de consola."
            )

        best = trained_models[0]
        model = best["model_object"]

        _update_session(
            session_id, progress=0.80, message="Inicializando herramientas XAI…"
        )
        model_info = generate_model_info(model, X_test, y_test, task_type)

        df_clean = df.dropna().reset_index(drop=True)
        toolkit = XAIToolkit(
            model=model,
            x_test=X_test,
            dataset_metadata=metadata,
            dataset=df_clean,
            target=target,
        )

        _update_session(
            session_id, progress=0.95, message="Configurando agente XAI…"
        )
        agent_executor = setup_xai_agent(metadata, model_info, toolkit)

        # --- NUEVO: Generar dataset aumentado para el explorador ---
        _update_session(
            session_id, progress=0.98, message="Generando predicciones para el explorador..."
        )
        
        # OJO: Dependiendo de cómo funcione tu `preprocess_dataset`, 
        # debes asegurarte de pasarle a model.predict() los datos ya escalados/codificados.
        # Si 'scaler' puede transformar el df completo:
        X_full_scaled = scaler.transform(df_clean.drop(columns=[target])) 
        y_pred = model.predict(X_full_scaled)
        
        # Añadir columnas al dataframe limpio original (para que el usuario lo lea bien)
        df_clean['Target_Real'] = df_clean[target]
        df_clean['Target_Predicho'] = y_pred
        
        with _sessions_lock:
            session = _sessions.get(session_id)
            
        # Guardarlo como CSV en la carpeta temporal de la sesión
        augmented_csv_path = os.path.join(session["_tmp_dir"], "augmented_dataset.csv")
        df_clean.to_csv(augmented_csv_path, index=False)
        # -----------------------------------------------------------

        _update_session(
            session_id,
            status="ready",
            progress=1.0,
            message="¡Pipeline inicializado correctamente!",
            agent=agent_executor,
            model_info=model_info,
        )

    except Exception as exc:
        _update_session(
            session_id,
            status="error",
            message=str(exc),
            error=str(exc),
        )


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
async def upload_csv(file: UploadFile = File(...)) -> JSONResponse:
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Solo se aceptan archivos CSV.")

    raw = await file.read()

    # Write to a temp file to let pandas handle encoding / dialect detection
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
    preview_html = df.head(5).to_html(
        classes="preview-table", border=0, index=False, escape=True
    )

    return JSONResponse(
        {
            "columns": columns,
            "preview_html": preview_html,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
        }
    )


# ------------------------------------------------------------------
# Initialize pipeline
# ------------------------------------------------------------------
@app.post("/api/initialize")
async def initialize(
    file: UploadFile = File(...),
    target: str = Form(...),
) -> JSONResponse:
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(400, "Solo se aceptan archivos CSV.")
    if not target or not target.strip():
        raise HTTPException(400, "El campo 'target' no puede estar vacío.")

    raw = await file.read()

    # Store CSV in a dedicated temp directory so the metadata cache JSON
    # generated by preprocess_dataset lands in the same directory.
    tmp_dir = tempfile.mkdtemp(prefix="xai_session_")
    csv_path = os.path.join(tmp_dir, "dataset.csv")
    with open(csv_path, "wb") as fh:
        fh.write(raw)

    session_id = str(uuid.uuid4())

    with _sessions_lock:
        _sessions[session_id] = {
            "status": "running",
            "progress": 0.0,
            "message": "Iniciando pipeline…",
            "agent": None,
            "model_info": None,
            "error": None,
            "_tmp_dir": tmp_dir,
        }

    thread = Thread(
        target=_run_full_pipeline,
        args=(session_id, csv_path, target.strip()),
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
            "dataset_metadata": session.get("dataset_metadata"), # <--- Añade esto
            "model_info": session.get("model_info"),
        }
    )


# ------------------------------------------------------------------
# Chat
# ------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str
    message: str


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
        result = agent.invoke({"input": body.message.strip()})
        response_text = result.get("output", str(result))

        # Collect the first valid plot produced during this turn
        plot_url: str | None = None
        for _action, observation in result.get("intermediate_steps", []):
            try:
                obs_dict = (
                    json.loads(observation)
                    if isinstance(observation, str)
                    else {}
                )
                if isinstance(obs_dict, dict) and "plot_path" in obs_dict:
                    raw_path = obs_dict["plot_path"]
                    safe_path = os.path.realpath(raw_path)
                    if (
                        safe_path.startswith(_PLOTS_DIR + os.sep)
                        and os.path.isfile(safe_path)
                    ):
                        plot_url = f"/api/plot/{os.path.basename(safe_path)}"
                        break
            except (json.JSONDecodeError, TypeError):
                pass

        return JSONResponse({"response": response_text, "plot_url": plot_url})

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

    csv_path = os.path.join(tmp_dir, "augmented_dataset.csv")
    if not os.path.exists(csv_path):
        raise HTTPException(400, "Dataset exploratorio no está listo aún.")

    try:
        # Cargar el dataset (en producción con datasets gigantes usarías chunking o dask)
        df = pd.read_csv(csv_path)
        
        total_rows = int(df.shape[0])
        start_idx = (page - 1) * size
        end_idx = start_idx + size
        
        # Extraer solo la página solicitada
        df_page = df.iloc[start_idx:end_idx]
        
        return JSONResponse({
            "total_rows": total_rows,
            "page": page,
            "size": size,
            "columns": df.columns.tolist(),
            "data": df_page.to_dict(orient="records")
        })
        
    except Exception as exc:
        raise HTTPException(500, f"Error al leer el dataset: {exc}")


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
