"""FastAPI orchestrator for the conversational layer.

The single entry point the viewer talks to. One request runs the whole chain:

    classify the question  ->  retrieve the data it needs  ->  build the prompt
    ->  ask the local LLM  ->  return the sentence

Every number in the answer comes from this process, never from the model. That
is the whole point of the retrieval-augmented pattern of section 4.3: the LLM
is a writer, not a calculator, and it is given no opportunity to be anything
else.

What is wired and what is not. The GRU is real: the forecast comes from the
trained checkpoint in models/. The data source is not: the three retrieval
functions read an in-memory synthetic frame generated at startup, because the
Aulas 3 sensors are still being installed and the Tec database is not
reachable yet. They are marked `TODO E5` and are the only places that change
when it is. The anomaly detector does not exist at all, and this module says
so rather than reporting that nothing was found, which would be a different
claim entirely.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-12 - Initial implementation.

Reference:
    Sections 4.3.1 to 4.3.4 of the research document.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from common import config as cfg
from common import synthetic
from gru import predict as pred
from llm import classifier, llm_client, context

# Days of synthetic history kept in memory as the stand-in for the database.
# Enough to answer a "yesterday" question and to leave contiguous windows.
SYNTHETIC_DAYS = 7
SYNTHETIC_SPACES = 5

# Default span for a historical question when the user names no period.
HISTORY_HOURS = 24

# Process-wide state, populated once at startup. Reloading the checkpoint per
# request would add seconds of latency to every question for no benefit.
STATE: dict[str, Any] = {
    "net": None,
    "normalizer": None,
    "metadata": None,
    "data": None,
    "model_error": None,
}


# --------------------------------------------------------------------------
# Request and response shapes
# --------------------------------------------------------------------------
class Question(BaseModel):
    """Body of a /chat or /contexto request."""

    question: str = Field(min_length=1, max_length=500)
    space_id: int | None = Field(
        default=None,
        description=(
            "Space selected in the viewer. Used when the question names none. "
            "A space named in the text wins over this value."
        ),
    )


class Answer(BaseModel):
    """Body of a /chat response."""

    response: str
    intent_type: str
    space: int | None
    horizon_min: int | None
    forecast_method: str | None
    warning: str | None


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------
@asynccontextmanager
async def lifespan_handler(app: FastAPI):
    """Load the model and the stand-in data once, before serving.

    A missing checkpoint is not fatal: the service still starts so that
    /health can report what is wrong instead of refusing to boot, which is
    harder to diagnose.

    Args:
        app: The application being started.

    Yields:
        Control back to FastAPI while the app serves requests.
    """
    try:
        net, normalizer, metadata = pred.load_model()
        STATE.update(net=net, normalizer=normalizer, metadata=metadata)
        print(
            f"GRU cargado: implementacion {net.implementation}, "
            f"{net.n_parameters():,} parametros"
        )
    except (FileNotFoundError, RuntimeError) as e:
        STATE["model_error"] = str(e)
        print(f"ADVERTENCIA: el GRU no se pudo cargar: {e}")

    # TODO E5: replace with a connection to the Tec database. Everything
    # downstream reads this frame through the three retrieval functions, so
    # only they change.
    STATE["data"] = synthetic.generate(
        days=SYNTHETIC_DAYS, n_spaces=SYNTHETIC_SPACES
    )
    print(
        f"Datos sinteticos en memoria: {len(STATE['data']):,} filas, "
        f"{SYNTHETIC_SPACES} espacios, {SYNTHETIC_DAYS} dias"
    )
    yield


app = FastAPI(
    title="Gemelo Digital Aulas 3",
    description="Capa conversacional: GRU + Llama 3.1 8B vía Ollama.",
    version="0.1.0",
    lifespan=lifespan_handler,
)


# --------------------------------------------------------------------------
# Data retrieval
# --------------------------------------------------------------------------
def _space_rows(space_id: int) -> pd.DataFrame:
    """Return every row of one space, sorted by time.

    TODO E5: replace with a query to the Tec database.

    Args:
        space_id: Space identifier.

    Returns:
        The rows for that space, possibly empty.
    """
    data = STATE["data"]
    own_rows = data[data[cfg.SPACE_COLUMN] == space_id]
    return own_rows.sort_values(cfg.TIME_COLUMN).reset_index(drop=True)


def _last_contiguous_window(rows: pd.DataFrame, steps: int) -> pd.DataFrame | None:
    """Find the most recent run of `pasos` consecutive readings.

    Taking the last N rows is not enough. Readings arrive as missing rows when
    a node loses the network, so the last 12 rows can span a two-day hole and
    still look like a valid window. The network would consume it without
    complaint and forecast from a sequence that never happened.

    Args:
        filas: Rows of one space, sorted by time.
        pasos: How many consecutive readings are needed.

    Returns:
        The last window whose steps are exactly one sampling interval apart,
        or None if the history holds no such run.
    """
    if len(rows) < steps:
        return None

    step = pd.Timedelta(seconds=cfg.INTERVAL_SECONDS)
    stamps = pd.to_datetime(rows[cfg.TIME_COLUMN])
    contiguous = stamps.diff() == step

    # Walk backwards: the first window that is fully contiguous and complete
    # is the freshest usable one.
    for end in range(len(rows), steps - 1, -1):
        start = end - steps
        window = rows.iloc[start:end]
        if not contiguous.iloc[start + 1 : end].all():
            continue
        if window[cfg.ACTIVE_VARIABLES].isna().to_numpy().any():
            continue
        return window
    return None


def _current_data(space_id: int) -> dict:
    """Retrieve the latest reading of one space.

    TODO E5: replace with a query to `lecturas_agregadas` for the most recent
    row of that space.

    Args:
        space_id: Space identifier.

    Returns:
        A dict shaped for context._prompt_current.
    """
    rows = _space_rows(space_id)
    if rows.empty:
        return {"space": space_id, "readings": {}, "timestamp": None}

    last = rows.iloc[-1]
    return {
        "space": space_id,
        "timestamp": str(last[cfg.TIME_COLUMN]),
        "readings": {
            variable: _number(last[variable])
            for variable in cfg.ACTIVE_VARIABLES
        },
    }


def _history_data(
    space_id: int, variable: str, hours: int = HISTORY_HOURS
) -> dict:
    """Summarize one variable of one space over a past window.

    TODO E5: replace with a ranged query to `lecturas_agregadas`.

    Args:
        space_id: Space identifier.
        variable: Which variable to summarize.
        horas: How far back to look.

    Returns:
        A dict shaped for context._prompt_history.
    """
    rows = _space_rows(space_id)
    if rows.empty:
        return {"space": space_id, "variable": variable, "summary": {}}

    stamps = pd.to_datetime(rows[cfg.TIME_COLUMN])
    cutoff = stamps.max() - pd.Timedelta(hours=hours)
    recent = rows[stamps >= cutoff]
    series = recent[variable].dropna()

    return {
        "space": space_id,
        "variable": variable,
        "since": str(cutoff),
        "until": str(stamps.max()),
        "summary": {
            "max": _number(series.max()) if len(series) else None,
            "min": _number(series.min()) if len(series) else None,
            "promedio": _number(series.mean()) if len(series) else None,
            "n_readings": int(len(series)),
        },
    }


def _anomaly_data(space_id: int) -> dict:
    """Report what the anomaly detector found.

    TODO E5: replace with a call to src/anomalias/detector.py once it exists.

    Until then this returns "no_disponible" rather than an empty list of
    anomalies. The distinction matters: an empty list means the detector ran
    and found nothing, and claiming that while no detector exists would put a
    false all-clear in front of the user.

    Args:
        space_id: Space identifier.

    Returns:
        A dict shaped for context._prompt_anomaly.
    """
    return {
        "space": space_id,
        "state": "unavailable",
        "anomalies": [],
        "detail": (
            "El detector 3σ (src/anomalias/) todavía no está implementado, "
            "así que no se ha evaluado si hay anomalías."
        ),
    }


def _forecast_data(intent: classifier.Intent, space_id: int) -> dict:
    """Run the GRU forward for one space.

    Args:
        intencion: The classified question, which carries the horizon.
        space_id: Space identifier.

    Returns:
        A dict shaped for context._prompt_forecast. When the model or a
        usable window is unavailable it carries an "error" key instead of
        predictions, so the prompt can say so plainly.
    """
    if STATE["net"] is None:
        return {
            "space": space_id,
            "current_reading": {},
            "predictions": [],
            "error": (
                "El modelo GRU no está cargado: "
                f"{STATE['model_error']}"
            ),
        }

    metadata = STATE["metadata"]
    rows = _space_rows(space_id)
    window = _last_contiguous_window(rows, metadata["input_steps"])

    if window is None:
        return {
            "space": space_id,
            "current_reading": {},
            "predictions": [],
            "error": (
                f"No hay {metadata['input_steps']} lecturas consecutivas "
                f"para el espacio {space_id}, así que no se puede predecir."
            ),
        }

    try:
        output = pred.predict(
            STATE["net"],
            STATE["normalizer"],
            metadata,
            window,
            space_id=space_id,
            horizon_min=intent.horizon_min,
        )
    except ValueError as e:
        return {
            "space": space_id,
            "current_reading": {},
            "predictions": [],
            "error": str(e),
        }

    last = window.iloc[-1]
    return {
        "space": space_id,
        "current_reading": {
            variable: _number(last[variable])
            for variable in metadata["active_variables"]
        },
        "timestamp": str(last[cfg.TIME_COLUMN]),
        **output,
    }


def _number(value: object) -> float | None:
    """Coerce a pandas scalar to a plain float for JSON.

    Args:
        valor: Value taken from a DataFrame.

    Returns:
        The value as a rounded float, or None when it is missing.
    """
    if value is None or pd.isna(value):
        return None
    return round(float(value), 2)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _retrieve(intent: classifier.Intent, space_id: int) -> dict:
    """Fetch whatever the classified intent needs.

    Args:
        intencion: Result of classifier.clasificar().
        space_id: Space to answer about.

    Returns:
        The data dict for context.build_prompt().
    """
    if intent.kind == "forecast":
        return _forecast_data(intent, space_id)
    if intent.kind == "history":
        variable = intent.variables[0] if intent.variables else "co2"
        return _history_data(space_id, variable)
    if intent.kind == "anomaly":
        return _anomaly_data(space_id)
    return _current_data(space_id)


def _chosen_space(intent: classifier.Intent, requested: int | None) -> int:
    """Decide which space the answer is about.

    A space named in the question wins over the one selected in the viewer,
    because the text is the more explicit signal. With neither, the first
    space the model knows is used so the request still produces an answer.

    Args:
        intencion: The classified question.
        pedido: Space id sent in the request body, if any.

    Returns:
        The space id to answer about.
    """
    if intent.space_id is not None:
        return intent.space_id
    if requested is not None:
        return requested
    metadata = STATE["metadata"]
    if metadata:
        return min(int(k) for k in metadata["space_index"])
    return 1


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Report whether the service can answer questions.

    Returns:
        A dict naming the state of the model, the data and Ollama. Never
        raises: a health check that fails to respond is useless.
    """
    ready, reason = llm_client.available()
    data = STATE["data"]
    return {
        "status": "ok",
        "gru_loaded": STATE["net"] is not None,
        "model_error": STATE["model_error"],
        "ollama_required": True,
        "ollama_ready": ready,
        "ollama_detail": reason,
        "rows_in_memory": 0 if data is None else int(len(data)),
        "data_source": "synthetic",
        "anomaly_detector": "not_implemented",
    }


@app.post("/context")
def context_for(question: Question) -> dict:
    """Run the chain up to the prompt, without calling the LLM.

    Exists so the retrieval half can be tested and demonstrated when Ollama is
    not installed, and so a wrong answer can be traced to either the data or
    the model instead of guessing between them.

    Args:
        pregunta: The request body.

    Returns:
        The classified intent, the retrieved data and both prompts.
    """
    intent = classifier.classify(question.question)
    space = _chosen_space(intent, question.space_id)
    data = _retrieve(intent, space)
    system, user = context.build_prompt(intent, data)

    return {
        "intent": {
            "kind": intent.kind,
            "variables": intent.variables,
            "space_id": space,
            "horizon_min": intent.horizon_min,
        },
        "data": data,
        "system_prompt": system,
        "user_prompt": user,
    }


@app.post("/chat", response_model=Answer)
def chat(question: Question) -> Answer:
    """Answer one question about the building.

    Args:
        pregunta: The request body.

    Returns:
        The generated sentence plus the routing decisions that produced it, so
        the caller can show what the answer was based on.
    """
    intent = classifier.classify(question.question)
    space = _chosen_space(intent, question.space_id)
    data = _retrieve(intent, space)
    system, user = context.build_prompt(intent, data)

    try:
        response = llm_client.generate_answer(system, user)
    except RuntimeError as e:
        # Returned as a normal body rather than a 5xx: the chain did its work
        # and the caller is better served by a sentence explaining what is
        # missing than by an error the viewer has to interpret.
        response = f"No puedo responder ahora mismo. {e}"

    return Answer(
        response=response,
        intent_type=intent.kind,
        space=space,
        horizon_min=(
            data.get("horizon_min") if intent.kind == "forecast" else None
        ),
        forecast_method=(
            data.get("method") if intent.kind == "forecast" else None
        ),
        warning=data.get("warning") or data.get("error") or data.get("detail"),
    )
