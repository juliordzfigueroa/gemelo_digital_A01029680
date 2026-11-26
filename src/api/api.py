"""FastAPI orchestrator for the conversational layer.

The single entry point the viewer talks to. One request runs the whole chain:

    classify the question  ->  retrieve the data it needs  ->  build the prompt
    ->  ask the local LLM  ->  return the sentence

Every number in the answer comes from this process, never from the model. That
is the whole point of the retrieval-augmented pattern of section 4.3: the LLM
is a writer, not a calculator, and it is given no opportunity to be anything
else.

Where each step lives: llm/classifier.py reads the question, api/retrieval.py
fetches the data, llm/context.py and llm/floor_prompt.py write the prompt,
llm/llm_client.py talks to Ollama, and llm/replies.py holds every sentence the
code writes on its own. This file only routes between them and serves the
viewer.

Usage, from the src/ directory:
    uvicorn api.api:app --port 8000

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Retrieval moved to api/retrieval.py; removed the space_id
    field of the request, which no client sends since the viewer dropped
    its space selector.

Reference:
    Sections 4.3.1 to 4.3.4 of the research document.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from api import estimates
from api import retrieval as ret
from common import config as cfg
from common import data as dat
from common import spaces
from gru import predict as pred
from llm import classifier, context, floor_prompt, llm_client, replies
from llm import formatting as fmt


# --------------------------------------------------------------------------
# Request and response shapes
# --------------------------------------------------------------------------
class Question(BaseModel):
    """Body of a /chat or /context request."""

    question: str = Field(min_length=1, max_length=500)


class Answer(BaseModel):
    """Body of a /chat response. The viewer reads every field."""

    response: str
    intent_type: str
    space: int | None = None
    floor: int | None = None
    building: bool = False
    horizon_min: int | None = None
    forecast_method: str | None = None
    warning: str | None = None
    origin: str | None = None  # "medido" or "estimado"; None for a fixed reply


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
        ret.STATE.update(net=net, normalizer=normalizer, metadata=metadata)
        print(f"GRU cargado: {net.n_parameters():,} parametros")
    except (FileNotFoundError, RuntimeError) as e:
        ret.STATE["model_error"] = str(e)
        print(f"ADVERTENCIA: el GRU no se pudo cargar: {e}")

    # TODO E5: replace with a connection to the Tec database. Everything
    # downstream reads this frame through retrieval.space_rows().
    try:
        data = dat.load_data()
        print(
            f"Datos en memoria: {len(data):,} filas, espacios "
            f"{sorted(int(s) for s in data[cfg.SPACE_COLUMN].unique())}, hasta "
            f"{data[cfg.TIME_COLUMN].max()}"
        )
    except FileNotFoundError as e:
        data = pd.DataFrame(columns=[cfg.TIME_COLUMN, cfg.SPACE_COLUMN, *cfg.ACTIVE_VARIABLES])
        print(f"ADVERTENCIA: sin datos: {e}")
    ret.STATE["data"] = data

    reason = estimates.load(data)
    if reason:
        print(f"Sin estimaciones para salones sin sensor: {reason}")
    else:
        print("Estimaciones del generador listas para los salones sin sensor")
    yield


app = FastAPI(
    title="Gemelo Digital Aulas 3",
    description="Capa conversacional: GRU + Llama 3.1 8B vía Ollama.",
    version="0.1.0",
    lifespan=lifespan_handler,
)

# The viewer page lives with the API so /chat and /spaces are same-origin
# requests; only the clouds come from the scan server, which allows this origin.
if cfg.VIEWER_DIR.is_dir():
    app.mount("/viewer", StaticFiles(directory=cfg.VIEWER_DIR, html=True), name="viewer")


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------
def _target_and_data(intent: classifier.Intent) -> tuple[int | None, int | None, dict]:
    """Decide what the answer is about, and fetch it.

    Order of precedence: a space named in the text, an unknown room number,
    a floor, the whole building. Several outcomes need no LLM at all and come
    back as a "direct" sentence: a room without a sensor, an unknown room, a
    message that is not a question about the building, or a question that
    does not say where. A fixed sentence there is both faster and safer than
    asking the model to explain an absence.

    Args:
        intent: The classified question.

    Returns:
        The space id (None unless the answer is about one space), the floor
        (None unless it is about one floor) and the data dict, which holds
        either the input for context.build_prompt() or a "direct" answer.
    """
    question = intent.original_text
    space = intent.space_id

    if space is not None:
        if not ret.space_rows(space).empty:
            return space, None, ret.retrieve(intent, space)
        label = str(spaces.labels().get(space, space))
        if not estimates.available(space):
            text = replies.no_sensor(label, spaces.aliases(space), ret.rooms_with_data(), question)
            return space, None, {"direct": text}
        # No sensor, but the generator can estimate it: the present state and
        # the last day, never a GRU forecast or an anomaly check.
        if intent.kind in ("current", "history"):
            return space, None, estimates.estimate(intent, space, ret.latest_time())
        template = "estimate_no_forecast" if intent.kind == "forecast" else "estimate_no_anomaly"
        return space, None, {"direct": replies.reply(template, question, subject=replies.capitalized(label))}

    if intent.unknown_room:
        text = replies.unknown_room(
            intent.unknown_room, question, classifier.FLOOR_SPACES, spaces.labels()
        )
        return None, None, {"direct": text}
    if intent.floor is not None:
        return None, intent.floor, ret.floor_data(intent, intent.floor)
    if intent.building:
        return None, None, ret.floor_data(intent, None)
    if intent.chat:
        return None, None, {"direct": replies.reply(intent.chat, question)}
    # A building question that does not say where. Answering about a default
    # space would report on a room the user never asked about.
    return None, None, {"direct": replies.reply("ask_target", question)}


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
    data = ret.STATE["data"]
    return {
        "status": "ok",
        "gru_loaded": ret.STATE["net"] is not None,
        "model_error": ret.STATE["model_error"],
        "ollama_required": True,
        "ollama_ready": ready,
        "ollama_detail": reason,
        "rows_in_memory": 0 if data is None else int(len(data)),
        "data_source": str(cfg.REAL_DATA.name),
        "anomaly_detector": "3sigma" if ret.STATE["normalizer"] is not None else "unavailable",
        "estimates": estimates.STATE["calibration"] is not None,
    }


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send a browser opening the bare address to the viewer."""
    return RedirectResponse("/viewer/")


@app.get("/spaces")
def get_spaces() -> dict:
    """List the floors and spaces the viewer draws, with their latest reading.

    Returns:
        The scan server address, the floor clouds, and every space with the
        name to show, its marker, dimensions and latest reading (empty when
        it has no sensor).
    """
    registry = spaces.read()
    labels = spaces.labels()
    output = []
    for space in registry["spaces"]:
        current = ret.current_data(int(space["space_id"]))
        output.append({
            "space_id": space["space_id"],
            "name": space["name"],
            "label": labels[int(space["space_id"])],
            "room": space.get("room"),
            "floor": space["floor"],
            "kind": space["kind"],
            "marker": space.get("marker"),
            "dimensions": space.get("dimensions"),
            "timestamp": current["timestamp"],
            "readings": current["readings"],
        })
    return {"scans_url": cfg.SCANS_URL, "floors": registry["floors"], "spaces": output}


@app.get("/spaces/registry")
def get_registry() -> dict:
    """Return config/spaces.json as it is on disk.

    The viewer merges the markers and dimensions measured in Potree into this
    copy and offers it for download, so every other field (alias, notes,
    descriptions) survives the round trip untouched.

    Returns:
        The parsed registry file.
    """
    return spaces.read()


@app.post("/context")
def context_for(question: Question) -> dict:
    """Run the chain up to the prompt, without calling the LLM.

    Exists so the retrieval half can be tested and demonstrated when Ollama is
    not installed, and so a wrong answer can be traced to either the data or
    the model instead of guessing between them.

    Args:
        question: The request body.

    Returns:
        The classified intent, the retrieved data and both prompts.
    """
    intent = classifier.classify(question.question)
    space, floor, data = _target_and_data(intent)
    system, user = None, None
    if not data.get("direct"):
        system, user = context.build_prompt(intent, data)

    return {
        "intent": {
            "kind": intent.kind,
            "variables": intent.variables,
            "space_id": space,
            "floor": floor,
            "building": data.get("scope") == "building",
            "chat": intent.chat,
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
        question: The request body.

    Returns:
        The generated sentence plus the routing decisions that produced it, so
        the viewer can show what the answer was based on.
    """
    intent = classifier.classify(question.question)
    space, floor, data = _target_and_data(intent)
    if data.get("direct"):
        return Answer(response=data["direct"], intent_type=intent.kind, space=space)

    system, user = context.build_prompt(intent, data)
    try:
        response = llm_client.generate_answer(system, user)
    except RuntimeError as e:
        # Returned as a normal body rather than a 5xx: the chain did its work
        # and the user is better served by a sentence explaining what is
        # missing than by an error the viewer has to interpret.
        response = f"No puedo responder ahora mismo. {e}"
    else:
        # The code adds what the model must not be trusted with: the
        # room-by-room detail of a floor, or the label of an estimate.
        detail = floor_prompt.detail(data) if data.get("scope") else ""
        if data.get("origin") == "estimado":
            detail = replies.estimate_note(
                data["space"], spaces.aliases(space), data["based_on"],
                fmt.expected_error(data["expected_error"]), question.question,
            )
        if detail:
            response = f"{response.rstrip()}\n\n{detail}"

    is_forecast = intent.kind == "forecast"
    return Answer(
        response=response,
        intent_type=intent.kind,
        space=space,
        floor=floor,
        building=data.get("scope") == "building",
        horizon_min=data.get("horizon_min") if is_forecast else None,
        forecast_method=data.get("method") if is_forecast else None,
        warning=data.get("warning") or data.get("error") or data.get("detail"),
        origin=data.get("origin", "medido"),
    )
