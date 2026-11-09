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
functions read the cleaned sensor file written by ingestion.cleaning, loaded at
startup, because the Tec database is not reachable yet. So "current" means
the latest reading in that file, not the present moment. They are marked `TODO E5` and are the only places that change
when it is. The anomaly detector does not exist at all, and this module says
so rather than reporting that nothing was found, which would be a different
claim entirely.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-05 - Answer questions about a whole floor.

Reference:
    Sections 4.3.1 to 4.3.4 of the research document.
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from common import config as cfg
from common import data as dat
from gru import predict as pred
from llm import classifier, context, llm_client, replies

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
    """Body of a /chat or /context request."""

    question: str = Field(min_length=1, max_length=500)
    space_id: int | None = Field(
        default=None,
        description=(
            "Space selected in the viewer. Used when the question names "
            "neither a space nor a floor; either one in the text wins."
        ),
    )


class Answer(BaseModel):
    """Body of a /chat response."""

    response: str
    intent_type: str
    space: int | None
    floor: int | None = None
    building: bool = False
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
    try:
        STATE["data"] = dat.load_data()
        data = STATE["data"]
        print(
            f"Datos en memoria: {len(data):,} filas, espacios "
            f"{sorted(data[cfg.SPACE_COLUMN].unique())}, hasta "
            f"{data[cfg.TIME_COLUMN].max()}"
        )
    except FileNotFoundError as e:
        STATE["data"] = pd.DataFrame(
            columns=[cfg.TIME_COLUMN, cfg.SPACE_COLUMN, *cfg.ACTIVE_VARIABLES]
        )
        print(f"ADVERTENCIA: sin datos: {e}")
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

    TODO E5: replace with a call to the anomaly detector once it exists.

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
            "El detector de anomalías (regla 3σ) todavía no está implementado, "
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
    window = pred.last_contiguous_window(rows, metadata["input_steps"])

    if window is None:
        return {
            "space": space_id,
            "current_reading": {},
            "predictions": [],
            "error": (
                f"No hay {metadata['input_steps']} lecturas consecutivas "
                "recientes, así que no se puede predecir."
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
# Floor questions
# --------------------------------------------------------------------------
def _space_labels() -> dict[int, str]:
    """Return the name a person uses for each space.

    The room number when there is one ("3308"). Spaces without one, like the
    open offices of floor 4, get a readable name built from the registry name
    ("space_4_2" becomes "oficina 4-2"). The prompt gets these instead of ids
    so the answer never says "espacio 5" or "space_4_2".

    Returns:
        A dict from space id to its label.
    """
    registry = json.loads(cfg.SPACES_FILE.read_text(encoding="utf-8"))
    labels = {}
    for s in registry["spaces"]:
        if s.get("room"):
            label = s["room"]
        else:
            number = s["name"].removeprefix("space_").replace("_", "-")
            label = f"oficina {number}" if s.get("kind") == "open_office" else f"espacio {number}"
        labels[int(s["space_id"])] = label
    return labels


def _floor_summary(rooms: list[dict], key: str, variables: list[str]) -> dict:
    """Mean, minimum and maximum of each variable across the rooms of a floor.

    Computed here and not by the LLM, like every other number in the answer.

    Args:
        rooms: One entry per room, each holding a dict of values under `key`.
        key: Which dict of each room to summarize.
        variables: Variables to summarize.

    Returns:
        Per variable: mean, min and max, each extreme with the room it is in,
        and how many rooms had a value.
    """
    summary = {}
    for variable in variables:
        pairs = [
            (room[key][variable], room["room"])
            for room in rooms
            if room.get(key, {}).get(variable) is not None
        ]
        if not pairs:
            continue
        values = [v for v, _ in pairs]
        low, high = min(pairs), max(pairs)
        summary[variable] = {
            "mean": round(sum(values) / len(values), 2),
            "min": low[0], "min_room": low[1],
            "max": high[0], "max_room": high[1],
            "n_rooms": len(pairs),
        }
    return summary


def _floor_data(intent: classifier.Intent, floor: int | None) -> dict:
    """Retrieve what a question about a whole floor, or the building, needs.

    Runs the same per-space retrieval as a single-space question for every
    space that has readings, and adds a summary. Spaces without a sensor are
    named so the answer can say the floor is only partly measured.

    TODO E5: inherits the stand-in data source of the per-space functions.

    Args:
        intent: The classified question; carries the kind, the variables
            and, for forecasts, the horizon.
        floor: Floor number, or None for every space in the building.

    Returns:
        A dict shaped for context._prompt_floor, with "scope" set to
        "floor" or "building".
    """
    labels = _space_labels()
    if floor is None:
        space_ids = sorted(s for spaces in classifier.FLOOR_SPACES.values() for s in spaces)
    else:
        space_ids = sorted(classifier.FLOOR_SPACES.get(floor, []))
    variables = intent.variables or list(cfg.ACTIVE_VARIABLES)
    data: dict[str, Any] = {
        "scope": "building" if floor is None else "floor",
        "floor": floor,
        "kind": intent.kind,
        "variables": variables,
        "n_spaces": len(space_ids),
        "rooms": [],
        "without_data": [],
        "errors": [],
    }

    if intent.kind == "anomaly":
        data["state"] = "unavailable"
        data["detail"] = _anomaly_data(space_ids[0] if space_ids else 0)["detail"]
        return data

    if intent.kind == "history":
        # Same limit as the single-space path: one variable per question.
        variables = [intent.variables[0] if intent.variables else "co2"]
        data["variables"] = variables

    for space_id in space_ids:
        room = labels.get(space_id, str(space_id))
        if _space_rows(space_id).empty:
            data["without_data"].append(room)
            continue

        if intent.kind == "forecast":
            item = _forecast_data(intent, space_id)
            if item.get("error"):
                reason = item["error"].split(", así que")[0].rstrip(".")
                data["errors"].append({"room": room, "reason": reason[:1].lower() + reason[1:]})
                continue
            last = item["predictions"][-1] if item["predictions"] else {}
            data["rooms"].append({
                "room": room,
                "current": {v: item["current_reading"].get(v) for v in variables},
                "predicted": {v: _number(last.get(v)) for v in variables},
            })
            data["timestamp"] = item.get("timestamp")
            data["horizon_min"] = item.get("horizon_min")
            data["method"] = item.get("method")
            data["warning"] = item.get("warning")
        elif intent.kind == "history":
            item = _history_data(space_id, variables[0])
            data["rooms"].append({"room": room, "summary": item["summary"]})
            data["since"], data["until"] = item.get("since"), item.get("until")
        else:
            item = _current_data(space_id)
            data["rooms"].append({
                "room": room,
                "timestamp": item["timestamp"],
                "readings": {v: item["readings"].get(v) for v in variables},
            })

    if intent.kind == "forecast":
        data["summary"] = _floor_summary(data["rooms"], "predicted", variables)
    elif intent.kind == "history":
        summaries = data["rooms"]
        variable = variables[0]
        maxima = [(r["summary"]["max"], r["room"]) for r in summaries if r["summary"].get("max") is not None]
        minima = [(r["summary"]["min"], r["room"]) for r in summaries if r["summary"].get("min") is not None]
        means = [r["summary"]["promedio"] for r in summaries if r["summary"].get("promedio") is not None]
        data["summary"] = {variable: {
            "max": max(maxima)[0] if maxima else None,
            "max_room": max(maxima)[1] if maxima else None,
            "min": min(minima)[0] if minima else None,
            "min_room": min(minima)[1] if minima else None,
            "mean": round(sum(means) / len(means), 2) if means else None,
            "n_rooms": len(summaries),
        }}
    else:
        data["summary"] = _floor_summary(data["rooms"], "readings", variables)
        stamps = [r["timestamp"] for r in data["rooms"] if r["timestamp"]]
        data["timestamp"] = max(stamps) if stamps else None
    return data


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
        data = _forecast_data(intent, space_id)
    elif intent.kind == "history":
        variable = intent.variables[0] if intent.variables else "co2"
        data = _history_data(space_id, variable)
    elif intent.kind == "anomaly":
        data = _anomaly_data(space_id)
    else:
        data = _current_data(space_id)
    # The prompt names the space by its room number: given the id, Llama
    # answered "espacio 308" and, with the old ids, "espacio 5".
    data["space"] = _space_labels().get(space_id, space_id)
    return data


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
    return int(STATE["data"][cfg.SPACE_COLUMN].min())


def _target_and_data(
    intent: classifier.Intent, requested: int | None
) -> tuple[int | None, int | None, dict]:
    """Decide what the answer is about, and fetch it.

    Order of precedence: a space named in the text, an unknown room number,
    a floor, the whole building, and last the space sent in the request
    body. Several outcomes need no LLM at all and come back as a "direct"
    sentence: an off-topic question, no target, an unknown room, or a room
    without a sensor. A fixed sentence there is both faster and safer than
    asking the model to explain an absence.

    Args:
        intent: The classified question.
        requested: Space id sent in the request body, if any.

    Returns:
        The space id (None unless the answer is about one space), the floor
        (None unless it is about one floor) and the data dict, which holds
        either the input for context.build_prompt() or a "direct" answer.
    """
    question = intent.original_text
    if intent.space_id is None:
        if intent.unknown_room:
            return None, None, {"direct": _unknown_room_message(intent.unknown_room, question)}
        if intent.floor is not None:
            return None, intent.floor, _floor_data(intent, intent.floor)
        if intent.building:
            return None, None, _floor_data(intent, None)
        if intent.chat:
            return None, None, {"direct": replies.reply(intent.chat, question)}
        if requested is None:
            # A building question that does not say where. Answering about a
            # default space would report on a room the user never asked about.
            return None, None, {"direct": replies.reply("ask_target", question)}

    space = _chosen_space(intent, requested)
    if _space_rows(space).empty:
        return space, None, {"direct": _no_sensor_message(space, question)}
    return space, None, _retrieve(intent, space)


def _with_article(label: str) -> str:
    """Put the right article before a space label: "el 3309", "la oficina 4-1"."""
    return f"la {label}" if label.startswith("oficina") else f"el {label}"


def _rooms_with_data() -> list[str]:
    """Return the labels of the spaces that have readings, sorted."""
    labels = _space_labels()
    ids = sorted(int(s) for s in STATE["data"][cfg.SPACE_COLUMN].unique())
    return [str(labels.get(s, s)) for s in ids]


def _unknown_room_message(number: str, question: str = "") -> str:
    """Say that a room number is not in the registry, and what is.

    Args:
        number: The four-digit number from the question.
        question: The question as typed, to pick the wording.

    Returns:
        A sentence for the user.
    """
    if not number.startswith("3"):
        return replies.reply("unknown_room_other_building", question, number=number)
    floor = int(number[1])
    if floor not in classifier.FLOOR_SPACES:
        return replies.reply(
            "unknown_room_no_floor", question, number=number,
            top_floor=max(classifier.FLOOR_SPACES),
        )
    labels = _space_labels()
    known = replies.spoken_list(
        [str(labels[s]) for s in sorted(classifier.FLOOR_SPACES[floor])]
    )
    where = "la planta baja" if floor == 0 else f"el piso {floor}"
    return replies.reply(
        "unknown_room_floor", question, number=number, where=where, known=known
    )


def _no_sensor_message(space_id: int, question: str = "") -> str:
    """Say that a registered space has no readings, and which spaces do.

    Args:
        space_id: A space with no rows in the data.
        question: The question as typed. When it named the space by another
            number (the 3302 is the 3301), the answer says so.

    Returns:
        A sentence for the user.
    """
    label = str(_space_labels().get(space_id, space_id))
    registry = json.loads(cfg.SPACES_FILE.read_text(encoding="utf-8"))
    aliases = next(
        (s.get("alias", []) for s in registry["spaces"] if int(s["space_id"]) == space_id), []
    )
    asked = next(
        (a for a in aliases if a != label and a.isdigit() and a in question.lower()), None
    )
    rooms = replies.spoken_list(
        [_with_article(r) if r.startswith("oficina") else r for r in _rooms_with_data()]
    )
    if asked:
        return replies.reply("no_sensor_alias", question, asked=asked, label=label, rooms=rooms)
    named = _with_article(label)
    return replies.reply(
        "no_sensor", question, subject=named[:1].upper() + named[1:], rooms=rooms
    )


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
        "data_source": str(cfg.REAL_DATA.name),
        "anomaly_detector": "not_implemented",
    }


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Send a browser opening the bare address to the viewer."""
    return RedirectResponse("/viewer/")


@app.get("/spaces")
def spaces() -> dict:
    """List the floors and spaces the viewer draws, with their latest reading.

    The registry is read on every call rather than at startup, so a marker
    measured in Potree and written to spaces.json shows up on reload without
    restarting the API.

    Returns:
        The scan server address, the floor clouds, and every space with its
        marker, dimensions and latest reading (empty when it has no sensor).
    """
    registry = json.loads(cfg.SPACES_FILE.read_text(encoding="utf-8"))
    output = []
    for space in registry["spaces"]:
        current = _current_data(int(space["space_id"]))
        output.append({
            "space_id": space["space_id"],
            "name": space["name"],
            "room": space.get("room"),
            "floor": space["floor"],
            "kind": space["kind"],
            "marker": space.get("marker"),
            "dimensions": space.get("dimensions"),
            "timestamp": current["timestamp"],
            "readings": current["readings"],
        })
    return {
        "scans_url": cfg.SCANS_URL,
        "floors": registry["floors"],
        "spaces": output,
    }


@app.get("/spaces/registry")
def spaces_registry() -> dict:
    """Return config/spaces.json as it is on disk.

    The viewer merges the markers and dimensions measured in Potree into this
    copy and offers it for download, so every other field (alias, notes,
    descriptions) survives the round trip untouched.

    Returns:
        The parsed registry file.
    """
    return json.loads(cfg.SPACES_FILE.read_text(encoding="utf-8"))


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
    space, floor, data = _target_and_data(intent, question.space_id)
    if data.get("direct"):
        system, user = None, None
    else:
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
        pregunta: The request body.

    Returns:
        The generated sentence plus the routing decisions that produced it, so
        the caller can show what the answer was based on.
    """
    intent = classifier.classify(question.question)
    space, floor, data = _target_and_data(intent, question.space_id)
    if data.get("direct"):
        return Answer(
            response=data["direct"], intent_type=intent.kind, space=space,
            floor=None, horizon_min=None, forecast_method=None, warning=None,
        )
    system, user = context.build_prompt(intent, data)

    try:
        response = llm_client.generate_answer(system, user)
    except RuntimeError as e:
        # Returned as a normal body rather than a 5xx: the chain did its work
        # and the caller is better served by a sentence explaining what is
        # missing than by an error the viewer has to interpret.
        response = f"No puedo responder ahora mismo. {e}"
    else:
        detail = context.floor_detail(data) if data.get("scope") else ""
        if detail:
            response = f"{response.rstrip()}\n\n{detail}"

    return Answer(
        response=response,
        intent_type=intent.kind,
        space=space,
        floor=floor,
        building=data.get("scope") == "building",
        horizon_min=(
            data.get("horizon_min") if intent.kind == "forecast" else None
        ),
        forecast_method=(
            data.get("method") if intent.kind == "forecast" else None
        ),
        warning=data.get("warning") or data.get("error") or data.get("detail"),
    )
