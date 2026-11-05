"""Rule-based intent classifier for the user's question.

Routes a question to one of four intents so the orchestrator knows which data
to retrieve before calling the LLM, and reads what the question is about: one
space ("3308") or a whole floor ("piso 3", "tercer piso", "planta baja"). Deliberately deterministic: the same
question always yields the same intent, and no model call is involved.

Using the LLM to classify would add a second failure mode to every request.
A wrong routing decision sends the wrong data into the prompt, and a prompt
built from the wrong data produces a confident, wrong answer. Keyword rules
are auditable and free; that is the better trade here.

Keywords stay in Spanish because they match what the user types. Identifiers
and documentation follow the project standard.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-05 - Read the floor a question names.

Reference:
    Section 4.3.4 of the research document.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from common import config as cfg

# --------------------------------------------------------------------------
# Classification vocabulary
# --------------------------------------------------------------------------
ANOMALY_WORDS = [
    "anomalía", "anomalia", "anomalías", "anomalias",
    "raro", "rara", "extraño", "extraña",
    "fuera de lo normal", "anormal", "inusual",
    "problema", "alerta", "alarma", "peligro", "peligroso",
    "excede", "supera", "demasiado", "muy alto", "muy alta",
    "algo mal", "algo está mal", "algo esta mal",
]

FORECAST_WORDS = [
    "estará", "estara", "estarán", "estaran",
    "va a estar", "van a estar",
    "predice", "predecir", "predicción", "prediccion", "predecirá",
    "pronostica", "pronóstico", "pronostico",
    "próximos", "proximos", "próximas", "proximas",
    "más tarde", "mas tarde", "al rato",
    "futuro", "tendencia", "subirá", "subira", "bajará", "bajara",
]

HISTORY_WORDS = [
    "ayer", "anteayer", "la semana pasada", "el mes pasado",
    "histórico", "historico", "histórica", "historica", "historial",
    "cuándo fue", "cuando fue", "cuándo estuvo", "cuando estuvo",
    "pico", "máximo", "maximo", "mínimo", "minimo",
    "promedio", "últimos días", "ultimos dias",
    "últimas horas", "ultimas horas", "últimas 24", "ultimas 24",
    "se registró", "se registro", "hubo",
]

CURRENT_WORDS = [
    "ahora", "actualmente", "en este momento", "hoy",
    "cuál es", "cual es", "cuáles son", "cuales son",
    "actual", "cómo está", "como esta", "cómo se encuentra",
    "lectura", "dime el", "dime la",
]

# Maps each active variable to the words a user might use for it. Keys come
# from config so that narrowing ACTIVE_VARIABLES cannot leave this dict
# claiming to detect something the model no longer predicts.
VARIABLE_SYNONYMS = {
    "co2": ["co2", "co₂", "dióxido de carbono", "dioxido de carbono", "aire"],
    "temperature": ["temperatura", "calor", "frío", "frio", "fría", "fria", "grados"],
    "humidity": ["humedad", "húmedo", "humedo", "húmeda", "humeda"],
    "light": ["luz", "iluminación", "iluminacion", "iluminancia", "lux", "luminosidad"],
    "motion": ["movimiento", "ocupación", "ocupacion", "personas", "gente", "ocupado"],
}

def load_spaces(path: Path = cfg.SPACES_FILE) -> dict[int, list[str]]:
    """Read the words that name each space from the shared registry.

    Args:
        ruta: The space registry shared with the cleaning, the API and the
            viewer, so that all of them agree on what "3308" refers to.

    Returns:
        A dict from space_id to the lowercase names a user might type.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    output = {}
    for space in data["spaces"]:
        names = {space["name"], *space.get("alias", [])}
        if space.get("room"):
            names.add(space["room"])
        output[int(space["space_id"])] = sorted(
            {str(n).lower() for n in names if n}, key=len, reverse=True
        )
    return output


def load_floor_spaces(path: Path = cfg.SPACES_FILE) -> dict[int, list[int]]:
    """Group the registered spaces by floor.

    Args:
        path: The shared space registry.

    Returns:
        A dict from floor number to the space ids on that floor.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    output: dict[int, list[int]] = {}
    for space in data["spaces"]:
        output.setdefault(int(space["floor"]), []).append(int(space["space_id"]))
    return output


SPACES = load_spaces()
FLOOR_SPACES = load_floor_spaces()

# Floor 0 is the ground floor ("planta baja"); "primer piso" is floor 1, as
# the room numbers say (3305 is on floor 3).
FLOOR_NUMBER_WORDS = {"cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4}
FLOOR_ORDINALS = {
    "primer": 1, "primero": 1, "segundo": 2,
    "tercer": 3, "tercero": 3, "cuarto": 4,
}
FLOOR_NOUN = r"(?:piso|nivel|planta)"

# Horizon ceiling, kept in sync with the iterative predictor.
MAX_HORIZON_MIN = 120
DEFAULT_HORIZON = 30


@dataclass
class Intent:
    """What the orchestrator needs to know before fetching data.

    Attributes:
        tipo: One of "predictiva", "actual", "historica" or "anomalia".
        variables: Variables the question mentions. Empty means all of them.
        space_id: Space the question refers to, or None if it names none.
        floor: Floor the question refers to, or None. Only set when the
            question names no space: "el 3308 del piso 3" is about 3308.
        horizon_min: Minutes ahead, meaningful only when tipo is
            "predictiva".
        original_text: The question as the user typed it, so the prompt can
            quote it back verbatim.
    """

    kind: str
    variables: list[str] = field(default_factory=list)
    space_id: int | None = None
    floor: int | None = None
    horizon_min: int = DEFAULT_HORIZON
    original_text: str = ""


def classify(question: str) -> Intent:
    """Classify a question into one intent.

    Args:
        pregunta: The question as typed by the user.

    Returns:
        A populated Intent.

    Reference:
        Section 4.3.4 of the research document.
    """
    text = question.lower().strip()
    kind = _detect_kind(text)
    space_id = _extract_space(text)

    return Intent(
        kind=kind,
        variables=_extract_variables(text),
        space_id=space_id,
        floor=_extract_floor(text) if space_id is None else None,
        horizon_min=(
            _extract_horizon(text)
            if kind == "forecast"
            else DEFAULT_HORIZON
        ),
        original_text=question,
    )


def _detect_kind(text: str) -> str:
    """Pick the intent by keyword match, resolving conflicts by priority.

    Priority is anomaly, then predictive, then historical, then current. It
    encodes urgency rather than precision: a question that says "raro" or
    "peligroso" goes to the detector even when it also asks about the future,
    because answering the safety question matters more than the horizon.

    The cost is visible: "¿el CO2 va a estar muy alto en una hora?" is routed
    as an anomaly and loses its horizon. Fixing it needs a real classifier,
    not a longer keyword list.

    Args:
        texto: The lowercased question.

    Returns:
        The intent name.
    """
    if _contains_any(text, ANOMALY_WORDS):
        return "anomaly"
    if _contains_any(text, FORECAST_WORDS) or _mentions_future(text):
        return "forecast"
    if _contains_any(text, HISTORY_WORDS):
        return "history"
    if _contains_any(text, CURRENT_WORDS):
        return "current"
    # Nothing matched. The present state is the cheapest answer to produce and
    # the least likely to mislead, so it is the fallback.
    return "current"


def _mentions_future(text: str) -> bool:
    """Detect an explicit future time reference such as "en 45 minutos".

    Separate from the keyword list because it is a pattern, not a word, and
    because the same regexes are reused to read the horizon.

    Args:
        texto: The lowercased question.

    Returns:
        True if the question names a time offset into the future.
    """
    return bool(
        re.search(r"\ben\s+\d+\s*(min|hora)", text)
        or re.search(r"\bdentro\s+de\s+\d+\s*(min|hora)", text)
        or any(
            phrase in text
            for phrase in (
                "en media hora", "en una hora", "en dos horas",
                "dentro de una hora", "dentro de media hora",
            )
        )
    )


def _extract_variables(text: str) -> list[str]:
    """Collect the environmental variables the question mentions.

    Args:
        texto: The lowercased question.

    Returns:
        Variable names in the order config declares them. Empty means the
        question did not single any out, which the orchestrator reads as all.
    """
    return [
        variable
        for variable in cfg.ACTIVE_VARIABLES
        if _contains_any(text, VARIABLE_SYNONYMS.get(variable, [variable]))
    ]


def _extract_space(text: str) -> int | None:
    """Find which space the question refers to.

    Args:
        texto: The lowercased question.

    Returns:
        The space id, or None when the question names no space, in which case
        the caller supplies the one selected in the viewer.
    """
    # Longest names first, so "3110" is never read as "311" plus a stray zero.
    candidates = sorted(
        ((name, space_id) for space_id, names in SPACES.items() for name in names),
        key=lambda pair: len(pair[0]),
        reverse=True,
    )
    for name, space_id in candidates:
        if name in text:
            return space_id
    return None


def _extract_floor(text: str) -> int | None:
    """Find which floor the question refers to.

    Needs the word "piso", "nivel" or "planta" next to the number, so "en dos
    horas" or a room number is never read as a floor. "Cuarto" alone means a
    room in Spanish, so it only counts as an ordinal next to one of those
    words.

    Args:
        text: The lowercased question.

    Returns:
        The floor number, or None when the question names no floor of the
        building.
    """
    floor = None
    if "planta baja" in text:
        floor = 0
    elif found := re.search(rf"\b{FLOOR_NOUN}\s+(\d+)\b", text):
        floor = int(found.group(1))
    elif found := re.search(
        rf"\b{FLOOR_NOUN}\s+({'|'.join(FLOOR_NUMBER_WORDS)})\b", text
    ):
        floor = FLOOR_NUMBER_WORDS[found.group(1)]
    elif found := re.search(
        rf"\b({'|'.join(FLOOR_ORDINALS)})\s+{FLOOR_NOUN}\b", text
    ) or re.search(rf"\b{FLOOR_NOUN}\s+({'|'.join(FLOOR_ORDINALS)})\b", text):
        floor = FLOOR_ORDINALS[found.group(1)]
    return floor if floor in FLOOR_SPACES else None


def _extract_horizon(text: str) -> int:
    """Read the prediction horizon in minutes.

    Values are capped at MAX_HORIZON_MIN and rounded up to the sampling
    interval, because the predictor only emits whole 5-minute steps.

    Args:
        texto: The lowercased question.

    Returns:
        Minutes ahead, between the sampling interval and MAX_HORIZON_MIN.
    """
    minutes = None

    found = re.search(r"(\d+)\s*min", text)
    if found:
        minutes = int(found.group(1))
    else:
        found = re.search(r"(\d+)\s*hora", text)
        if found:
            minutes = int(found.group(1)) * 60

    if minutes is None:
        for phrase, value in (
            ("dos horas", 120), ("2 horas", 120),
            ("hora y media", 90), ("1.5 horas", 90),
            ("una hora", 60), ("1 hora", 60),
            ("media hora", 30),
        ):
            if phrase in text:
                minutes = value
                break

    if minutes is None:
        return DEFAULT_HORIZON

    step = cfg.INTERVAL_SECONDS // 60
    minutes = max(step, min(minutes, MAX_HORIZON_MIN))
    # Round up to a whole step: asking for 7 minutes gets 10, never 5, so the
    # answer always covers at least what was asked.
    return int(-(-minutes // step) * step)


def _contains_any(text: str, words: list[str]) -> bool:
    """Test whether any of the given phrases appears in the text.

    Args:
        texto: The lowercased question.
        palabras: Phrases to look for.

    Returns:
        True if at least one phrase is present.
    """
    return any(word in text for word in words)


def main() -> int:
    """Classify a set of example questions and print the routing table.

    Returns:
        Process exit code.
    """
    examples = [
        "¿Cómo está el CO2 del salón 3308?",
        "¿Cuál es la temperatura actual del 3206?",
        "¿Cómo va a estar el CO2 del 3308 en 45 minutos?",
        "¿La humedad va a subir en dos horas?",
        "¿Cuál fue el pico de CO2 ayer en el 3201?",
        "¿Hay algo raro en el 3208?",
        "¿El CO2 está demasiado alto?",
        "¿Y el 3305?",
        "¿El CO2 va a estar muy alto en una hora?",
        "¿Cómo está el piso 3?",
        "¿Cómo va a estar la temperatura del segundo piso en una hora?",
        "¿Cuál fue el máximo de CO2 en la planta baja?",
        "¿Cómo está el 3308 del piso 3?",
        "¿Cómo está el cuarto 3305?",
    ]

    print(f"{'question':62} {'kind':9} {'esp':>4} {'piso':>4} {'min':>4}  variables")
    print("-" * 110)
    for question in examples:
        i = classify(question)
        space = i.space_id if i.space_id is not None else "-"
        floor = i.floor if i.floor is not None else "-"
        variables = ", ".join(i.variables) if i.variables else "(todas)"
        print(
            f"{question[:62]:62} {i.kind:9} {str(space):>4} {str(floor):>4} "
            f"{i.horizon_min:>4}  {variables}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
