"""Rule-based intent classifier for the user's question.

Routes a question to one of four intents so the orchestrator knows which data
to retrieve before calling the LLM, and reads what the question is about: one
space ("3308"), a whole floor ("piso 3", "tercer piso", "planta baja") or the
whole building. Deliberately deterministic: the same question always yields
the same intent, and no model call is involved.

Using the LLM to classify would add a second failure mode to every request.
A wrong routing decision sends the wrong data into the prompt, and a prompt
built from the wrong data produces a confident, wrong answer. Keyword rules
are auditable and free; that is the better trade here.

The words it looks for live in llm/vocabulary.py; this file holds the rules.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-27 - Reads which labels of an answer a question asks the
    meaning of.

Reference:
    Section 4.3.4 of the research document.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import date, time

from common import config as cfg
from common import spaces
from llm import vocabulary as voc

# Read once at import: a space added to spaces.json needs a restart to be
# recognized in a question.
SPACES = spaces.names()
FLOOR_SPACES = spaces.floor_spaces()

# Every name of every space, longest first, so "3110" is never read as "311"
# plus a stray zero.
NAMES_LONGEST_FIRST = sorted(
    ((name, space_id) for space_id, names in SPACES.items() for name in names),
    key=lambda pair: len(pair[0]),
    reverse=True,
)

# Room numbers are building 3, then floor, then room: 3305 is floor 3,
# room 05. Four digits from 1900 to 2099 are read as years, not rooms.
ROOM_NUMBER = re.compile(r"\b(\d{4})\b")

# Horizon when a forecast question names none: one direct pass of the GRU.
DEFAULT_HORIZON = 30

# A date without a year is in the semester's year.
YEAR = int(cfg.SEMESTER_START[:4])

# "3 de noviembre", "10 de nov".
DAY_OF_MONTH = re.compile(rf"\b(\d{{1,2}})\s+de\s+({'|'.join(voc.MONTHS)})\b")

# "a las 10", "a las 10:30", "a la 1". The minutes are optional.
CLOCK_TIME = re.compile(r"\ba\s+las?\s+(\d{1,2})(?::(\d{2}))?\b")


@dataclass
class Intent:
    """What the orchestrator needs to know before fetching data.

    Attributes:
        kind: One of "current", "forecast", "history" or "anomaly".
        variables: Variables the question mentions. Empty means all of them.
        space_id: Space the question refers to, or None if it names none.
        floor: Floor the question refers to, or None. Only set when the
            question names no space: "el 3308 del piso 3" is about 3308.
        building: The question asks about every space ("todos los
            salones", "el edificio"). Only set when it names no space or floor.
        unknown_room: A room-like number that is not in the registry
            ("3310"), so the answer can say so instead of asking which room.
        chat: Set when the message is not a question about the building:
            "help" (a greeting or "what is this"), "thanks", or
            "out_of_scope" (something a general chatbot would answer, or
            nothing recognizable at all). None for building questions.
        horizon_min: Minutes ahead, meaningful only when kind is "forecast".
        on_day: The date the question asks about ("el 3 de noviembre"), or
            None for the present.
        weekday: A day of the week it names ("el martes"), Monday being 0,
            or None. Which week is decided by api._moment.
        week: "last" ("el martes pasado"), "next" ("el próximo martes") or
            None, when the question says which week.
        day_offset: Days from the present it names ("ayer" is -1), or None.
        at_time: The time of day it asks about ("a las 10", "en la tarde"),
            or None.
        terms: Labels of an answer it asks the meaning of ("¿qué significa
            estimado?"), as keys of vocabulary.TERM_WORDS. Empty for a
            question about the building.
        original_text: The question as the user typed it, so the prompt can
            quote it back verbatim.
    """

    kind: str
    variables: list[str] = field(default_factory=list)
    space_id: int | None = None
    floor: int | None = None
    building: bool = False
    unknown_room: str | None = None
    chat: str | None = None
    horizon_min: int = DEFAULT_HORIZON
    on_day: date | None = None
    weekday: int | None = None
    week: str | None = None
    day_offset: int | None = None
    at_time: time | None = None
    terms: list[str] = field(default_factory=list)
    original_text: str = ""


def classify(question: str) -> Intent:
    """Classify a question into one intent.

    Args:
        question: The question as typed by the user.

    Returns:
        A populated Intent.

    Reference:
        Section 4.3.4 of the research document.
    """
    text = question.lower().strip()
    kind = _detect_kind(text)
    variables = _extract_variables(text)
    space_id = _extract_space(text)

    # A space named in the text wins over a floor, and a floor over the
    # building: "el 3308 del piso 3" is about 3308.
    floor = None
    building = False
    unknown_room = None
    if space_id is None:
        floor = _extract_floor(text)
        unknown_room = _extract_unknown_room(text)
        if floor is None:
            building = _contains_word(text, voc.BUILDING_WORDS)

    has_target = space_id is not None or floor is not None or building or unknown_room
    chat = None if has_target else _detect_chat(text, variables)

    horizon = DEFAULT_HORIZON
    if kind == "forecast":
        horizon = _extract_horizon(text)

    return Intent(
        kind=kind,
        variables=variables,
        space_id=space_id,
        floor=floor,
        building=building,
        unknown_room=unknown_room,
        chat=chat,
        horizon_min=horizon,
        on_day=_extract_day(text),
        weekday=_extract_weekday(text),
        week=_extract_week(text),
        day_offset=_extract_day_offset(text),
        at_time=_extract_time(text),
        terms=_extract_terms(text),
        original_text=question,
    )


def _extract_terms(text: str) -> list[str]:
    """Find the labels of an answer the question asks the meaning of.

    It needs a label ("estimado") and a way of asking what it means ("qué
    significa", "por qué"): "¿cómo está el 3201 estimado?" asks for the
    estimate, not for a definition.

    Args:
        text: The lowercased question.

    Returns:
        Keys of vocabulary.TERM_WORDS, in that order, or an empty list.
    """
    if not _contains_word(text, voc.MEANING_WORDS):
        return []
    if _contains_word(text, voc.ALL_TERMS_WORDS):
        return ["medido", "simulado", "estimado"]
    terms = [term for term, words in voc.TERM_WORDS.items() if _contains_word(text, words)]
    # "medidos y estimados" is one label, not "medido" plus "estimado".
    if "mixto" in terms:
        terms = [t for t in terms if t not in ("medido", "simulado", "estimado")]
    return terms


def _detect_chat(text: str, variables: list[str]) -> str | None:
    """Tell a building question without a target from everything else.

    Called only when the question names no space, floor or building. A
    question that still mentions a variable, a time ahead or behind, an
    anomaly or a building word ("¿cómo está el CO₂?") is about the building
    and only lacks the place; anything else is not for this assistant.

    Args:
        text: The lowercased question.
        variables: Variables found in it.

    Returns:
        "help", "thanks", "out_of_scope", or None for a building question
        that only needs to say where.
    """
    about_building = bool(
        variables
        or _mentions_future(text)
        or _contains_any(text, voc.ANOMALY_WORDS + voc.FORECAST_WORDS + voc.HISTORY_WORDS)
        or _contains_word(text, voc.DOMAIN_WORDS)
    )
    # Explicit general-chatbot requests win even over building words:
    # "explícame qué es el CO₂" asks for a lesson, not a reading.
    if _contains_word(text, voc.OUT_OF_SCOPE_WORDS):
        return "out_of_scope"
    if about_building:
        return None
    if _contains_word(text, voc.THANKS_WORDS):
        return "thanks"
    if _contains_word(text, voc.HELP_WORDS):
        return "help"
    return "out_of_scope"


def _detect_kind(text: str) -> str:
    """Pick the intent by keyword match, resolving conflicts by priority.

    Priority is anomaly, then forecast, then history, then current. It
    encodes urgency rather than precision: a question that says "raro" or
    "peligroso" goes to the detector even when it also asks about the future,
    because answering the safety question matters more than the horizon.

    The cost is visible: "¿el CO2 va a estar muy alto en una hora?" is routed
    as an anomaly and loses its horizon. Fixing it needs a real classifier,
    not a longer keyword list.

    Args:
        text: The lowercased question.

    Returns:
        The intent name.
    """
    if _contains_any(text, voc.ANOMALY_WORDS):
        return "anomaly"
    if _contains_any(text, voc.FORECAST_WORDS) or _mentions_future(text):
        return "forecast"
    if _contains_any(text, voc.HISTORY_WORDS):
        return "history"
    # Everything else asks for the present state, which is also the cheapest
    # answer to produce and the least likely to mislead.
    return "current"


def _mentions_future(text: str) -> bool:
    """Detect an explicit future time reference such as "en 45 minutos".

    Args:
        text: The lowercased question.

    Returns:
        True if the question names a time offset into the future.
    """
    if re.search(r"\ben\s+\d+\s*(min|hora)", text):
        return True
    if re.search(r"\bdentro\s+de\s+\d+\s*(min|hora)", text):
        return True
    return _contains_any(text, voc.FUTURE_PHRASES)


def _extract_variables(text: str) -> list[str]:
    """Collect the environmental variables the question mentions.

    Goes through config.ACTIVE_VARIABLES rather than the synonym table, so
    narrowing the active variables cannot leave the classifier detecting
    something the model no longer predicts.

    Args:
        text: The lowercased question.

    Returns:
        Variable names in the order config declares them. Empty means the
        question did not single any out, which the orchestrator reads as all.
    """
    found = []
    for variable in cfg.ACTIVE_VARIABLES:
        if _contains_any(text, voc.VARIABLE_SYNONYMS.get(variable, [variable])):
            found.append(variable)
    return found


def _extract_space(text: str) -> int | None:
    """Find which registered space the question names.

    Args:
        text: The lowercased question.

    Returns:
        The space id, or None when the question names no space.
    """
    for name, space_id in NAMES_LONGEST_FIRST:
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
    noun = voc.FLOOR_NOUN
    number_words = "|".join(voc.FLOOR_NUMBER_WORDS)
    ordinals = "|".join(voc.FLOOR_ORDINALS)

    floor = None
    digit = re.search(rf"\b{noun}\s+(\d+)\b", text)
    word = re.search(rf"\b{noun}\s+({number_words})\b", text)
    ordinal = re.search(rf"\b({ordinals})\s+{noun}\b", text) or re.search(
        rf"\b{noun}\s+({ordinals})\b", text
    )
    if "planta baja" in text:
        floor = 0
    elif digit:
        floor = int(digit.group(1))
    elif word:
        floor = voc.FLOOR_NUMBER_WORDS[word.group(1)]
    elif ordinal:
        floor = voc.FLOOR_ORDINALS[ordinal.group(1)]
    return floor if floor in FLOOR_SPACES else None


def _extract_day(text: str) -> date | None:
    """Find the date the question asks about.

    Args:
        text: The lowercased question.

    Returns:
        The date, in the semester's year, or None when it names no day or
        the day does not exist ("31 de noviembre").
    """
    found = DAY_OF_MONTH.search(text)
    if not found:
        return None
    try:
        return date(YEAR, voc.MONTHS[found.group(2)], int(found.group(1)))
    except ValueError:
        return None


def _extract_time(text: str) -> time | None:
    """Find the time of day the question asks about.

    The hour is taken as written; "de la tarde" or "pm" after it moves it to
    the afternoon. "A las 3" stays at 3:00, and the answer shows the time it
    used, so a misreading is visible.

    Args:
        text: The lowercased question.

    Returns:
        The time, or None when it names no hour or the hour does not exist.
    """
    found = CLOCK_TIME.search(text)
    if not found:
        for phrase, hour in voc.DAY_PARTS.items():
            if phrase in text:
                return time(hour, 0)
        return None
    hour = int(found.group(1))
    minute = int(found.group(2) or 0)
    after = text[found.end() : found.end() + 15]
    if hour < 12 and any(after.lstrip().startswith(w) for w in voc.AFTERNOON_WORDS):
        hour += 12
    if hour > 23 or minute > 59:
        return None
    return time(hour, minute)


def _extract_weekday(text: str) -> int | None:
    """Find a day of the week the question names.

    Args:
        text: The lowercased question.

    Returns:
        Monday as 0 to Sunday as 6, or None.
    """
    for word, weekday in voc.WEEKDAYS.items():
        if _contains_word(text, [word]):
            return weekday
    return None


def _extract_week(text: str) -> str | None:
    """Find whether the question says which week: "pasado" or "próximo".

    Args:
        text: The lowercased question.

    Returns:
        "last", "next" or None. "Pasado mañana" is a day, not a week.
    """
    without_days = text.replace("pasado mañana", "")
    if _contains_word(without_days, voc.LAST_WEEK_WORDS):
        return "last"
    if _contains_word(without_days, voc.NEXT_WEEK_WORDS):
        return "next"
    return None


def _extract_day_offset(text: str) -> int | None:
    """Find a day counted from the present: "hoy", "ayer", "mañana".

    Args:
        text: The lowercased question.

    Returns:
        Days from the present, or None. "La mañana" (the morning) does not
        count as tomorrow.
    """
    for phrase, offset in voc.RELATIVE_DAYS:
        if phrase == "mañana":
            # Only a "mañana" that does not follow "la": "mañana a las 9 de la
            # mañana" is tomorrow, "a las 9 de la mañana" is today.
            if re.search(r"(?<!la )\bmañana\b", text):
                return offset
        elif _contains_word(text, [phrase]):
            return offset
    return None


def _extract_unknown_room(text: str) -> str | None:
    """Find a room-like number that the registry does not know.

    Runs only when no registered space matched, so "3308" never gets here.

    Args:
        text: The lowercased question.

    Returns:
        The first four-digit number that is not a year, or None.
    """
    for found in ROOM_NUMBER.finditer(text):
        number = found.group(1)
        if not 1900 <= int(number) <= 2099:
            return number
    return None


def _extract_horizon(text: str) -> int:
    """Read the prediction horizon in minutes.

    Values are capped at config.MAX_HORIZON_MIN and rounded up to the
    sampling interval, because the predictor only emits whole 5-minute steps.

    Args:
        text: The lowercased question.

    Returns:
        Minutes ahead, between the sampling interval and MAX_HORIZON_MIN.
    """
    minutes = None
    in_minutes = re.search(r"(\d+)\s*min", text)
    in_hours = re.search(r"(\d+)\s*hora", text)
    if in_minutes:
        minutes = int(in_minutes.group(1))
    elif in_hours:
        minutes = int(in_hours.group(1)) * 60
    else:
        for phrase, value in voc.HORIZON_PHRASES:
            if phrase in text:
                minutes = value
                break

    if minutes is None:
        return DEFAULT_HORIZON

    step = cfg.INTERVAL_SECONDS // 60
    minutes = max(step, min(minutes, cfg.MAX_HORIZON_MIN))
    # Round up to a whole step: asking for 7 minutes gets 10, never 5, so the
    # answer always covers at least what was asked.
    return math.ceil(minutes / step) * step


def _contains_any(text: str, words: list[str]) -> bool:
    """Test whether any of the given phrases appears in the text.

    Args:
        text: The lowercased question.
        words: Phrases to look for.

    Returns:
        True if at least one phrase is present.
    """
    return any(word in text for word in words)


def _contains_word(text: str, words: list[str]) -> bool:
    """Test whether any phrase appears as whole words, not inside another word.

    Args:
        text: The lowercased question.
        words: Phrases to look for.

    Returns:
        True if at least one phrase is present with word boundaries on both
        sides ("todo" matches "de todo" but not "método").
    """
    return any(re.search(rf"\b{re.escape(word)}\b", text) for word in words)


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
        "Qué chingados es esto?",
        "Dame la temperatura de todos los salones",
        "Todos",
        "3310",
        "2905",
        "3302",
        "¿Cómo está la sala de cómputo?",
        "¿Cómo estuvo el CO2 del 3308 en octubre de 2026?",
        "Hola",
        "Gracias",
        "¿Cuál es la capital de Francia?",
        "Escríbeme un poema sobre el aire",
        "Explícame qué es el CO2",
        "¿Qué temperatura hace en París?",
        "¿Cómo están los valores?",
        "¿Qué opinas de la inteligencia artificial?",
    ]

    print(f"{'question':62} {'kind':9} {'esp':>4} {'piso':>4} {'edif':>4} {'otro':>12} {'min':>4}  variables")
    print("-" * 120)
    for question in examples:
        i = classify(question)
        space = i.space_id if i.space_id is not None else "-"
        floor = i.floor if i.floor is not None else "-"
        variables = ", ".join(i.variables) if i.variables else "(todas)"
        print(
            f"{question[:62]:62} {i.kind:9} {str(space):>4} {str(floor):>4} "
            f"{'sí' if i.building else '-':>4} {i.unknown_room or i.chat or '-':>12} "
            f"{i.horizon_min:>4}  {variables}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
