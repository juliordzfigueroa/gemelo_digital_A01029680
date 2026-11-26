"""Prompt and room-by-room detail for questions about a floor or the building.

The work is split in two on purpose. The model writes one or two sentences
from a summary the API computed (mean, lowest and highest room), and detail()
writes the line of every room from the numbers, which the API appends after
the model's sentences. Asked to write both, Llama ran out of tokens halfway
through six rooms, called 632 ppm "alta" and listed rooms 3310 to 3334, which
do not exist.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Moved here from context.py; anomaly questions added.

Reference:
    Section 4.3.4 of the research document.
"""

from __future__ import annotations

from llm import formatting as fmt
from llm.classifier import Intent

# The last instruction of every floor prompt.
SUMMARY_ONLY = (
    "Escribe SOLO el resumen {of_phrase}, en una o dos oraciones: el "
    "promedio, los espacios extremos y, si los hay, los que están fuera de "
    "rango. Usa únicamente los adjetivos de la sección FUERA DE RANGO. No "
    "hagas una lista por salón ni menciones espacios sin datos: el detalle lo "
    "agrega el sistema."
)

# The same instruction for an anomaly question, which has no average to give.
ANOMALIES_ONLY = (
    "Escribe SOLO el resumen {of_phrase}, en una o dos oraciones: cuántos "
    "espacios tienen anomalías y en qué variable, o que no hay ninguna. No "
    "hagas una lista por salón ni menciones espacios sin datos: el detalle lo "
    "agrega el sistema."
)


def scope_names(data: dict) -> tuple[str, str]:
    """Return the heading and the phrase that name a floor or the building.

    Args:
        data: Built by retrieval.floor_data().

    Returns:
        A tuple such as ("PISO 3", "el piso 3").
    """
    if data.get("scope") == "building":
        return "EDIFICIO AULAS 3", "el edificio"
    if data["floor"] == 0:
        return "PLANTA BAJA (piso 0)", "la planta baja"
    return f"PISO {data['floor']}", f"el piso {data['floor']}"


def prompt(intent: Intent, data: dict) -> str:
    """Ask for the summary of a floor or of the building, and nothing else.

    Args:
        intent: The classified question.
        data: Built by retrieval.floor_data(): "scope", "floor", "kind", "rooms",
            "summary" and, by kind, "timestamp", "horizon_min", "method",
            "warning", "since", "until".

    Returns:
        The user prompt.
    """
    rooms = data.get("rooms", [])
    heading, phrase = scope_names(data)
    parts = [f"El usuario preguntó: «{intent.original_text}»", ""]

    if data.get("state") == "unavailable":
        parts += [
            f"NO SE PUDO EVALUAR {phrase}: {data.get('detail', '')}",
            "Dile al usuario que esta revisión no está disponible y por qué. "
            "No afirmes que no hay anomalías.",
        ]
        return "\n".join(parts)

    if not rooms:
        parts += [
            f"{heading}: ningún espacio tiene datos para responder.",
            "",
            "Dilo en una oración, sin inventar valores.",
        ]
        return "\n".join(parts)

    parts.append(f"{heading}: {len(rooms)} espacios con datos.")
    kind = data.get("kind")
    if kind == "anomaly":
        parts += _anomaly_summary(data)
        parts += ["", ANOMALIES_ONLY.format(of_phrase=_of(phrase))]
        return "\n".join(parts)

    if kind == "forecast":
        parts += _forecast_summary(data)
    elif kind == "history":
        parts += _history_summary(data)
    else:
        parts += _current_summary(data)
    parts += ["", SUMMARY_ONLY.format(of_phrase=_of(phrase))]
    return "\n".join(parts)


def _of(phrase: str) -> str:
    """Put "de" before a phrase, contracted when it starts with "el".

    "el piso 3" becomes "del piso 3"; "la planta baja", "de la planta baja".
    """
    if phrase.startswith("el "):
        return "del " + phrase[3:]
    return "de " + phrase


def _of_what(data: dict) -> str:
    """Return "DEL EDIFICIO" or "DEL PISO", for the summary heading."""
    return "DEL EDIFICIO" if data.get("scope") == "building" else "DEL PISO"


def _summary_line(variable: str, s: dict) -> str:
    """Write the mean, lowest and highest room of one variable on one line.

    Args:
        variable: Variable name.
        s: Its summary, with mean, min, min_room, max and max_room.

    Returns:
        The line, indented for the prompt.
    """
    unit = fmt.UNIT_LABELS.get(variable, "")
    return (
        f"  {variable}: promedio {fmt.number(s.get('mean'))} {unit}; "
        f"mínimo {fmt.number(s.get('min'))} en {s.get('min_room')}; "
        f"máximo {fmt.number(s.get('max'))} en {s.get('max_room')}"
    )


def _forecast_summary(data: dict) -> list[str]:
    """Summary block of a floor forecast, with its warning if there is one."""
    lines = [
        "",
        f"RESUMEN {_of_what(data)}, predicción del GRU a "
        f"+{data.get('horizon_min', '?')} min (método {data.get('method', '?')}):",
    ]
    for variable, s in data.get("summary", {}).items():
        lines.append(_summary_line(variable, s))
    if data.get("warning"):
        lines += ["", f"ADVERTENCIA QUE DEBES TRANSMITIR: {data['warning']}"]
    return lines


def _history_summary(data: dict) -> list[str]:
    """Summary block of a floor history: one variable, across its rooms."""
    variable = data.get("variables", ["co2"])[0]
    unit = fmt.UNIT_LABELS.get(variable, "")
    s = data.get("summary", {}).get(variable, {})
    return [
        "",
        f"RESUMEN {_of_what(data)}, historial de {variable} "
        f"({data.get('since', '?')} a {data.get('until', '?')}):",
        f"  máximo {fmt.number(s.get('max'))} {unit} en {s.get('max_room')}; "
        f"mínimo {fmt.number(s.get('min'))} {unit} en {s.get('min_room')}; "
        f"promedio de los salones {fmt.number(s.get('mean'))} {unit}",
    ]


def _current_summary(data: dict) -> list[str]:
    """Summary block of the present state, plus every reading out of range."""
    lines = [
        "",
        f"RESUMEN {_of_what(data)} (lecturas actuales, "
        f"{data.get('timestamp', 'hora desconocida')}):",
    ]
    for variable, s in data.get("summary", {}).items():
        lines.append(_summary_line(variable, s))

    # The adjectives the model may use come from here, so "alto" means the
    # same in every answer.
    notes = []
    for room in data.get("rooms", []):
        for note in fmt.threshold_notes(room["readings"]).splitlines():
            notes.append(f"  {room['room']}: {note.strip()}")
    lines += ["", "FUERA DE RANGO:", *(notes or ["  ninguno"])]
    return lines


def _anomaly_summary(data: dict) -> list[str]:
    """What the 3-sigma detector found in the latest reading of every room."""
    checked = ", ".join(fmt.VARIABLE_NAMES.get(v, v) for v in data.get("variables", []))
    lines = [
        "",
        f"DETECTOR DE ANOMALÍAS (regla 3σ), lecturas de las "
        f"{data.get('timestamp', 'hora desconocida')}. Variables revisadas: {checked}.",
    ]
    found = False
    for room in data.get("rooms", []):
        for item in room["anomalies"]:
            lines.append(f"  {room['room']}: {fmt.anomaly(item)}{_older_reading(data, room)}")
            found = True
    if not found:
        lines.append("  Ningún espacio tiene anomalías.")
    return lines


def _older_reading(data: dict, room: dict) -> str:
    """Say when a room's reading is older than the rest of the floor's.

    A sensor that stopped reporting still has a "latest" reading, possibly
    days old, and an anomaly in it must not read as if it were happening now.

    Args:
        data: Built by retrieval.floor_data(); "timestamp" is the newest
            reading of the floor.
        room: One entry of data["rooms"].

    Returns:
        ", lectura del <fecha>" when the room's reading is older, else "".
    """
    if room.get("timestamp") and room["timestamp"] != data.get("timestamp"):
        return f", lectura del {room['timestamp']}"
    return ""


def detail(data: dict) -> str:
    """Write the room-by-room detail of a floor or building answer.

    Plain text built from the numbers, appended by the API after the
    model's summary, so every room and value in it is exactly what was
    retrieved. Ends with the rooms without data: named for a floor, counted
    for the building, where the list would run to twenty.

    Args:
        data: Built by retrieval.floor_data().

    Returns:
        The detail block, possibly empty.
    """
    if data.get("state") == "unavailable":
        return ""

    lines = []
    for room in data.get("rooms", []):
        lines.append(f"• {room['room']}: {_room_values(data, room)}")
    for error in data.get("errors", []):
        lines.append(f"• {error['room']}: sin pronóstico ({error['reason']})")

    without = data.get("without_data", [])
    if without and data.get("scope") == "building":
        lines.append(f"Los otros {len(without)} espacios del edificio no tienen sensor.")
    elif without:
        _, phrase = scope_names(data)
        lines.append(f"Sin sensor en {phrase}: {', '.join(without)}.")
    return "\n".join(lines)


def _room_values(data: dict, room: dict) -> str:
    """Write the values of one room for the detail block.

    Args:
        data: Built by retrieval.floor_data(); its "kind" and "variables" decide
            what is written.
        room: One entry of data["rooms"].

    Returns:
        Text such as "CO₂ 421.4 ppm · temperatura 22.7 °C".
    """
    kind = data.get("kind")
    variables = data.get("variables", [])

    if kind == "anomaly":
        if not room["anomalies"]:
            return "sin anomalías"
        found = " · ".join(fmt.anomaly_short(item) for item in room["anomalies"])
        return found + _older_reading(data, room)

    if kind == "history":
        v = variables[0] if variables else "co2"
        r = room.get("summary", {})
        return (
            f"{fmt.VARIABLE_NAMES.get(v, v)} máximo {fmt.number(r.get('max'))}, "
            f"mínimo {fmt.number(r.get('min'))}, promedio {fmt.number(r.get('mean'))} "
            f"{fmt.short_unit(v)}"
        ).rstrip()

    parts = []
    for v in variables:
        name = fmt.VARIABLE_NAMES.get(v, v)
        unit = fmt.short_unit(v)
        if kind == "forecast":
            now, later = room["current"].get(v), room["predicted"].get(v)
            text = f"{name} {fmt.number(now)} → {fmt.number(later)} {unit}".rstrip()
            parts.append(f"{text} ({fmt.trend(v, now, later)})")
        else:
            parts.append(f"{name} {fmt.number(room['readings'].get(v))} {unit}".rstrip())
    return " · ".join(parts)
