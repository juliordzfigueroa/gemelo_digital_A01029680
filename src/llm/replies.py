"""Fixed replies for messages that need no data and no LLM.

Greetings, thanks, requests outside the assistant's scope, questions that do
not say where, room numbers that do not exist, and rooms without a sensor.
None of these has anything for the model to write: the facts are known and
short, and asking Llama to phrase an absence is how it once invented rooms
3310 to 3334. So the code answers, and the wording lives here, in one place.

The wording is conversational on purpose: these are the first answers a new
user sees. The most frequent ones come in two versions so the chat does not
repeat itself word for word. The version is picked from a hash of the
question, so the same question always gets the same answer and a test or a
demonstration can be repeated.

The text is Spanish because the user reads it. Documentation follows the
project standard and stays in English.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-27 - The greeting names the semester and the estimates.

Reference:
    Section 4.3.4 of the research document.
"""

from __future__ import annotations

import zlib

EXAMPLES = (
    "«¿cómo está el CO₂ del 3308?», «¿cómo va a estar el piso 3 en una hora?» "
    "o «dame la temperatura de todos los salones»"
)

TEMPLATES: dict[str, tuple[str, ...]] = {
    "help": (
        "¡Hola! Soy el asistente del gemelo digital de Aulas 3. Te puedo contar "
        "cómo están el CO₂, la temperatura, la humedad, la luz y el movimiento en "
        "los salones, cómo estuvieron algún día del semestre y qué se espera en "
        f"las próximas dos horas. Prueba con {EXAMPLES}. Si debajo de una "
        "respuesta ves «estimado» o «simulado», pregúntame qué significa.",
        "Hola, aquí el asistente del gemelo digital de Aulas 3. Mi trabajo es "
        "contarte cómo está el ambiente del edificio: lecturas de los sensores, "
        "estimaciones de los salones sin sensor, pronósticos de hasta dos horas e "
        "historial de cualquier día del semestre, por salón, por piso o de todo el "
        f"edificio. Puedes empezar con {EXAMPLES}.",
    ),
    "thanks": (
        "¡Con gusto! Si quieres saber algo más del edificio, aquí sigo.",
        "De nada. Cuando quieras revisar otro salón o piso, pregúntame.",
    ),
    "out_of_scope": (
        "Eso se sale de lo que sé hacer: no soy un asistente general, solo "
        "respondo sobre el ambiente del edificio Aulas 3 con los datos de sus "
        f"sensores y el pronóstico del modelo. Por ejemplo, {EXAMPLES}.",
        "Ahí no te puedo ayudar. Lo mío es el edificio Aulas 3: cómo están el "
        "CO₂, la temperatura, la humedad, la luz y el movimiento, y qué se espera "
        "en las próximas horas. Si te sirve, pregúntame por un salón, un piso o "
        "todo el edificio.",
    ),
    "ask_target": (
        "¿De qué salón o piso te gustaría saber? Puedes decirme, por ejemplo, "
        "«el 3308», «el piso 3» o «todos los salones».",
        "Claro, solo dime dónde: un salón como el 3308, un piso como el 3, o "
        "todo el edificio.",
    ),
    "unknown_room_other_building": (
        "El {number} no parece ser de Aulas 3: aquí los salones empiezan con 3 "
        "(edificio, piso y salón, como el 3305). ¿Quisiste decir otro número?",
    ),
    "unknown_room_floor": (
        "No tengo el {number} en mi registro. En {where} conozco estos: {known}. "
        "¿Es alguno de ellos?",
    ),
    "unknown_room_no_floor": (
        "No tengo el {number} en mi registro: Aulas 3 va de la planta baja al "
        "piso {top_floor}.",
    ),
    "no_sensor": (
        "{subject} todavía no tiene sensor, así que no tengo lecturas de ahí. Por "
        "ahora hay datos de {rooms}; si quieres, te cuento de alguno.",
    ),
    "no_sensor_alias": (
        "El {asked} es el mismo salón que el {label}, y ese todavía no tiene "
        "sensor. Por ahora hay datos de {rooms}.",
    ),
    "estimado_note": (
        "{alias}Valores estimados: {subject} no tiene sensor. Salen del generador "
        "físico, calibrado con {rooms}. Error típico: {error}.",
    ),
    "estimado_sensor_note": (
        "{alias}Valores estimados: el sensor {of_subject} no tiene datos suficientes. "
        "Salen del generador físico, calibrado con {rooms}. Error típico: {error}.",
    ),
    "simulado_note": (
        "{alias}Valores simulados: el sensor {of_subject} no midió ese momento. Salen "
        "del generador físico, calibrado con sus propias lecturas. Error típico: {error}.",
    ),
    "no_data_moment": (
        "No tengo datos {of_subject} para ese momento. De ahí hay medido {measured}, "
        "y generado {generated}.",
    ),
    "estimate_no_anomaly": (
        "{subject} no tiene sensor, y las anomalías solo se buscan en lecturas "
        "reales: en una estimación no hay nada que pueda fallar. Si quieres, te "
        "doy una estimación de cómo está ahora.",
    ),
}


def reply(kind: str, question: str = "", **values: object) -> str:
    """Return the fixed reply of one kind, filled in.

    Args:
        kind: A key of TEMPLATES.
        question: The question as typed. Picks the version, so the same
            question always gets the same wording.
        **values: The placeholders the template uses.

    Returns:
        The reply for the user.
    """
    versions = TEMPLATES[kind]
    chosen = versions[zlib.crc32(question.strip().lower().encode("utf-8")) % len(versions)]
    return chosen.format(**values)


def spoken_list(items: list[str]) -> str:
    """Join names the way a person says them: "a, b y c".

    Args:
        items: Names in order.

    Returns:
        The joined text.
    """
    if len(items) <= 1:
        return "".join(items)
    return f"{', '.join(items[:-1])} y {items[-1]}"


# Labels that take "la": "la oficina mentores", "la sala de juntas 1".
FEMININE_LABELS = ("oficina", "sala")


def asked_alias(label: str, aliases: list[str], question: str) -> str | None:
    """The other number the question used for a space, if it used one.

    Args:
        label: How the space is called ("3301").
        aliases: Other numbers it is known by, from the registry.
        question: The question as typed.

    Returns:
        The alias found in the question ("3302"), or None.
    """
    for alias in aliases:
        if alias != label and alias.isdigit() and alias in question.lower():
            return alias
    return None


def capitalized(label: str) -> str:
    """The label with its article, starting with a capital: "El 3201"."""
    named = with_article(label)
    return named[:1].upper() + named[1:]


def origin_note(
    origin: str, label: str, aliases: list[str], rooms: list[str], error: str, question: str,
    has_sensor: bool = False,
) -> str:
    """The note the code adds after an answer built from generated values.

    It says whether they are estimated (no sensor) or simulated (the sensor
    did not measure that moment), what they come from and how far off they
    can be. The model is asked to say so too, but this note is written by
    the code, so it is there even when the model forgets.

    Args:
        origin: "estimado" or "simulado".
        label: How the space is called.
        aliases: Other numbers it is known by.
        rooms: Labels of the measured rooms the parameters came from.
        error: The expected error, already written ("CO₂ ±190 ppm, ...").
        question: The question as typed.
        has_sensor: For an estimate, whether the space has a sensor without
            enough data (the 3304) rather than none.

    Returns:
        A sentence for the user.
    """
    asked = asked_alias(label, aliases, question)
    alias = f"El {asked} es el mismo salón que el {label}. " if asked else ""
    kind = "estimado_sensor" if origin == "estimado" and has_sensor else origin
    return reply(
        f"{kind}_note", question, alias=alias, subject=with_article(label),
        of_subject=of_label(label), rooms=spoken_list(rooms), error=error,
    )


def of_label(label: str) -> str:
    """Put "del" or "de la" before a space label: "del 3308", "de la sala de juntas 1"."""
    named = with_article(label)
    return "de " + named if named.startswith("la ") else "d" + named


def with_article(label: str) -> str:
    """Put the right article before a space label: "el 3309", "la sala de juntas 1"."""
    return f"la {label}" if label.startswith(FEMININE_LABELS) else f"el {label}"


def unknown_room(
    number: str, question: str, floor_spaces: dict[int, list[int]], labels: dict[int, str]
) -> str:
    """Say that a room number is not in the registry, and what is.

    Args:
        number: The four-digit number from the question.
        question: The question as typed, to pick the wording.
        floor_spaces: Space ids by floor, from the registry.
        labels: Label of every space, from the registry.

    Returns:
        A sentence for the user.
    """
    if not number.startswith("3"):
        return reply("unknown_room_other_building", question, number=number)

    floor = int(number[1])
    if floor not in floor_spaces:
        return reply(
            "unknown_room_no_floor", question, number=number, top_floor=max(floor_spaces)
        )

    known = spoken_list([str(labels[s]) for s in sorted(floor_spaces[floor])])
    where = "la planta baja" if floor == 0 else f"el piso {floor}"
    return reply("unknown_room_floor", question, number=number, where=where, known=known)


def no_sensor(label: str, aliases: list[str], rooms: list[str], question: str) -> str:
    """Say that a registered space has no readings, and which spaces do.

    Args:
        label: How the space is called ("3301", "sala de juntas 1").
        aliases: Other numbers it is known by, from the registry.
        rooms: Labels of the spaces that do have readings.
        question: The question as typed. When it named the space by another
            number (the 3302 is the 3301), the answer says so.

    Returns:
        A sentence for the user.
    """
    asked = asked_alias(label, aliases, question)

    # Room numbers go bare in a list ("3206, 3208"); names take their article
    # ("y la oficina mentores").
    named_rooms = []
    for room in rooms:
        named_rooms.append(room if room.isdigit() else with_article(room))
    listed = spoken_list(named_rooms)

    if asked:
        return reply("no_sensor_alias", question, asked=asked, label=label, rooms=listed)
    return reply("no_sensor", question, subject=capitalized(label), rooms=listed)
