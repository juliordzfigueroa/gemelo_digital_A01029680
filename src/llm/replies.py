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
    2026-11-09 - Initial implementation.

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
        "los salones, qué se espera en las próximas dos horas y cómo han estado en "
        f"el último día. Prueba con {EXAMPLES}.",
        "Hola, aquí el asistente del gemelo digital de Aulas 3. Mi trabajo es "
        "contarte cómo está el ambiente del edificio: lecturas de los sensores, "
        "pronósticos de hasta dos horas e historial del último día, por salón, por "
        f"piso o de todo el edificio. Puedes empezar con {EXAMPLES}.",
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
