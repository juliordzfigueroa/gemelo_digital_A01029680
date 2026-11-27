"""What the labels under an answer mean, in the user's words.

Every answer built with the generator says where its values come from:
"medido", "simulado", "estimado" or, for a floor or the building, "medidos y
estimados". This file explains those words in two places: the welcome the
viewer shows when it opens, and the answer to a question such as "¿qué
significa que un salón esté estimado?".

Like replies.py, the code writes these answers, not Llama. They come out the
same every time and at once, and the facts in them (which rooms have a
sensor, which days were measured, the typical error) come from the data.

The text is Spanish because the user reads it.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-27 - Initial implementation.

Reference:
    Sections 4.3.4 and 4.4 of the research document.
"""

from __future__ import annotations

# One per key of vocabulary.TERM_WORDS. The placeholders are the keys of
# api.origins.facts().
DEFINITIONS = {
    "medido": (
        "«Medido» quiere decir que el valor lo leyó el sensor del salón en ese "
        "momento. Hay sensores en {rooms}, con lecturas {measured_span}."
    ),
    "simulado": (
        "«Simulado» quiere decir que el salón sí tiene sensor, pero no midió ese "
        "momento: sus lecturas van {measured_span}. El valor lo calcula el "
        "generador físico con lo que ese mismo salón midió en esos días."
    ),
    "estimado": (
        "«Estimado» quiere decir que el salón no tiene sensor. El valor lo "
        "calcula el generador físico con el horario de clases del salón y con lo "
        "que midieron los salones del mismo tipo que sí tienen sensor. Sirve "
        "para darse una idea de cómo está, pero no es una lectura."
    ),
    "mixto": (
        "«Medidos y estimados» aparece en las respuestas de un piso o de todo el "
        "edificio: unos salones tienen lecturas del sensor y otros son "
        "estimados. Si dice «simulados y estimados», ninguno se midió en ese "
        "momento: los que tienen sensor van simulados y los demás, estimados. La "
        "lista de salones de la respuesta marca cuál es cuál."
    ),
    "error": (
        "El error típico dice qué tanto se aleja lo simulado o lo estimado de lo "
        "que mide un sensor: {error}. Se calculó estimando cada salón con sensor "
        "como si no lo tuviera y comparando su día típico, hora por hora, con el "
        "que midió. Vale para el patrón del día; un momento en particular puede "
        "alejarse más."
    ),
    "generador": (
        "El generador físico calcula el ambiente de un salón cada 5 minutos: "
        "cuánta gente hay según el horario de clases, cuánto suben el CO₂ y la "
        "humedad con esa gente y cuánto bajan con la ventilación, y la "
        "temperatura, la luz y el movimiento. Está calibrado con los salones que "
        "tienen sensor y sus valores cubren el semestre: {generated_span}."
    ),
}

# What one space gets, when the question names it. Picked by
# api.origins.space_status().
SPACE_STATUS = {
    "own": (
        "{subject} tiene sensor: lo que midió ({measured_span}) sale como "
        "medido, y cualquier otro momento del semestre, como simulado."
    ),
    "few": (
        "{subject} tiene sensor, pero midió muy poco para calibrar el generador "
        "con sus lecturas: lo que midió sale como medido, y lo demás, como "
        "estimado, con base en {based_on}."
    ),
    "none": (
        "{subject} no tiene sensor, así que todo lo suyo sale como estimado, con "
        "base en {based_on}."
    ),
    "only_measured": (
        "{subject} tiene sensor, pero midió muy poco para calibrar el generador: "
        "solo tengo lo que midió ({measured_span}), sin simulado ni estimado."
    ),
    "nothing": (
        "{subject} no tiene sensor, y ningún espacio de su tipo se ha medido: no "
        "hay de dónde sacar una estimación, así que no tengo valores de ahí."
    ),
}

WELCOME = (
    "¡Hola! Soy el asistente del gemelo digital de Aulas 3. Pregúntame por el "
    "CO₂, la temperatura, la humedad, la luz o el movimiento de un salón, de un "
    "piso o de todo el edificio: cómo está, cómo estuvo algún día del semestre "
    "o qué se espera en las próximas dos horas.\n\n"
    "Debajo de cada respuesta te digo de dónde salen los valores:\n"
    "• Medido: lo leyó el sensor del salón. Hay sensores en {rooms}.\n"
    "• Simulado: el salón tiene sensor, pero no midió ese momento. El generador "
    "físico lo calcula con lo que ese salón midió otros días.\n"
    "• Estimado: el salón no tiene sensor. El generador lo calcula con lo que "
    "midieron los salones de su tipo.\n\n"
    "Lo simulado y lo estimado se alejan de lo medido, en promedio, {error}. "
    "Si tienes dudas, pregúntame, por ejemplo, «¿qué significa estimado?»."
)

# Without the generator there is nothing simulated or estimated to explain.
WELCOME_MEASURED_ONLY = (
    "¡Hola! Soy el asistente del gemelo digital de Aulas 3. Pregúntame por el "
    "CO₂, la temperatura, la humedad, la luz o el movimiento de un salón, de un "
    "piso o de todo el edificio: cómo está, cómo estuvo o qué se espera en las "
    "próximas dos horas. Por ahora solo tengo lo que midieron los sensores, en "
    "{rooms}, {measured_span}."
)


def welcome(facts: dict) -> str:
    """Write the first message of the chat: what it answers and what the labels mean.

    Args:
        facts: From api.origins.facts().

    Returns:
        The message for the user.
    """
    if not facts["generated"]:
        return WELCOME_MEASURED_ONLY.format(**facts)
    return WELCOME.format(**facts)


def explain(terms: list[str], facts: dict, status: str | None = None, **space: str) -> str:
    """Define the labels a question asks about, and say which one a space gets.

    Args:
        terms: Keys of DEFINITIONS, from Intent.terms.
        facts: From api.origins.facts().
        status: Key of SPACE_STATUS for the space the question names, or
            None when it names none.
        **space: What SPACE_STATUS quotes: subject, measured_span, based_on.

    Returns:
        The answer for the user, one paragraph per label. The space goes
        first: "¿por qué el 3201 sale estimado?" is answered by its status.
    """
    paragraphs = [DEFINITIONS[term].format(**facts) for term in terms]
    if status:
        paragraphs.insert(0, SPACE_STATUS[status].format(**space))
    return "\n\n".join(paragraphs)
