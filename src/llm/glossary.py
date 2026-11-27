"""What the labels under an answer mean, in the user's words.

Every answer built with the generator says where its values come from:
"medido", "simulado", "estimado" or, for a floor or the building, "medidos y
estimados". This file explains those words in two places: the welcome the
viewer shows when it opens, and the answer to a question such as "¿qué
significa que un salón esté estimado?".

The welcome is fixed text: it appears as the page opens and cannot wait for
the model. A question is answered the way data questions are. The code
picks the facts (the definitions and, if the question names a space, what
that space gets); Llama writes two or three sentences from them; the code
checks that every number in those sentences is in the facts and adds a line
with what must always be said (generator or GRU, and the typical error). If
the check fails or Ollama is off, the answer is the facts themselves, which
are written to be read.

The text is Spanish because the user reads it.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-27 - Llama writes the answer from the facts; the GRU explained.

Reference:
    Sections 4.3.4 and 4.4 of the research document.
"""

from __future__ import annotations

import re

# One per key of vocabulary.TERM_WORDS. The placeholders are the keys of
# api.origins.facts(). They are both the facts handed to Llama and the
# answer when Llama cannot be used, so they are written to be read.
DEFINITIONS = {
    "medido": (
        "Medido quiere decir que el valor lo leyó el sensor del salón en ese "
        "momento. Hay sensores en {rooms}, con lecturas {measured_span}. El "
        "pronóstico de esos salones lo hace el GRU, a partir de su última hora "
        "de lecturas."
    ),
    "simulado": (
        "Simulado quiere decir que el salón sí tiene sensor, pero no midió ese "
        "momento: las lecturas van {measured_span}. El valor lo calcula el "
        "generador físico con lo que ese mismo salón midió en esos días. No es "
        "un pronóstico del GRU."
    ),
    "estimado": (
        "Estimado quiere decir que el salón no tiene sensor. Todo lo suyo, "
        "también el pronóstico, lo calcula el generador físico con el horario "
        "de clases del salón y con lo que midieron los salones del mismo tipo "
        "que sí tienen sensor. El GRU no lo puede pronosticar, porque necesita "
        "una hora de lecturas reales del salón. Sirve para darse una idea de "
        "cómo está, pero no es una lectura."
    ),
    "mixto": (
        "Medidos y estimados aparece en la respuesta de un piso o de todo el "
        "edificio, y quiere decir que hay de los dos: los salones con sensor "
        "traen su lectura, y los que no tienen sensor, una estimación del "
        "generador físico. La lista de salones de la respuesta marca cuáles son "
        "estimados. Si en cambio dice simulados y estimados, en ese momento no "
        "midió ningún sensor: los salones con sensor van simulados y los demás, "
        "estimados."
    ),
    "error": (
        "El error típico dice qué tanto se aleja lo simulado o lo estimado de "
        "lo que mide un sensor: {error}. Se calculó estimando cada salón con "
        "sensor como si no lo tuviera y comparando su día típico, hora por "
        "hora, con el que midió. Vale para el patrón del día; un momento en "
        "particular puede alejarse más."
    ),
    "generador": (
        "El generador físico calcula el ambiente de un salón cada 5 minutos: "
        "cuánta gente hay según el horario de clases, cuánto suben el CO₂ y la "
        "humedad con esa gente y cuánto bajan con la ventilación, y la "
        "temperatura, la luz y el movimiento. Está calibrado con los salones "
        "que tienen sensor y cubre el semestre, {generated_span}. No es el GRU: "
        "el generador calcula valores donde no hay lecturas, y el GRU pronostica "
        "a partir de lecturas reales."
    ),
    "gru": (
        "El GRU es una red neuronal entrenada con las lecturas de los sensores. "
        "Pronostica hasta dos horas adelante a partir de la última hora medida "
        "de un salón, así que solo pronostica salones con sensor y lecturas "
        "recientes. El pronóstico de los demás lo hace el generador físico."
    ),
}

# What one space gets, when the question names it. Picked by
# api.origins.space_status().
SPACE_STATUS = {
    "own": (
        "{subject} tiene sensor: lo que midió ({measured_span}) sale como "
        "medido, y su pronóstico lo hace el GRU. Cualquier otro momento del "
        "semestre sale como simulado, del generador con sus propias lecturas."
    ),
    "few": (
        "{subject} tiene sensor, pero midió muy poco para calibrar el generador "
        "con sus lecturas: lo que midió sale como medido, y lo demás, también "
        "el pronóstico, como estimado, del generador con base en {based_on}."
    ),
    "none": (
        "{subject} no tiene sensor, así que todo lo suyo, también el pronóstico, "
        "sale como estimado: del generador físico, con base en {based_on}."
    ),
    "only_measured": (
        "{subject} tiene sensor, pero midió muy poco ({measured_span}) para "
        "calibrar el generador o para que el GRU lo pronostique: solo tengo esas "
        "lecturas, sin simulado ni estimado."
    ),
    "nothing": (
        "{subject} no tiene sensor, y ningún espacio de su tipo se ha medido: no "
        "hay de dónde sacar una estimación, así que no tengo valores de ahí."
    ),
}

# The line the code adds after Llama's sentences: what the answer must say
# even when the model leaves it out.
NOTE = "Lo simulado y lo estimado salen del generador físico, no del GRU. Error típico: {error}."
GENERATED_TERMS = ("simulado", "estimado", "mixto", "error", "generador")

# Words Llama may use only if the facts do. Most judge a value or give a
# cause: "no es confiable" or "el sensor no está funcionando" are claims
# nobody measured. "hechos" is the prompt showing through.
JUDGMENT_WORDS = (
    "hechos",
    "confiable", "confiables", "fiable", "seguro", "segura", "alto", "alta",
    "bajo", "baja", "malo", "mala", "funcionando", "no funciona",
    "falla", "fallando", "descompuesto", "roto", "dañado", "exacto", "exacta",
)

SYSTEM_PROMPT = (
    "Eres el asistente del gemelo digital del edificio Aulas 3, en el Tec de "
    "Monterrey campus Santa Fe. Te preguntan qué quieren decir las etiquetas "
    "de tus respuestas (medido, simulado, estimado) o de dónde salen los "
    "valores.\n"
    "Reglas:\n"
    "1. Contesta en español, de tú, en dos o tres oraciones, como se lo "
    "explicarías a un compañero.\n"
    "2. Responde justo lo que se preguntó, solo con los HECHOS. No agregues "
    "números, salones, fechas ni causas que no estén en ellos.\n"
    "3. Cuando hables de un valor, di si sale del sensor, del generador "
    "físico o del GRU, dentro de la explicación, no como frase aparte.\n"
    "4. Nunca contestes con un sí o un no. No digas si algo es confiable, ni "
    "si un error es alto o bajo: di para qué sirve y cuánto se aleja, con el "
    "error típico de los HECHOS.\n"
    "5. Sin listas, sin encabezados y sin repetir la pregunta.\n\n"
    "Ejemplo, con otros HECHOS:\n"
    "PREGUNTA: ¿qué onda con lo estimado?\n"
    "RESPUESTA: Que un salón salga estimado quiere decir que no tiene sensor, "
    "así que nadie midió esos valores. Los calcula el generador físico con el "
    "horario de clases del salón y con lo que midieron otros salones de su "
    "tipo; te dan una idea de cómo está, pero no son una lectura.\n\n"
    "Otro ejemplo:\n"
    "PREGUNTA: ¿le puedo creer a lo simulado?\n"
    "RESPUESTA: Úsalo para ver cómo va el día del salón, no para un momento "
    "exacto. Lo calcula el generador físico, y en promedio se aleja de lo que "
    "mediría el sensor según su error típico."
)

WELCOME = (
    "¡Hola! Soy el asistente del gemelo digital de Aulas 3. Pregúntame cómo "
    "están el CO₂, la temperatura, la humedad, la luz o el movimiento de un "
    "salón, de un piso o de todo el edificio, cómo estuvieron algún día del "
    "semestre o qué se espera en las próximas dos horas.\n\n"
    "Abajo de cada respuesta te digo de dónde salen los números. Medido es lo "
    "que leyó un sensor; hay en {rooms}. Simulado es un salón con sensor en un "
    "momento que no midió, y estimado, un salón sin sensor: esos dos los "
    "calcula el generador físico, con un error típico de {error}.\n\n"
    "Si algo no queda claro, pregúntame, por ejemplo, «¿qué significa "
    "estimado?»."
)

# Without the generator there is nothing simulated or estimated to explain.
WELCOME_MEASURED_ONLY = (
    "¡Hola! Soy el asistente del gemelo digital de Aulas 3. Pregúntame cómo "
    "están el CO₂, la temperatura, la humedad, la luz o el movimiento de un "
    "salón, de un piso o de todo el edificio, cómo estuvieron o qué se espera "
    "en las próximas dos horas. Por ahora solo tengo lo que midieron los "
    "sensores, en {rooms}, {measured_span}."
)

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
YES_OR_NO = re.compile(r"\W*(sí|si|no)\b[,.!]")


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
    """Write the facts behind an answer about the labels, one paragraph each.

    They go to Llama as its facts, and to the user as the answer when Llama
    cannot be used.

    Args:
        terms: Keys of DEFINITIONS, from Intent.terms.
        facts: From api.origins.facts().
        status: Key of SPACE_STATUS for the space the question names, or
            None when it names none.
        **space: What SPACE_STATUS quotes: subject, measured_span, based_on.

    Returns:
        The paragraphs. The space goes first: "¿por qué el 3201 sale
        estimado?" is answered by its status.
    """
    paragraphs = [DEFINITIONS[term].format(**facts) for term in terms]
    if status:
        paragraphs.insert(0, SPACE_STATUS[status].format(**space))
    return "\n\n".join(paragraphs)


def prompt(question: str, facts_text: str) -> tuple[str, str]:
    """Build the two prompts for Llama.

    Args:
        question: The question as typed.
        facts_text: From explain().

    Returns:
        The system prompt and the user prompt.
    """
    return SYSTEM_PROMPT, f"HECHOS:\n{facts_text}\n\nPREGUNTA: {question}"


def grounded(answer: str, facts_text: str) -> bool:
    """Test that Llama said nothing the facts do not say.

    Three checks. Every number must be one of the facts: room numbers,
    dates and errors are all numbers, so a room or an error the model made
    up fails here. No word of JUDGMENT_WORDS unless the facts use it: the
    model once answered "no es confiable", and once blamed a broken sensor.
    And no answer that starts with "sí" or "no": to "¿le puedo creer?" it
    answered "No, no es una lectura", which reads as "don't trust it".

    Args:
        answer: What Llama wrote.
        facts_text: The facts it was given.

    Returns:
        True when the answer passes both checks.
    """
    known = {n.replace(",", ".") for n in NUMBER.findall(facts_text)}
    if not all(n.replace(",", ".") in known for n in NUMBER.findall(answer)):
        return False
    if YES_OR_NO.match(answer.lower()):
        return False
    answer_words = set(re.findall(r"\w+", answer.lower()))
    facts_words = set(re.findall(r"\w+", facts_text.lower()))
    return not any(w in answer_words and w not in facts_words for w in JUDGMENT_WORDS)


def note(terms: list[str], facts: dict, status: str | None = None) -> str:
    """The line the code adds after Llama's sentences.

    Args:
        terms: Keys of DEFINITIONS.
        facts: From api.origins.facts().
        status: Key of SPACE_STATUS, or None.

    Returns:
        That generated values come from the generator and not the GRU, with
        their typical error; empty when the answer is not about them, or is
        about a space that has none.
    """
    if status is not None:
        about_generated = status in ("own", "few", "none")
    else:
        about_generated = any(t in GENERATED_TERMS for t in terms)
    return NOTE.format(**facts) if about_generated else ""
