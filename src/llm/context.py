"""Prompt construction for the conversational layer.

Builds the two strings sent to the LLM: a system prompt that fixes its role and
its limits, and a user prompt carrying the data the orchestrator retrieved.
Questions about one space are written here; questions about a floor or the
whole building in llm/floor_prompt.py; how each number is written, in
llm/formatting.py.

This is the retrieval-augmented half of the pattern. The division of labour is
strict and worth stating plainly: the GRU, the database and the anomaly
detector produce every number, and the LLM only reads them and writes a
sentence. It is never asked to compute, compare across spaces or estimate
anything, because a language model asked to do arithmetic will produce a
fluent, plausible, wrong answer, and nothing downstream could tell.

The prompt text is Spanish because the answer is for the user. Documentation
follows the project standard and stays in English.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Generated values (simulated or estimated) are labelled in
    every prompt of one space.

Reference:
    Section 4.3.4 of the research document.
"""

from __future__ import annotations

from llm import floor_prompt
from llm import formatting as fmt
from llm.classifier import Intent

# Three of these instructions are defensive rather than stylistic. Saying "use
# only the data I give you" and "say so instead of inventing" is what keeps the
# model from filling a gap with a confident guess, which is the failure this
# architecture exists to avoid. Demanding units is what makes a wrong number
# visible as wrong.
SYSTEM_PROMPT = (
    "Eres el asistente del Gemelo Digital del edificio Aulas 3 del "
    "Tecnológico de Monterrey Campus Santa Fe.\n"
    "Reglas:\n"
    "- Responde en español, en máximo 3 oraciones.\n"
    "- Usa exclusivamente los datos que recibes como contexto. "
    "No calcules, no estimes y no completes lo que falte.\n"
    "- Si los datos no alcanzan para responder, dilo explícitamente en lugar "
    "de inventar un valor.\n"
    "- Incluye las unidades: ppm para CO₂, °C para temperatura, %RH para "
    "humedad, % para luz.\n"
    "- Si el contexto marca una anomalía o un valor peligroso, dilo primero.\n"
    "- No repitas la pregunta ni expliques de dónde vienen los datos."
)


def build_prompt(intent: Intent, data: dict) -> tuple[str, str]:
    """Build the system and user prompts for one question.

    Args:
        intent: Result of classifier.classify().
        data: Data the orchestrator retrieved. Its shape depends on the
            intent; each prompt function documents what it reads.

    Returns:
        A tuple of (system prompt, user prompt).
    """
    if data.get("scope") in ("floor", "building"):
        return SYSTEM_PROMPT, floor_prompt.prompt(intent, data)
    if intent.kind == "forecast":
        user = _prompt_forecast(intent, data)
    elif intent.kind == "history":
        user = _prompt_history(intent, data)
    elif intent.kind == "anomaly":
        user = _prompt_anomaly(intent, data)
    else:
        user = _prompt_current(intent, data)
    if data.get("origin") in ORIGIN_HEADINGS:
        user = _with_origin(user, data)
    return SYSTEM_PROMPT, user


def _prompt_current(intent: Intent, data: dict) -> str:
    """Describe the present state of one space.

    Args:
        intent: The classified question.
        data: Expects "space", "readings" and "timestamp".

    Returns:
        The user prompt.
    """
    readings = data.get("readings", {})
    # Only what was asked: handed all five, Llama reported all five.
    if intent.variables:
        readings = {v: x for v, x in readings.items() if v in intent.variables}
    notes = fmt.threshold_notes(readings)

    parts = [
        f"El usuario preguntó: «{intent.original_text}»",
        "",
        f"LECTURAS ACTUALES del espacio {data.get('space', 'desconocido')} "
        f"a las {data.get('timestamp', 'hora desconocida')}:",
        fmt.readings(readings),
    ]
    if notes:
        parts += ["", "REFERENCIA DE RANGOS:", notes]
    parts += ["", "Responde en español usando solo estos datos."]
    return "\n".join(parts)


def _prompt_forecast(intent: Intent, data: dict) -> str:
    """Describe a forecast for one space.

    Only the first, middle and last predicted steps go into the prompt. Handing
    the model all 24 rows of a two-hour forecast buries the trend in noise and
    invites it to recite the table instead of reading it.

    Args:
        intent: The classified question.
        data: Expects "space", "current_reading", "predictions",
            "horizon_min", "method" and optionally "warning", or an "error"
            when there is no forecast.

    Returns:
        The user prompt.
    """
    if data.get("error"):
        return (
            f"El usuario preguntó: «{intent.original_text}»\n\n"
            f"NO SE PUDO PREDECIR: {data['error']}\n\n"
            "Dile al usuario que la predicción no está disponible y por qué. "
            "No estimes el valor tú."
        )

    current = data.get("current_reading", {})
    variables = intent.variables or list(current.keys())

    lines = []
    for step in _milestones(data.get("predictions", [])):
        values = {v: step[v] for v in variables if v in step}
        lines.append(f"  +{step.get('offset_min', '?')} min: {fmt.compact(values)}")

    parts = [
        f"El usuario preguntó: «{intent.original_text}»",
        "",
        f"ESTADO ACTUAL del espacio {data.get('space', 'desconocido')}:",
        fmt.readings({v: current[v] for v in variables if v in current}),
        "",
        _forecast_heading(data),
        "\n".join(lines) if lines else "  Sin datos de predicción.",
    ]
    if data.get("warning"):
        parts += ["", f"ADVERTENCIA QUE DEBES TRANSMITIR: {data['warning']}"]
    parts += [
        "",
        "Responde en español usando solo estos datos. Di si la tendencia "
        "sube, baja o se mantiene.",
    ]
    return "\n".join(parts)


def _forecast_heading(data: dict) -> str:
    """Name what produced the forecast: the GRU, or the generator's rows ahead."""
    horizon = data.get("horizon_min", "?")
    if data.get("method") == "generador":
        return f"PRONÓSTICO DEL GENERADOR (horario y clima de esas horas, horizonte {horizon} min):"
    return f"PREDICCIÓN DEL GRU (método {data.get('method', 'directo')}, horizonte {horizon} min):"


def _prompt_history(intent: Intent, data: dict) -> str:
    """Describe a summary of past readings.

    Args:
        intent: The classified question.
        data: Expects "space", "variable", "since", "until" and a "summary"
            holding max, min, mean and n_readings.

    Returns:
        The user prompt.
    """
    summary = data.get("summary", {})
    variable = data.get("variable", "la variable")
    unit = fmt.UNIT_LABELS.get(variable, "")

    return "\n".join([
        f"El usuario preguntó: «{intent.original_text}»",
        "",
        f"HISTORIAL de {variable} en el espacio {data.get('space', 'desconocido')}",
        f"Periodo: {data.get('since', '?')} a {data.get('until', '?')}",
        f"  Máximo:   {fmt.number(summary.get('max'))} {unit}".rstrip(),
        f"  Mínimo:   {fmt.number(summary.get('min'))} {unit}".rstrip(),
        f"  Promedio: {fmt.number(summary.get('mean'))} {unit}".rstrip(),
        f"  Lecturas: {summary.get('n_readings', 'desconocidas')}",
        "",
        "Responde en español usando solo estos datos.",
    ])


def _prompt_anomaly(intent: Intent, data: dict) -> str:
    """Describe what the 3-sigma detector found in the latest reading.

    Three cases, kept apart on purpose: it could not run, it ran and found
    nothing, it ran and found something. Reporting "nothing found" when
    nothing ran would hand the user a false all-clear, the one answer this
    layer must never produce.

    The comfort ranges go in too, labelled as such: a CO2 of 1,162 ppm is
    within the statistical band and still "ALTO" for a person, and without
    that line the model would read "no anomalies" as "all is well".

    Args:
        intent: The classified question.
        data: Expects "space" and "state"; when evaluated, also "timestamp",
            "checked", "readings" and the list of "anomalies"; when
            unavailable, the "detail" saying why.

    Returns:
        The user prompt.
    """
    space = data.get("space", "desconocido")
    parts = [f"El usuario preguntó: «{intent.original_text}»", ""]

    if data.get("state") == "unavailable":
        parts += [
            f"NO SE PUDO EVALUAR el espacio {space}: {data.get('detail', '')}",
            "Dile al usuario que esta revisión no está disponible y por qué. "
            "No afirmes que no hay anomalías.",
        ]
    else:
        checked = ", ".join(fmt.VARIABLE_NAMES.get(v, v) for v in data.get("checked", []))
        parts.append(
            f"DETECTOR DE ANOMALÍAS (regla 3σ) del espacio {space}, lectura de las "
            f"{data.get('timestamp', 'hora desconocida')}. Variables revisadas: {checked}."
        )
        anomalies = data.get("anomalies", [])
        if anomalies:
            parts.append(f"Encontró {len(anomalies)} anomalía(s):")
            for item in anomalies:
                parts.append(f"  - {fmt.anomaly(item)}")
        else:
            parts.append(
                "No encontró anomalías: todas las variables revisadas están dentro "
                "de su banda normal."
            )
        notes = fmt.threshold_notes(data.get("readings", {}))
        if notes:
            parts += ["", "RANGOS DE CONFORT (otra referencia, aparte del detector):", notes]

    parts += [
        "",
        "Responde en español usando solo estos datos. Si hay anomalías, dilas "
        "primero con su valor y aclara que pueden venir del ambiente o de una "
        "falla del sensor. Si no las hay, dilo sin exagerar.",
    ]
    return "\n".join(parts)


# How the prompt names each generated origin, and what it must say about it.
ORIGIN_HEADINGS = {
    "estimado": "{why} Estos valores son ESTIMADOS por el generador físico, "
                "calibrado con los salones {rooms}; no son mediciones.",
    "simulado": "EL SENSOR DEL {space} NO MIDIÓ ESE MOMENTO. Estos valores son SIMULADOS "
                "por el generador físico, calibrado con las lecturas del propio {rooms}; "
                "no son mediciones.",
}


def _with_origin(user: str, data: dict) -> str:
    """Wrap a prompt built from generated rows with what makes them generated.

    The heading goes right after the question and the expected error at the
    end, so the same three prompts serve measured and generated data. The
    API also appends a note written by the code, so the label is there even
    if the model drops it.

    Args:
        user: The prompt as built for measured data.
        data: The retrieved data, with "origin", "based_on" and
            "expected_error".

    Returns:
        The wrapped prompt.
    """
    origin = data["origin"]
    question, rest = user.split("\n\n", 1)
    space = data.get("space", "desconocido")
    why = f"EL SENSOR DEL {space} NO TIENE DATOS SUFICIENTES." if data.get("has_sensor")         else f"EL ESPACIO {space} NO TIENE SENSOR."
    heading = ORIGIN_HEADINGS[origin].format(space=space, why=why, rooms=", ".join(data.get("based_on", [])))
    error = fmt.expected_error(data.get("expected_error"))
    adjective = "estimados" if origin == "estimado" else "simulados"
    return "\n".join([
        question,
        "",
        heading,
        rest,
        "",
        f"ERROR TÍPICO DE ESTOS VALORES: {error}.",
        f"Di en la primera oración que son valores {adjective}, no medidos.",
    ])


def _milestones(predictions: list[dict]) -> list[dict]:
    """Pick the first, middle and last step of a forecast.

    Args:
        predictions: Steps as returned by gru.predict.predict().

    Returns:
        At most three steps, in chronological order.
    """
    if len(predictions) <= 3:
        return predictions
    return [predictions[0], predictions[len(predictions) // 2], predictions[-1]]


def main() -> int:
    """Print the prompt built for one example of each intent.

    Returns:
        Process exit code.
    """
    import sys

    from llm import classifier

    # The prompts carry "CO₂", and the subscript does not exist in the
    # cp1252 code page the Windows console still defaults to, so printing it
    # raises UnicodeEncodeError. The prompt itself must keep the character:
    # it travels to Ollama as UTF-8 and ends up in an answer a person reads.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    cases = [
        (
            "¿Cómo está el CO2 del salón 3308?",
            {
                "space": "3308",
                "timestamp": "2026-10-12 10:30",
                "readings": {
                    "co2": 1180.0, "temperature": 27.4, "humidity": 61.0,
                    "light": 38.0, "motion": 1.0,
                },
            },
        ),
        (
            "¿Cómo va a estar el CO2 del 3308 en dos horas?",
            {
                "space": "3308",
                "current_reading": {"co2": 680.0},
                "predictions": [
                    {"offset_min": m, "co2": 680.0 + m * 2} for m in range(5, 125, 5)
                ],
                "horizon_min": 120,
                "method": "iterativo",
                "warning": "El error se acumula en cada iteración.",
            },
        ),
        (
            "¿Hay algo raro en el 3206?",
            {
                "space": "3206",
                "state": "evaluated",
                "timestamp": "2026-10-12 10:30",
                "checked": ["co2", "temperature", "humidity", "light"],
                "readings": {"co2": 2400.0, "temperature": 24.1, "humidity": 52.0, "light": 55.0},
                "anomalies": [{
                    "variable": "co2", "value": 2400.0, "sigma": 3.3,
                    "band": [-520.4, 2251.1], "direction": "por encima",
                }],
            },
        ),
    ]

    for question, data in cases:
        intent = classifier.classify(question)
        _, user = build_prompt(intent, data)
        print("=" * 76)
        print(f"  tipo: {intent.kind}")
        print("=" * 76)
        print(user)
        print()

    print("=" * 76)
    print("SYSTEM PROMPT (igual para todos)")
    print("=" * 76)
    print(SYSTEM_PROMPT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
