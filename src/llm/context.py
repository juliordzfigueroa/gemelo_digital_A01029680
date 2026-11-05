"""Prompt construction for the conversational layer.

Builds the two strings sent to the LLM: a system prompt that fixes its role and
its limits, and a user prompt carrying the data the orchestrator retrieved.

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
    2026-11-05 - Added the prompt for questions about a whole floor.

Reference:
    Section 4.3.4 of the research document.
"""

from __future__ import annotations

from llm.classifier import Intent

# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------
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

# Comfort ranges used to annotate the readings so the model can say "normal" or
# "alto" without judging it on its own.
#
# These are working values, not taken from a standard: section 4.3.4 does not
# fix them, and the building's own criteria are still pending. They belong in
# config.py once they are decided, next to the physical limits.
HEALTHY_RANGE = {
    "co2": (None, 800.0),
    "temperature": (18.0, 26.0),
    "humidity": (30.0, 60.0),
}

# Escalating CO2 bands, checked from worst to least bad.
CO2_LEVELS = (
    (1500.0, "PELIGROSO", "riesgo para la salud, ventilar de inmediato"),
    (1000.0, "ALTO", "ventilación recomendada"),
    (800.0, "MODERADO", "aceptable, conviene vigilar"),
)

# Changes smaller than this read as "se mantiene": below the resolution that
# matters for a person, and below the sensors' own noise.
TREND_TOLERANCE = {
    "co2": 5.0,
    "temperature": 0.1,
    "humidity": 0.5,
    "light": 1.0,
    "motion": 0.05,
}

UNIT_LABELS = {
    "co2": "ppm",
    "temperature": "°C",
    "humidity": "%RH",
    "light": "%",
    "motion": "(1 = hay movimiento, 0 = no)",
}


def build_prompt(intent: Intent, data: dict) -> tuple[str, str]:
    """Build the system and user prompts for one question.

    Args:
        intencion: Result of classifier.clasificar().
        datos: Data the orchestrator retrieved. Its shape depends on the
            intent; each _prompt_* function documents what it reads and
            tolerates missing keys.

    Returns:
        A tuple of (system prompt, user prompt).
    """
    if data.get("floor") is not None:
        return SYSTEM_PROMPT, _prompt_floor(intent, data)

    builders = {
        "current": _prompt_current,
        "forecast": _prompt_forecast,
        "history": _prompt_history,
        "anomaly": _prompt_anomaly,
    }
    constructor = builders.get(intent.kind)
    if constructor is None:
        return SYSTEM_PROMPT, (
            f"El usuario preguntó: «{intent.original_text}»\n\n"
            "No se reconoció el tipo de consulta, así que no se recuperó "
            "ningún dato. Dile que reformule la pregunta."
        )
    return SYSTEM_PROMPT, constructor(intent, data)


def _prompt_current(intent: Intent, data: dict) -> str:
    """Describe the present state of one space.

    Args:
        intencion: The classified question.
        datos: Expects "espacio", "lecturas" and "timestamp".

    Returns:
        The user prompt.
    """
    readings = data.get("readings", {})
    # Only what was asked: handed all five, Llama reported all five.
    if intent.variables:
        readings = {v: x for v, x in readings.items() if v in intent.variables}
    notes = _threshold_notes(readings)

    parts = [
        f"El usuario preguntó: «{intent.original_text}»",
        "",
        f"LECTURAS ACTUALES del espacio {data.get('space', 'desconocido')} "
        f"a las {data.get('timestamp', 'hora desconocida')}:",
        _format_readings(readings),
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
        intencion: The classified question.
        datos: Expects "espacio", "lectura_actual", "predicciones",
            "horizon_min", "metodo" and optionally "advertencia".

    Returns:
        The user prompt.
    """
    current = data.get("current_reading", {})
    predictions = data.get("predictions", [])
    variables = intent.variables or list(current.keys())

    if data.get("error"):
        return (
            f"El usuario preguntó: «{intent.original_text}»\n\n"
            f"NO SE PUDO PREDECIR: {data['error']}\n\n"
            "Dile al usuario que la predicción no está disponible y por qué. "
            "No estimes el valor tú."
        )

    lines = [
        f"  +{p.get('offset_min', '?')} min: "
        + _compact({v: p[v] for v in variables if v in p})
        for p in _milestones(predictions)
    ]

    parts = [
        f"El usuario preguntó: «{intent.original_text}»",
        "",
        f"ESTADO ACTUAL del espacio {data.get('space', 'desconocido')}:",
        _format_readings({v: current[v] for v in variables if v in current}),
        "",
        f"PREDICCIÓN DEL GRU (método {data.get('method', 'directo')}, "
        f"horizonte {data.get('horizon_min', '?')} min):",
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


def _prompt_history(intent: Intent, data: dict) -> str:
    """Describe a summary of past readings.

    Args:
        intencion: The classified question.
        datos: Expects "espacio", "variable", "desde", "hasta" and a "resumen"
            holding maximo, minimo, promedio and n_lecturas.

    Returns:
        The user prompt.
    """
    summary = data.get("summary", {})
    variable = data.get("variable", "la variable")
    unit = UNIT_LABELS.get(variable, "")

    rows = [
        f"  Máximo:   {_value(summary.get('max'))} {unit}".rstrip(),
        f"  Mínimo:   {_value(summary.get('min'))} {unit}".rstrip(),
        f"  Promedio: {_value(summary.get('promedio'))} {unit}".rstrip(),
        f"  Lecturas: {summary.get('n_readings', 'desconocidas')}",
    ]

    return "\n".join(
        [
            f"El usuario preguntó: «{intent.original_text}»",
            "",
            f"HISTORIAL de {variable} en el espacio "
            f"{data.get('space', 'desconocido')}",
            f"Periodo: {data.get('since', '?')} a {data.get('until', '?')}",
            *rows,
            "",
            "Responde en español usando solo estos datos.",
        ]
    )


def _prompt_anomaly(intent: Intent, data: dict) -> str:
    """Describe what the anomaly detector found, or that it found nothing.

    Args:
        intencion: The classified question.
        datos: Expects "espacio", "estado" and a list under "anomalias", each
            entry holding variable, valor, sigma and timestamp.

    Returns:
        The user prompt.
    """
    space = data.get("space", "desconocido")
    anomalies = data.get("anomalies") or []
    state = data.get("state")

    if state == "unavailable":
        # An empty list of anomalies and an absent detector are different
        # claims. Reporting "nothing found" when nothing ran would hand the
        # user a false all-clear, which is the one answer this layer must
        # never produce.
        body = (
            f"NO SE PUDO EVALUAR el espacio {space}: "
            + data.get(
                "detail",
                "el detector de anomalías todavía no está implementado.",
            )
            + "\nDile al usuario que esta capacidad aún no está disponible. "
            "No afirmes que no hay anomalías."
        )
    elif not anomalies:
        body = (
            f"El detector estadístico (regla 3σ) no encontró anomalías en el "
            f"espacio {space} en este momento. Todas las variables están "
            "dentro de su banda normal."
        )
    else:
        lines = []
        for a in anomalies:
            sigma = a.get("sigma")
            # Formatted defensively: a detector that reports a missing sigma
            # should degrade to a readable sentence, not crash the request.
            deviation = f"{sigma:.1f}σ" if isinstance(sigma, (int, float)) else "σ desconocida"
            lines.append(
                f"  - {a.get('variable', '?')}: {_value(a.get('value'))} "
                f"({deviation} fuera de lo normal) a las {a.get('timestamp', '?')}"
            )
        body = (
            f"El detector estadístico (regla 3σ) encontró "
            f"{len(anomalies)} anomalía(s) en el espacio {space}:\n"
            + "\n".join(lines)
        )

    return "\n".join(
        [
            f"El usuario preguntó: «{intent.original_text}»",
            "",
            body,
            "",
            "Responde en español usando solo estos datos. Si hay anomalías, "
            "di qué tan grave es y recomienda una acción concreta. Si no se "
            "pudieron evaluar, dilo y no inventes un diagnóstico.",
        ]
    )


def _prompt_floor(intent: Intent, data: dict) -> str:
    """Describe a whole floor: a summary first, then each room.

    The summary (mean, lowest and highest room) comes computed from the API;
    the model only reads it. Rooms are named by their room number, never by
    id, so the answer cannot say "espacio 5".

    Args:
        intent: The classified question.
        data: Built by api._floor_data: "floor", "kind", "rooms", "summary",
            "without_data", "n_spaces" and, by kind, "timestamp",
            "horizon_min", "method", "warning", "since", "until", "errors".

    Returns:
        The user prompt.
    """
    floor = data["floor"]
    rooms = data.get("rooms", [])
    without = data.get("without_data", [])
    floor_name = "PLANTA BAJA (piso 0)" if floor == 0 else f"PISO {floor}"
    floor_phrase = "la planta baja" if floor == 0 else f"el piso {floor}"
    parts = [f"El usuario preguntó: «{intent.original_text}»", ""]

    if data.get("state") == "unavailable":
        parts += [
            f"NO SE PUDO EVALUAR {floor_phrase}: {data.get('detail', '')}",
            "Dile al usuario que esta capacidad aún no está disponible. "
            "No afirmes que no hay anomalías.",
        ]
        return "\n".join(parts)

    coverage = f"{floor_name}: {len(rooms)} de {data.get('n_spaces', '?')} espacios con datos."
    if without:
        coverage += f" Sin datos: {', '.join(without)}."
    parts.append(coverage)

    if not rooms:
        parts += [
            *[f"  {e}" for e in data.get("errors", [])],
            "",
            "Ningún espacio de este piso tiene datos para responder. Dilo así, "
            "sin inventar valores.",
        ]
        return "\n".join(parts)

    kind = data.get("kind")
    summary = data.get("summary", {})

    def summary_line(variable: str, s: dict) -> str:
        unit = UNIT_LABELS.get(variable, "")
        return (
            f"  {variable}: promedio {_value(s.get('mean'))} {unit}; "
            f"mínimo {_value(s.get('min'))} en {s.get('min_room')}; "
            f"máximo {_value(s.get('max'))} en {s.get('max_room')}"
        )

    if kind == "forecast":
        parts += [
            "",
            f"RESUMEN DEL PISO, predicción del GRU a +{data.get('horizon_min', '?')} min "
            f"(método {data.get('method', '?')}):",
            *[summary_line(v, s) for v, s in summary.items()],
            "",
            f"DETALLE POR SALÓN (ahora → en {data.get('horizon_min', '?')} min):",
        ]
        for room in rooms:
            changes = ", ".join(
                f"{v} {_value(room['current'].get(v))} → {_value(room['predicted'].get(v))} "
                f"{UNIT_LABELS.get(v, '')}".rstrip()
                + f" ({_trend(v, room['current'].get(v), room['predicted'].get(v))})"
                for v in data.get("variables", [])
            )
            parts.append(f"  {room['room']}: {changes}")
        if data.get("warning"):
            parts += ["", f"ADVERTENCIA QUE DEBES TRANSMITIR: {data['warning']}"]
    elif kind == "history":
        variable = data.get("variables", ["co2"])[0]
        unit = UNIT_LABELS.get(variable, "")
        s = summary.get(variable, {})
        parts += [
            "",
            f"RESUMEN DEL PISO, historial de {variable} "
            f"({data.get('since', '?')} a {data.get('until', '?')}):",
            f"  máximo {_value(s.get('max'))} {unit} en {s.get('max_room')}; "
            f"mínimo {_value(s.get('min'))} {unit} en {s.get('min_room')}; "
            f"promedio de los salones {_value(s.get('mean'))} {unit}",
            "",
            "DETALLE POR SALÓN:",
        ]
        for room in rooms:
            r = room.get("summary", {})
            parts.append(
                f"  {room['room']}: máximo {_value(r.get('max'))}, mínimo "
                f"{_value(r.get('min'))}, promedio {_value(r.get('promedio'))} {unit}".rstrip()
            )
    else:
        parts += [
            "",
            f"RESUMEN DEL PISO (lecturas actuales, {data.get('timestamp', 'hora desconocida')}):",
            *[summary_line(v, s) for v, s in summary.items()],
            "",
            "DETALLE POR SALÓN:",
        ]
        notes = []
        for room in rooms:
            parts.append(f"  {room['room']}: {_compact(room['readings'])}")
            room_notes = _threshold_notes(room["readings"])
            if room_notes:
                notes += [f"  {room['room']}: {line.strip()}" for line in room_notes.splitlines()]
        if notes:
            parts += ["", "REFERENCIA DE RANGOS:", *notes]

    if data.get("errors"):
        parts += ["", "SALONES SIN PREDICCIÓN:", *[f"  {e}" for e in data["errors"]]]

    parts += [
        "",
        "Responde en español usando solo estos datos. Primero resume el piso en "
        "una oración (promedio y salones extremos); después dedica una oración "
        "corta a cada salón. Para esta pregunta puedes pasar del límite de 3 "
        "oraciones. Si hay espacios sin datos, menciónalo al final.",
    ]
    return "\n".join(parts)


def _trend(variable: str, now: object, later: object) -> str:
    """Say whether a value goes up, down or stays, with the change.

    Written here and handed to the model as text: asked to read the direction
    from two numbers, Llama answered "disminuirá de 665 a 728 ppm".

    Args:
        variable: Variable name, for its tolerance and unit.
        now: Current value.
        later: Predicted value.

    Returns:
        "sube 62.8 ppm", "baja 0.4 °C", "se mantiene", or "sin dato".
    """
    if not isinstance(now, (int, float)) or not isinstance(later, (int, float)):
        return "sin dato"
    change = later - now
    if abs(change) < TREND_TOLERANCE.get(variable, 0.0):
        return "se mantiene"
    unit = UNIT_LABELS.get(variable, "") if variable != "motion" else ""
    word = "sube" if change > 0 else "baja"
    return f"{word} {abs(change):.1f} {unit}".rstrip()


def _milestones(predictions: list[dict]) -> list[dict]:
    """Pick the first, middle and last step of a forecast.

    Args:
        predicciones: Steps as returned by gru.predict.predict().

    Returns:
        At most three steps, in chronological order.
    """
    if len(predictions) <= 3:
        return predictions
    return [
        predictions[0],
        predictions[len(predictions) // 2],
        predictions[-1],
    ]


def _format_readings(readings: dict) -> str:
    """Render one reading per line, with units.

    Args:
        lecturas: Variable name to value.

    Returns:
        The formatted block, or a line saying there is no data.
    """
    if not readings:
        return "  Sin datos."
    return "\n".join(
        f"  {variable}: {_value(value)} {UNIT_LABELS.get(variable, '')}".rstrip()
        for variable, value in readings.items()
    )


def _compact(values: dict) -> str:
    """Render several readings on one line.

    Args:
        valores: Variable name to value.

    Returns:
        A string such as "co2=680 ppm, temperatura=22.1 °C".
    """
    return ", ".join(
        f"{variable}={_value(value)} {UNIT_LABELS.get(variable, '')}".rstrip()
        for variable, value in values.items()
    )


def _value(value: object) -> str:
    """Format a number for the prompt, or mark it as unavailable.

    Args:
        valor: The value to render.

    Returns:
        The number with one decimal, or "no disponible" when it is missing.
    """
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return "no disponible"


def _threshold_notes(readings: dict) -> str:
    """Annotate readings that fall outside their comfort range.

    Giving the model the band a value sits in, instead of letting it decide,
    is what keeps two identical readings from being described as "normal" in
    one answer and "elevado" in the next.

    Args:
        lecturas: Variable name to value, in physical units.

    Returns:
        One line per annotated variable, empty when everything is in range.
    """
    notes = []

    co2 = readings.get("co2")
    if isinstance(co2, (int, float)):
        for threshold, label, note in CO2_LEVELS:
            if co2 > threshold:
                notes.append(f"  CO₂ {co2:.0f} ppm: {label} ({note})")
                break

    for variable in ("temperature", "humidity"):
        value = readings.get(variable)
        if not isinstance(value, (int, float)):
            continue
        lower, upper = HEALTHY_RANGE.get(variable, (None, None))
        unit = UNIT_LABELS.get(variable, "")
        if lower is not None and value < lower:
            notes.append(
                f"  {variable} {value:.1f} {unit}: por debajo del rango "
                f"de confort ({lower:.0f}-{upper:.0f})"
            )
        elif upper is not None and value > upper:
            notes.append(
                f"  {variable} {value:.1f} {unit}: por encima del rango "
                f"de confort ({lower:.0f}-{upper:.0f})"
            )

    return "\n".join(notes)


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
                "space": 1,
                "timestamp": "2026-10-12 10:30",
                "readings": {
                    "co2": 1180.0, "temperature": 27.4, "humidity": 61.0,
                    "light": 380.0, "motion": 1.0,
                },
            },
        ),
        (
            "¿Cómo va a estar el CO2 del 3308 en dos horas?",
            {
                "space": 1,
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
            "¿Hay algo raro en el 3208?",
            {"space": 5, "state": "anomaly", "anomalies": [
                {"variable": "co2", "value": 1840.0, "sigma": 4.2,
                 "timestamp": "2026-10-12 10:25"},
                {"variable": "humidity", "value": 88.0, "sigma": None,
                 "timestamp": "2026-10-12 10:25"},
            ]},
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
