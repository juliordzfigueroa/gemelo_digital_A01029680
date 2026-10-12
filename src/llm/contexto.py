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
    2026-10-12 - Initial implementation.

Reference:
    Section 4.3.4 of the research document.
"""

from __future__ import annotations

from llm.clasificador import Intencion

# --------------------------------------------------------------------------
# System prompt
# --------------------------------------------------------------------------
# Three of these instructions are defensive rather than stylistic. Saying "use
# only the data I give you" and "say so instead of inventing" is what keeps the
# model from filling a gap with a confident guess, which is the failure this
# architecture exists to avoid. Demanding units is what makes a wrong number
# visible as wrong.
SISTEMA = (
    "Eres el asistente del Gemelo Digital del edificio Aulas 3 del "
    "Tecnológico de Monterrey Campus Santa Fe.\n"
    "Reglas:\n"
    "- Responde en español, en máximo 3 oraciones.\n"
    "- Usa exclusivamente los datos que recibes como contexto. "
    "No calcules, no estimes y no completes lo que falte.\n"
    "- Si los datos no alcanzan para responder, dilo explícitamente en lugar "
    "de inventar un valor.\n"
    "- Incluye las unidades: ppm para CO₂, °C para temperatura, %RH para "
    "humedad, lux para luz.\n"
    "- Si el contexto marca una anomalía o un valor peligroso, dilo primero.\n"
    "- No repitas la pregunta ni expliques de dónde vienen los datos."
)

# Comfort ranges used to annotate the readings so the model can say "normal" or
# "alto" without judging it on its own.
#
# These are working values, not taken from a standard: section 4.3.4 does not
# fix them, and the building's own criteria are still pending. They belong in
# config.py once they are decided, next to the physical limits.
RANGO_SALUDABLE = {
    "co2": (None, 800.0),
    "temperatura": (18.0, 26.0),
    "humedad": (30.0, 60.0),
}

# Escalating CO2 bands, checked from worst to least bad.
NIVELES_CO2 = (
    (1500.0, "PELIGROSO", "riesgo para la salud, ventilar de inmediato"),
    (1000.0, "ALTO", "ventilación recomendada"),
    (800.0, "MODERADO", "aceptable, conviene vigilar"),
)

ETIQUETAS_UNIDAD = {
    "co2": "ppm",
    "temperatura": "°C",
    "humedad": "%RH",
    "luz": "lux",
    "movimiento": "(1 = hay movimiento, 0 = no)",
}


def construir_prompt(intencion: Intencion, datos: dict) -> tuple[str, str]:
    """Build the system and user prompts for one question.

    Args:
        intencion: Result of clasificador.clasificar().
        datos: Data the orchestrator retrieved. Its shape depends on the
            intent; each _prompt_* function documents what it reads and
            tolerates missing keys.

    Returns:
        A tuple of (system prompt, user prompt).
    """
    constructores = {
        "actual": _prompt_actual,
        "predictiva": _prompt_predictiva,
        "historica": _prompt_historica,
        "anomalia": _prompt_anomalia,
    }
    constructor = constructores.get(intencion.tipo)
    if constructor is None:
        return SISTEMA, (
            f"El usuario preguntó: «{intencion.texto_original}»\n\n"
            "No se reconoció el tipo de consulta, así que no se recuperó "
            "ningún dato. Dile que reformule la pregunta."
        )
    return SISTEMA, constructor(intencion, datos)


def _prompt_actual(intencion: Intencion, datos: dict) -> str:
    """Describe the present state of one space.

    Args:
        intencion: The classified question.
        datos: Expects "espacio", "lecturas" and "timestamp".

    Returns:
        The user prompt.
    """
    lecturas = datos.get("lecturas", {})
    comentarios = _comentarios_umbrales(lecturas)

    partes = [
        f"El usuario preguntó: «{intencion.texto_original}»",
        "",
        f"LECTURAS ACTUALES del espacio {datos.get('espacio', 'desconocido')} "
        f"a las {datos.get('timestamp', 'hora desconocida')}:",
        _formatear_lecturas(lecturas),
    ]
    if comentarios:
        partes += ["", "REFERENCIA DE RANGOS:", comentarios]
    partes += ["", "Responde en español usando solo estos datos."]
    return "\n".join(partes)


def _prompt_predictiva(intencion: Intencion, datos: dict) -> str:
    """Describe a forecast for one space.

    Only the first, middle and last predicted steps go into the prompt. Handing
    the model all 24 rows of a two-hour forecast buries the trend in noise and
    invites it to recite the table instead of reading it.

    Args:
        intencion: The classified question.
        datos: Expects "espacio", "lectura_actual", "predicciones",
            "horizonte_min", "metodo" and optionally "advertencia".

    Returns:
        The user prompt.
    """
    actual = datos.get("lectura_actual", {})
    predicciones = datos.get("predicciones", [])
    variables = intencion.variables or list(actual.keys())

    if datos.get("error"):
        return (
            f"El usuario preguntó: «{intencion.texto_original}»\n\n"
            f"NO SE PUDO PREDECIR: {datos['error']}\n\n"
            "Dile al usuario que la predicción no está disponible y por qué. "
            "No estimes el valor tú."
        )

    lineas = [
        f"  +{p.get('offset_min', '?')} min: "
        + _compacto({v: p[v] for v in variables if v in p})
        for p in _hitos(predicciones)
    ]

    partes = [
        f"El usuario preguntó: «{intencion.texto_original}»",
        "",
        f"ESTADO ACTUAL del espacio {datos.get('espacio', 'desconocido')}:",
        _formatear_lecturas({v: actual[v] for v in variables if v in actual}),
        "",
        f"PREDICCIÓN DEL GRU (método {datos.get('metodo', 'directo')}, "
        f"horizonte {datos.get('horizonte_min', '?')} min):",
        "\n".join(lineas) if lineas else "  Sin datos de predicción.",
    ]

    if datos.get("advertencia"):
        partes += ["", f"ADVERTENCIA QUE DEBES TRANSMITIR: {datos['advertencia']}"]

    partes += [
        "",
        "Responde en español usando solo estos datos. Di si la tendencia "
        "sube, baja o se mantiene.",
    ]
    return "\n".join(partes)


def _prompt_historica(intencion: Intencion, datos: dict) -> str:
    """Describe a summary of past readings.

    Args:
        intencion: The classified question.
        datos: Expects "espacio", "variable", "desde", "hasta" and a "resumen"
            holding maximo, minimo, promedio and n_lecturas.

    Returns:
        The user prompt.
    """
    resumen = datos.get("resumen", {})
    variable = datos.get("variable", "la variable")
    unidad = ETIQUETAS_UNIDAD.get(variable, "")

    filas = [
        f"  Máximo:   {_valor(resumen.get('maximo'))} {unidad}".rstrip(),
        f"  Mínimo:   {_valor(resumen.get('minimo'))} {unidad}".rstrip(),
        f"  Promedio: {_valor(resumen.get('promedio'))} {unidad}".rstrip(),
        f"  Lecturas: {resumen.get('n_lecturas', 'desconocidas')}",
    ]

    return "\n".join(
        [
            f"El usuario preguntó: «{intencion.texto_original}»",
            "",
            f"HISTORIAL de {variable} en el espacio "
            f"{datos.get('espacio', 'desconocido')}",
            f"Periodo: {datos.get('desde', '?')} a {datos.get('hasta', '?')}",
            *filas,
            "",
            "Responde en español usando solo estos datos.",
        ]
    )


def _prompt_anomalia(intencion: Intencion, datos: dict) -> str:
    """Describe what the anomaly detector found, or that it found nothing.

    Args:
        intencion: The classified question.
        datos: Expects "espacio", "estado" and a list under "anomalias", each
            entry holding variable, valor, sigma and timestamp.

    Returns:
        The user prompt.
    """
    espacio = datos.get("espacio", "desconocido")
    anomalias = datos.get("anomalias") or []
    estado = datos.get("estado")

    if estado == "no_disponible":
        # An empty list of anomalies and an absent detector are different
        # claims. Reporting "nothing found" when nothing ran would hand the
        # user a false all-clear, which is the one answer this layer must
        # never produce.
        cuerpo = (
            f"NO SE PUDO EVALUAR el espacio {espacio}: "
            + datos.get(
                "detalle",
                "el detector de anomalías todavía no está implementado.",
            )
            + "\nDile al usuario que esta capacidad aún no está disponible. "
            "No afirmes que no hay anomalías."
        )
    elif not anomalias:
        cuerpo = (
            f"El detector estadístico (regla 3σ) no encontró anomalías en el "
            f"espacio {espacio} en este momento. Todas las variables están "
            "dentro de su banda normal."
        )
    else:
        lineas = []
        for a in anomalias:
            sigma = a.get("sigma")
            # Formatted defensively: a detector that reports a missing sigma
            # should degrade to a readable sentence, not crash the request.
            desvio = f"{sigma:.1f}σ" if isinstance(sigma, (int, float)) else "σ desconocida"
            lineas.append(
                f"  - {a.get('variable', '?')}: {_valor(a.get('valor'))} "
                f"({desvio} fuera de lo normal) a las {a.get('timestamp', '?')}"
            )
        cuerpo = (
            f"El detector estadístico (regla 3σ) encontró "
            f"{len(anomalias)} anomalía(s) en el espacio {espacio}:\n"
            + "\n".join(lineas)
        )

    return "\n".join(
        [
            f"El usuario preguntó: «{intencion.texto_original}»",
            "",
            cuerpo,
            "",
            "Responde en español usando solo estos datos. Si hay anomalías, "
            "di qué tan grave es y recomienda una acción concreta. Si no se "
            "pudieron evaluar, dilo y no inventes un diagnóstico.",
        ]
    )


def _hitos(predicciones: list[dict]) -> list[dict]:
    """Pick the first, middle and last step of a forecast.

    Args:
        predicciones: Steps as returned by gru.predecir.predecir().

    Returns:
        At most three steps, in chronological order.
    """
    if len(predicciones) <= 3:
        return predicciones
    return [
        predicciones[0],
        predicciones[len(predicciones) // 2],
        predicciones[-1],
    ]


def _formatear_lecturas(lecturas: dict) -> str:
    """Render one reading per line, with units.

    Args:
        lecturas: Variable name to value.

    Returns:
        The formatted block, or a line saying there is no data.
    """
    if not lecturas:
        return "  Sin datos."
    return "\n".join(
        f"  {variable}: {_valor(valor)} {ETIQUETAS_UNIDAD.get(variable, '')}".rstrip()
        for variable, valor in lecturas.items()
    )


def _compacto(valores: dict) -> str:
    """Render several readings on one line.

    Args:
        valores: Variable name to value.

    Returns:
        A string such as "co2=680 ppm, temperatura=22.1 °C".
    """
    return ", ".join(
        f"{variable}={_valor(valor)} {ETIQUETAS_UNIDAD.get(variable, '')}".rstrip()
        for variable, valor in valores.items()
    )


def _valor(valor: object) -> str:
    """Format a number for the prompt, or mark it as unavailable.

    Args:
        valor: The value to render.

    Returns:
        The number with one decimal, or "no disponible" when it is missing.
    """
    if isinstance(valor, (int, float)):
        return f"{valor:.1f}"
    return "no disponible"


def _comentarios_umbrales(lecturas: dict) -> str:
    """Annotate readings that fall outside their comfort range.

    Giving the model the band a value sits in, instead of letting it decide,
    is what keeps two identical readings from being described as "normal" in
    one answer and "elevado" in the next.

    Args:
        lecturas: Variable name to value, in physical units.

    Returns:
        One line per annotated variable, empty when everything is in range.
    """
    comentarios = []

    co2 = lecturas.get("co2")
    if isinstance(co2, (int, float)):
        for umbral, etiqueta, nota in NIVELES_CO2:
            if co2 > umbral:
                comentarios.append(f"  CO₂ {co2:.0f} ppm: {etiqueta} ({nota})")
                break

    for variable in ("temperatura", "humedad"):
        valor = lecturas.get(variable)
        if not isinstance(valor, (int, float)):
            continue
        minimo, maximo = RANGO_SALUDABLE.get(variable, (None, None))
        unidad = ETIQUETAS_UNIDAD.get(variable, "")
        if minimo is not None and valor < minimo:
            comentarios.append(
                f"  {variable} {valor:.1f} {unidad}: por debajo del rango "
                f"de confort ({minimo:.0f}-{maximo:.0f})"
            )
        elif maximo is not None and valor > maximo:
            comentarios.append(
                f"  {variable} {valor:.1f} {unidad}: por encima del rango "
                f"de confort ({minimo:.0f}-{maximo:.0f})"
            )

    return "\n".join(comentarios)


def main() -> int:
    """Print the prompt built for one example of each intent.

    Returns:
        Process exit code.
    """
    import sys

    from llm import clasificador

    # The prompts carry "CO₂", and the subscript does not exist in the
    # cp1252 code page the Windows console still defaults to, so printing it
    # raises UnicodeEncodeError. The prompt itself must keep the character:
    # it travels to Ollama as UTF-8 and ends up in an answer a person reads.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    casos = [
        (
            "¿Cómo está el CO2 del salón 3302?",
            {
                "espacio": 1,
                "timestamp": "2026-10-12 10:30",
                "lecturas": {
                    "co2": 1180.0, "temperatura": 27.4, "humedad": 61.0,
                    "luz": 380.0, "movimiento": 1.0,
                },
            },
        ),
        (
            "¿Cómo va a estar el CO2 del 3302 en dos horas?",
            {
                "espacio": 1,
                "lectura_actual": {"co2": 680.0},
                "predicciones": [
                    {"offset_min": m, "co2": 680.0 + m * 2} for m in range(5, 125, 5)
                ],
                "horizonte_min": 120,
                "metodo": "iterativo",
                "advertencia": "El error se acumula en cada iteración.",
            },
        ),
        (
            "¿Hay algo raro en el pasillo?",
            {"espacio": 5, "estado": "anomalia", "anomalias": [
                {"variable": "co2", "valor": 1840.0, "sigma": 4.2,
                 "timestamp": "2026-10-12 10:25"},
                {"variable": "humedad", "valor": 88.0, "sigma": None,
                 "timestamp": "2026-10-12 10:25"},
            ]},
        ),
    ]

    for pregunta, datos in casos:
        intencion = clasificador.clasificar(pregunta)
        _, usuario = construir_prompt(intencion, datos)
        print("=" * 76)
        print(f"  tipo: {intencion.tipo}")
        print("=" * 76)
        print(usuario)
        print()

    print("=" * 76)
    print("SYSTEM PROMPT (igual para todos)")
    print("=" * 76)
    print(SISTEMA)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
