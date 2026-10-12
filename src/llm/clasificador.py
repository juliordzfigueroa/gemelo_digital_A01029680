"""Rule-based intent classifier for the user's question.

Routes a question to one of four intents so the orchestrator knows which data
to retrieve before calling the LLM. Deliberately deterministic: the same
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
    2026-10-12 - Initial implementation.

Reference:
    Section 4.3.4 of the research document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from comun import config as cfg

# --------------------------------------------------------------------------
# Classification vocabulary
# --------------------------------------------------------------------------
PALABRAS_ANOMALIA = [
    "anomalía", "anomalia", "anomalías", "anomalias",
    "raro", "rara", "extraño", "extraña",
    "fuera de lo normal", "anormal", "inusual",
    "problema", "alerta", "alarma", "peligro", "peligroso",
    "excede", "supera", "demasiado", "muy alto", "muy alta",
    "algo mal", "algo está mal", "algo esta mal",
]

PALABRAS_PREDICTIVA = [
    "estará", "estara", "estarán", "estaran",
    "va a estar", "van a estar",
    "predice", "predecir", "predicción", "prediccion", "predecirá",
    "pronostica", "pronóstico", "pronostico",
    "próximos", "proximos", "próximas", "proximas",
    "más tarde", "mas tarde", "al rato",
    "futuro", "tendencia", "subirá", "subira", "bajará", "bajara",
]

PALABRAS_HISTORICA = [
    "ayer", "anteayer", "la semana pasada", "el mes pasado",
    "histórico", "historico", "histórica", "historica", "historial",
    "cuándo fue", "cuando fue", "cuándo estuvo", "cuando estuvo",
    "pico", "máximo", "maximo", "mínimo", "minimo",
    "promedio", "últimos días", "ultimos dias",
    "últimas horas", "ultimas horas", "últimas 24", "ultimas 24",
    "se registró", "se registro", "hubo",
]

PALABRAS_ACTUAL = [
    "ahora", "actualmente", "en este momento", "hoy",
    "cuál es", "cual es", "cuáles son", "cuales son",
    "actual", "cómo está", "como esta", "cómo se encuentra",
    "lectura", "dime el", "dime la",
]

# Maps each active variable to the words a user might use for it. Keys come
# from config so that narrowing VARIABLES_ACTIVAS cannot leave this dict
# claiming to detect something the model no longer predicts.
SINONIMOS_VARIABLE = {
    "co2": ["co2", "co₂", "dióxido de carbono", "dioxido de carbono", "aire"],
    "temperatura": ["temperatura", "calor", "frío", "frio", "fría", "fria", "grados"],
    "humedad": ["humedad", "húmedo", "humedo", "húmeda", "humeda"],
    "luz": ["luz", "iluminación", "iluminacion", "iluminancia", "lux", "luminosidad"],
    "movimiento": ["movimiento", "ocupación", "ocupacion", "personas", "gente", "ocupado"],
}

# Room identifiers. Placeholder until the real Aulas 3 room numbers and their
# mapping to espacio_id are known; the synthetic generator uses ids 1 to 5.
#
# Reference: pending, see src/llm/README.md
ESPACIOS = {
    1: ["3302", "aula 1", "salón 1", "salon 1"],
    2: ["3305", "aula 2", "salón 2", "salon 2"],
    3: ["3201", "aula 3", "salón 3", "salon 3"],
    4: ["3105", "aula 4", "salón 4", "salon 4"],
    5: ["pasillo", "aula 5", "salón 5", "salon 5"],
}

# Horizon ceiling, kept in sync with the iterative predictor.
HORIZONTE_MAX_MIN = 120
HORIZONTE_POR_OMISION = 30


@dataclass
class Intencion:
    """What the orchestrator needs to know before fetching data.

    Attributes:
        tipo: One of "predictiva", "actual", "historica" or "anomalia".
        variables: Variables the question mentions. Empty means all of them.
        espacio_id: Space the question refers to, or None if it names none.
        horizonte_min: Minutes ahead, meaningful only when tipo is
            "predictiva".
        texto_original: The question as the user typed it, so the prompt can
            quote it back verbatim.
    """

    tipo: str
    variables: list[str] = field(default_factory=list)
    espacio_id: int | None = None
    horizonte_min: int = HORIZONTE_POR_OMISION
    texto_original: str = ""


def clasificar(pregunta: str) -> Intencion:
    """Classify a question into one intent.

    Args:
        pregunta: The question as typed by the user.

    Returns:
        A populated Intencion.

    Reference:
        Section 4.3.4 of the research document.
    """
    texto = pregunta.lower().strip()
    tipo = _detectar_tipo(texto)

    return Intencion(
        tipo=tipo,
        variables=_extraer_variables(texto),
        espacio_id=_extraer_espacio(texto),
        horizonte_min=(
            _extraer_horizonte(texto)
            if tipo == "predictiva"
            else HORIZONTE_POR_OMISION
        ),
        texto_original=pregunta,
    )


def _detectar_tipo(texto: str) -> str:
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
    if _contiene_alguna(texto, PALABRAS_ANOMALIA):
        return "anomalia"
    if _contiene_alguna(texto, PALABRAS_PREDICTIVA) or _menciona_futuro(texto):
        return "predictiva"
    if _contiene_alguna(texto, PALABRAS_HISTORICA):
        return "historica"
    if _contiene_alguna(texto, PALABRAS_ACTUAL):
        return "actual"
    # Nothing matched. The present state is the cheapest answer to produce and
    # the least likely to mislead, so it is the fallback.
    return "actual"


def _menciona_futuro(texto: str) -> bool:
    """Detect an explicit future time reference such as "en 45 minutos".

    Separate from the keyword list because it is a pattern, not a word, and
    because the same regexes are reused to read the horizon.

    Args:
        texto: The lowercased question.

    Returns:
        True if the question names a time offset into the future.
    """
    return bool(
        re.search(r"\ben\s+\d+\s*(min|hora)", texto)
        or re.search(r"\bdentro\s+de\s+\d+\s*(min|hora)", texto)
        or any(
            frase in texto
            for frase in (
                "en media hora", "en una hora", "en dos horas",
                "dentro de una hora", "dentro de media hora",
            )
        )
    )


def _extraer_variables(texto: str) -> list[str]:
    """Collect the environmental variables the question mentions.

    Args:
        texto: The lowercased question.

    Returns:
        Variable names in the order config declares them. Empty means the
        question did not single any out, which the orchestrator reads as all.
    """
    return [
        variable
        for variable in cfg.VARIABLES_ACTIVAS
        if _contiene_alguna(texto, SINONIMOS_VARIABLE.get(variable, [variable]))
    ]


def _extraer_espacio(texto: str) -> int | None:
    """Find which space the question refers to.

    Args:
        texto: The lowercased question.

    Returns:
        The space id, or None when the question names no space, in which case
        the caller supplies the one selected in the viewer.
    """
    for espacio_id, sinonimos in ESPACIOS.items():
        if _contiene_alguna(texto, sinonimos):
            return espacio_id
    return None


def _extraer_horizonte(texto: str) -> int:
    """Read the prediction horizon in minutes.

    Values are capped at HORIZONTE_MAX_MIN and rounded up to the sampling
    interval, because the predictor only emits whole 5-minute steps.

    Args:
        texto: The lowercased question.

    Returns:
        Minutes ahead, between the sampling interval and HORIZONTE_MAX_MIN.
    """
    minutos = None

    coincidencia = re.search(r"(\d+)\s*min", texto)
    if coincidencia:
        minutos = int(coincidencia.group(1))
    else:
        coincidencia = re.search(r"(\d+)\s*hora", texto)
        if coincidencia:
            minutos = int(coincidencia.group(1)) * 60

    if minutos is None:
        for frase, valor in (
            ("dos horas", 120), ("2 horas", 120),
            ("hora y media", 90), ("1.5 horas", 90),
            ("una hora", 60), ("1 hora", 60),
            ("media hora", 30),
        ):
            if frase in texto:
                minutos = valor
                break

    if minutos is None:
        return HORIZONTE_POR_OMISION

    paso = cfg.INTERVALO_SEGUNDOS // 60
    minutos = max(paso, min(minutos, HORIZONTE_MAX_MIN))
    # Round up to a whole step: asking for 7 minutes gets 10, never 5, so the
    # answer always covers at least what was asked.
    return int(-(-minutos // paso) * paso)


def _contiene_alguna(texto: str, palabras: list[str]) -> bool:
    """Test whether any of the given phrases appears in the text.

    Args:
        texto: The lowercased question.
        palabras: Phrases to look for.

    Returns:
        True if at least one phrase is present.
    """
    return any(palabra in texto for palabra in palabras)


def main() -> int:
    """Classify a set of example questions and print the routing table.

    Returns:
        Process exit code.
    """
    ejemplos = [
        "¿Cómo está el CO2 del salón 3302?",
        "¿Cuál es la temperatura actual del aula 2?",
        "¿Cómo va a estar el CO2 del 3302 en 45 minutos?",
        "¿La humedad va a subir en dos horas?",
        "¿Cuál fue el pico de CO2 ayer en el 3201?",
        "¿Hay algo raro en el pasillo?",
        "¿El CO2 está demasiado alto?",
        "¿Y el 3105?",
        "¿El CO2 va a estar muy alto en una hora?",
    ]

    print(f"{'pregunta':52} {'tipo':11} {'esp':>4} {'min':>4}  variables")
    print("-" * 100)
    for pregunta in ejemplos:
        i = clasificar(pregunta)
        espacio = i.espacio_id if i.espacio_id is not None else "-"
        variables = ", ".join(i.variables) if i.variables else "(todas)"
        print(
            f"{pregunta[:52]:52} {i.tipo:11} {str(espacio):>4} "
            f"{i.horizonte_min:>4}  {variables}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
