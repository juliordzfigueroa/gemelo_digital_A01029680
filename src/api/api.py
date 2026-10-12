"""FastAPI orchestrator for the conversational layer.

The single entry point the viewer talks to. One request runs the whole chain:

    classify the question  ->  retrieve the data it needs  ->  build the prompt
    ->  ask the local LLM  ->  return the sentence

Every number in the answer comes from this process, never from the model. That
is the whole point of the retrieval-augmented pattern of section 4.3: the LLM
is a writer, not a calculator, and it is given no opportunity to be anything
else.

What is wired and what is not. The GRU is real: the forecast comes from the
trained checkpoint in models/. The data source is not: the three retrieval
functions read an in-memory synthetic frame generated at startup, because the
Aulas 3 sensors are still being installed and the Tec database is not
reachable yet. They are marked `TODO E5` and are the only places that change
when it is. The anomaly detector does not exist at all, and this module says
so rather than reporting that nothing was found, which would be a different
claim entirely.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-12 - Initial implementation.

Reference:
    Sections 4.3.1 to 4.3.4 of the research document.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from comun import config as cfg
from comun import sintetico
from gru import predecir as pred
from llm import clasificador, cliente_llm, contexto

# Days of synthetic history kept in memory as the stand-in for the database.
# Enough to answer a "yesterday" question and to leave contiguous windows.
DIAS_SINTETICOS = 7
ESPACIOS_SINTETICOS = 5

# Default span for a historical question when the user names no period.
HORAS_HISTORICAS = 24

# Process-wide state, populated once at startup. Reloading the checkpoint per
# request would add seconds of latency to every question for no benefit.
ESTADO: dict[str, Any] = {
    "red": None,
    "normalizador": None,
    "metadatos": None,
    "datos": None,
    "error_modelo": None,
}


# --------------------------------------------------------------------------
# Request and response shapes
# --------------------------------------------------------------------------
class Pregunta(BaseModel):
    """Body of a /chat or /contexto request."""

    pregunta: str = Field(min_length=1, max_length=500)
    espacio_id: int | None = Field(
        default=None,
        description=(
            "Space selected in the viewer. Used when the question names none. "
            "A space named in the text wins over this value."
        ),
    )


class Respuesta(BaseModel):
    """Body of a /chat response."""

    respuesta: str
    tipo_intencion: str
    espacio: int | None
    horizonte_min: int | None
    metodo_prediccion: str | None
    advertencia: str | None


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------
@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    """Load the model and the stand-in data once, before serving.

    A missing checkpoint is not fatal: the service still starts so that
    /health can report what is wrong instead of refusing to boot, which is
    harder to diagnose.

    Args:
        app: The application being started.

    Yields:
        Control back to FastAPI while the app serves requests.
    """
    try:
        red, normalizador, metadatos = pred.cargar_modelo()
        ESTADO.update(red=red, normalizador=normalizador, metadatos=metadatos)
        print(
            f"GRU cargado: implementacion {red.implementacion}, "
            f"{red.n_parametros():,} parametros"
        )
    except (FileNotFoundError, RuntimeError) as e:
        ESTADO["error_modelo"] = str(e)
        print(f"ADVERTENCIA: el GRU no se pudo cargar: {e}")

    # TODO E5: replace with a connection to the Tec database. Everything
    # downstream reads this frame through the three retrieval functions, so
    # only they change.
    ESTADO["datos"] = sintetico.genera(
        dias=DIAS_SINTETICOS, n_espacios=ESPACIOS_SINTETICOS
    )
    print(
        f"Datos sinteticos en memoria: {len(ESTADO['datos']):,} filas, "
        f"{ESPACIOS_SINTETICOS} espacios, {DIAS_SINTETICOS} dias"
    )
    yield


app = FastAPI(
    title="Gemelo Digital Aulas 3",
    description="Capa conversacional: GRU + Llama 3.1 8B vía Ollama.",
    version="0.1.0",
    lifespan=ciclo_de_vida,
)


# --------------------------------------------------------------------------
# Data retrieval
# --------------------------------------------------------------------------
def _filas_de_espacio(espacio_id: int) -> pd.DataFrame:
    """Return every row of one space, sorted by time.

    TODO E5: replace with a query to the Tec database.

    Args:
        espacio_id: Space identifier.

    Returns:
        The rows for that space, possibly empty.
    """
    datos = ESTADO["datos"]
    propias = datos[datos[cfg.COLUMNA_ESPACIO] == espacio_id]
    return propias.sort_values(cfg.COLUMNA_TIEMPO).reset_index(drop=True)


def _ultima_ventana_contigua(filas: pd.DataFrame, pasos: int) -> pd.DataFrame | None:
    """Find the most recent run of `pasos` consecutive readings.

    Taking the last N rows is not enough. Readings arrive as missing rows when
    a node loses the network, so the last 12 rows can span a two-day hole and
    still look like a valid window. The network would consume it without
    complaint and forecast from a sequence that never happened.

    Args:
        filas: Rows of one space, sorted by time.
        pasos: How many consecutive readings are needed.

    Returns:
        The last window whose steps are exactly one sampling interval apart,
        or None if the history holds no such run.
    """
    if len(filas) < pasos:
        return None

    paso = pd.Timedelta(seconds=cfg.INTERVALO_SEGUNDOS)
    marcas = pd.to_datetime(filas[cfg.COLUMNA_TIEMPO])
    contiguo = marcas.diff() == paso

    # Walk backwards: the first window that is fully contiguous and complete
    # is the freshest usable one.
    for fin in range(len(filas), pasos - 1, -1):
        inicio = fin - pasos
        ventana = filas.iloc[inicio:fin]
        if not contiguo.iloc[inicio + 1 : fin].all():
            continue
        if ventana[cfg.VARIABLES_ACTIVAS].isna().to_numpy().any():
            continue
        return ventana
    return None


def _datos_actuales(espacio_id: int) -> dict:
    """Retrieve the latest reading of one space.

    TODO E5: replace with a query to `lecturas_agregadas` for the most recent
    row of that space.

    Args:
        espacio_id: Space identifier.

    Returns:
        A dict shaped for contexto._prompt_actual.
    """
    filas = _filas_de_espacio(espacio_id)
    if filas.empty:
        return {"espacio": espacio_id, "lecturas": {}, "timestamp": None}

    ultima = filas.iloc[-1]
    return {
        "espacio": espacio_id,
        "timestamp": str(ultima[cfg.COLUMNA_TIEMPO]),
        "lecturas": {
            variable: _numero(ultima[variable])
            for variable in cfg.VARIABLES_ACTIVAS
        },
    }


def _datos_historicos(
    espacio_id: int, variable: str, horas: int = HORAS_HISTORICAS
) -> dict:
    """Summarize one variable of one space over a past window.

    TODO E5: replace with a ranged query to `lecturas_agregadas`.

    Args:
        espacio_id: Space identifier.
        variable: Which variable to summarize.
        horas: How far back to look.

    Returns:
        A dict shaped for contexto._prompt_historica.
    """
    filas = _filas_de_espacio(espacio_id)
    if filas.empty:
        return {"espacio": espacio_id, "variable": variable, "resumen": {}}

    marcas = pd.to_datetime(filas[cfg.COLUMNA_TIEMPO])
    corte = marcas.max() - pd.Timedelta(hours=horas)
    recientes = filas[marcas >= corte]
    serie = recientes[variable].dropna()

    return {
        "espacio": espacio_id,
        "variable": variable,
        "desde": str(corte),
        "hasta": str(marcas.max()),
        "resumen": {
            "maximo": _numero(serie.max()) if len(serie) else None,
            "minimo": _numero(serie.min()) if len(serie) else None,
            "promedio": _numero(serie.mean()) if len(serie) else None,
            "n_lecturas": int(len(serie)),
        },
    }


def _datos_anomalia(espacio_id: int) -> dict:
    """Report what the anomaly detector found.

    TODO E5: replace with a call to src/anomalias/detector.py once it exists.

    Until then this returns "no_disponible" rather than an empty list of
    anomalies. The distinction matters: an empty list means the detector ran
    and found nothing, and claiming that while no detector exists would put a
    false all-clear in front of the user.

    Args:
        espacio_id: Space identifier.

    Returns:
        A dict shaped for contexto._prompt_anomalia.
    """
    return {
        "espacio": espacio_id,
        "estado": "no_disponible",
        "anomalias": [],
        "detalle": (
            "El detector 3σ (src/anomalias/) todavía no está implementado, "
            "así que no se ha evaluado si hay anomalías."
        ),
    }


def _datos_predictivos(intencion: clasificador.Intencion, espacio_id: int) -> dict:
    """Run the GRU forward for one space.

    Args:
        intencion: The classified question, which carries the horizon.
        espacio_id: Space identifier.

    Returns:
        A dict shaped for contexto._prompt_predictiva. When the model or a
        usable window is unavailable it carries an "error" key instead of
        predictions, so the prompt can say so plainly.
    """
    if ESTADO["red"] is None:
        return {
            "espacio": espacio_id,
            "lectura_actual": {},
            "predicciones": [],
            "error": (
                "El modelo GRU no está cargado: "
                f"{ESTADO['error_modelo']}"
            ),
        }

    metadatos = ESTADO["metadatos"]
    filas = _filas_de_espacio(espacio_id)
    ventana = _ultima_ventana_contigua(filas, metadatos["pasos_entrada"])

    if ventana is None:
        return {
            "espacio": espacio_id,
            "lectura_actual": {},
            "predicciones": [],
            "error": (
                f"No hay {metadatos['pasos_entrada']} lecturas consecutivas "
                f"para el espacio {espacio_id}, así que no se puede predecir."
            ),
        }

    try:
        salida = pred.predecir(
            ESTADO["red"],
            ESTADO["normalizador"],
            metadatos,
            ventana,
            espacio_id=espacio_id,
            horizonte_min=intencion.horizonte_min,
        )
    except ValueError as e:
        return {
            "espacio": espacio_id,
            "lectura_actual": {},
            "predicciones": [],
            "error": str(e),
        }

    ultima = ventana.iloc[-1]
    return {
        "espacio": espacio_id,
        "lectura_actual": {
            variable: _numero(ultima[variable])
            for variable in metadatos["variables_activas"]
        },
        "timestamp": str(ultima[cfg.COLUMNA_TIEMPO]),
        **salida,
    }


def _numero(valor: object) -> float | None:
    """Coerce a pandas scalar to a plain float for JSON.

    Args:
        valor: Value taken from a DataFrame.

    Returns:
        The value as a rounded float, or None when it is missing.
    """
    if valor is None or pd.isna(valor):
        return None
    return round(float(valor), 2)


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def _recuperar(intencion: clasificador.Intencion, espacio_id: int) -> dict:
    """Fetch whatever the classified intent needs.

    Args:
        intencion: Result of clasificador.clasificar().
        espacio_id: Space to answer about.

    Returns:
        The data dict for contexto.construir_prompt().
    """
    if intencion.tipo == "predictiva":
        return _datos_predictivos(intencion, espacio_id)
    if intencion.tipo == "historica":
        variable = intencion.variables[0] if intencion.variables else "co2"
        return _datos_historicos(espacio_id, variable)
    if intencion.tipo == "anomalia":
        return _datos_anomalia(espacio_id)
    return _datos_actuales(espacio_id)


def _espacio_elegido(intencion: clasificador.Intencion, pedido: int | None) -> int:
    """Decide which space the answer is about.

    A space named in the question wins over the one selected in the viewer,
    because the text is the more explicit signal. With neither, the first
    space the model knows is used so the request still produces an answer.

    Args:
        intencion: The classified question.
        pedido: Space id sent in the request body, if any.

    Returns:
        The space id to answer about.
    """
    if intencion.espacio_id is not None:
        return intencion.espacio_id
    if pedido is not None:
        return pedido
    metadatos = ESTADO["metadatos"]
    if metadatos:
        return min(int(k) for k in metadatos["indice_espacios"])
    return 1


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
@app.get("/health")
def health() -> dict:
    """Report whether the service can answer questions.

    Returns:
        A dict naming the state of the model, the data and Ollama. Never
        raises: a health check that fails to respond is useless.
    """
    listo, motivo = cliente_llm.disponible()
    datos = ESTADO["datos"]
    return {
        "status": "ok",
        "gru_cargado": ESTADO["red"] is not None,
        "error_modelo": ESTADO["error_modelo"],
        "ollama_requerido": True,
        "ollama_listo": listo,
        "ollama_detalle": motivo,
        "filas_en_memoria": 0 if datos is None else int(len(datos)),
        "fuente_de_datos": "sintetica",
        "detector_anomalias": "no_implementado",
    }


@app.post("/contexto")
def contexto_de(pregunta: Pregunta) -> dict:
    """Run the chain up to the prompt, without calling the LLM.

    Exists so the retrieval half can be tested and demonstrated when Ollama is
    not installed, and so a wrong answer can be traced to either the data or
    the model instead of guessing between them.

    Args:
        pregunta: The request body.

    Returns:
        The classified intent, the retrieved data and both prompts.
    """
    intencion = clasificador.clasificar(pregunta.pregunta)
    espacio = _espacio_elegido(intencion, pregunta.espacio_id)
    datos = _recuperar(intencion, espacio)
    sistema, usuario = contexto.construir_prompt(intencion, datos)

    return {
        "intencion": {
            "tipo": intencion.tipo,
            "variables": intencion.variables,
            "espacio_id": espacio,
            "horizonte_min": intencion.horizonte_min,
        },
        "datos": datos,
        "prompt_sistema": sistema,
        "prompt_usuario": usuario,
    }


@app.post("/chat", response_model=Respuesta)
def chat(pregunta: Pregunta) -> Respuesta:
    """Answer one question about the building.

    Args:
        pregunta: The request body.

    Returns:
        The generated sentence plus the routing decisions that produced it, so
        the caller can show what the answer was based on.
    """
    intencion = clasificador.clasificar(pregunta.pregunta)
    espacio = _espacio_elegido(intencion, pregunta.espacio_id)
    datos = _recuperar(intencion, espacio)
    sistema, usuario = contexto.construir_prompt(intencion, datos)

    try:
        respuesta = cliente_llm.generar_respuesta(sistema, usuario)
    except RuntimeError as e:
        # Returned as a normal body rather than a 5xx: the chain did its work
        # and the caller is better served by a sentence explaining what is
        # missing than by an error the viewer has to interpret.
        respuesta = f"No puedo responder ahora mismo. {e}"

    return Respuesta(
        respuesta=respuesta,
        tipo_intencion=intencion.tipo,
        espacio=espacio,
        horizonte_min=(
            datos.get("horizonte_min") if intencion.tipo == "predictiva" else None
        ),
        metodo_prediccion=(
            datos.get("metodo") if intencion.tipo == "predictiva" else None
        ),
        advertencia=datos.get("advertencia") or datos.get("error") or datos.get("detalle"),
    )
