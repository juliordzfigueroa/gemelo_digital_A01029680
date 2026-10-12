"""Direct and iterative forecasting with an already trained GRU.

entrenar.py answers "how good is this model" against a held-out split. This
module answers the different question the conversational layer asks: given the
last hour of readings for one space, what comes next.

Two modes, picked from the requested horizon:

    direct      One forward pass. The head emits all 6 steps (30 min) at once,
                which is exactly what the network was trained to produce.
    iterative   The predicted block is appended to the context window and the
                pass repeats, up to 2 hours. Error compounds at every
                iteration, so the result carries an explicit warning.

Three things this module does that are easy to get wrong:

  - The space id is not the embedding index. `espacio_id` 3302 is a room
    number; the network expects the row of the embedding table that space was
    assigned during training, which lives in the run metadata.
  - Time features are recomputed for the real future timestamps on every
    iteration. Carrying over the window's own time features would shift the
    model's idea of the time of day by 30 minutes per iteration, and the
    network leans on that signal.
  - Fed-back values are clipped to the instrument range before they re-enter
    the window. Without it an iteration can hand the next one a negative CO2
    and the error grows from something physically impossible.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-12 - Initial implementation.

Reference:
    Sections 4.2.5 and 4.2.6 of the research document.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from comun import config as cfg
from comun.datos import Normalizador, agregar_rasgos_de_tiempo
from gru import modelo as mod

# Longest horizon the iterative mode will produce. Beyond two hours the
# compounded error makes the output not worth reporting.
#
# Reference: Section 4.2.6 of the research document.
HORIZONTE_MAX_MIN = 120

# Minutes per step, derived from the sampling interval rather than hard-coded
# so that changing the interval in config.py cannot desynchronize this module.
MINUTOS_POR_PASO = cfg.INTERVALO_SEGUNDOS // 60


def _implementacion_desde_pesos(estado: dict) -> str:
    """Infer which recurrent implementation a checkpoint was trained with.

    The two implementations do not have the same parameters: the hand-written
    cell stores three weight matrices per layer, while nn.GRU stores fused
    input and hidden matrices plus two bias vectors per gate. Loading a
    checkpoint into the wrong one fails with an unreadable state_dict error,
    and older runs saved no `implementacion` field at all, so the weights
    themselves are the only reliable source.

    Args:
        estado: State dict loaded from a .pt file.

    Returns:
        Either "cho" or "pytorch".

    Raises:
        RuntimeError: If the checkpoint matches neither layout.
    """
    if any(clave.startswith("gru1.celda.") for clave in estado):
        return "cho"
    if "gru1.weight_ih_l0" in estado:
        return "pytorch"
    raise RuntimeError(
        "the checkpoint matches neither recurrent layout; its first keys are "
        f"{list(estado)[:4]}"
    )


def cargar_modelo(
    ruta_pesos: Path | None = None,
    ruta_normalizador: Path | None = None,
    ruta_metadatos: Path | None = None,
    device: str | None = None,
) -> tuple[mod.GRUAmbiental, Normalizador, dict]:
    """Load weights, normalizer and run metadata from disk.

    The three files are a unit. Without the normalizer the network's outputs
    cannot be turned back into ppm or degrees, and without the metadata there
    is no way to know which embedding row belongs to which space.

    Args:
        ruta_pesos: Path to the .pt file. Defaults to models/gru.pt.
        ruta_normalizador: Path to the normalizer JSON.
        ruta_metadatos: Path to the metadata JSON.
        device: "cuda", "cpu" or None to autodetect.

    Returns:
        A tuple of (network in eval mode, fitted normalizer, metadata dict).

    Raises:
        FileNotFoundError: If any of the three files is missing.
    """
    ruta_pesos = ruta_pesos or cfg.MODELOS / "gru.pt"
    ruta_normalizador = ruta_normalizador or cfg.MODELOS / "gru_normalizador.json"
    ruta_metadatos = ruta_metadatos or cfg.MODELOS / "gru_metadatos.json"

    for ruta in (ruta_pesos, ruta_normalizador, ruta_metadatos):
        if not Path(ruta).exists():
            raise FileNotFoundError(
                f"{ruta} does not exist; train the model first with "
                "`python -m gru.entrenar`"
            )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    metadatos = json.loads(Path(ruta_metadatos).read_text(encoding="utf-8"))
    estado = torch.load(ruta_pesos, map_location=device)

    # Derived from the weights even when the metadata states it: the weights
    # are what has to load, and a metadata field can be stale or absent.
    implementacion = _implementacion_desde_pesos(estado)

    red = mod.construir(
        n_espacios=len(metadatos["indice_espacios"]),
        n_rasgos_continuos=len(metadatos["columnas_rasgos"]),
        implementacion=implementacion,
    )
    red.load_state_dict(estado)
    red.to(device)
    red.eval()

    normalizador = Normalizador.cargar(Path(ruta_normalizador))
    return red, normalizador, metadatos


def indice_de_espacio(metadatos: dict, espacio_id: int) -> int:
    """Translate a space id into its row of the embedding table.

    Args:
        metadatos: Metadata dict returned by cargar_modelo().
        espacio_id: Space identifier as it appears in the data.

    Returns:
        The embedding index the network expects.

    Raises:
        ValueError: If that space was not part of the training run, in which
            case the network has no learned representation for it.
    """
    indices = metadatos["indice_espacios"]
    clave = str(espacio_id)
    if clave not in indices:
        raise ValueError(
            f"space {espacio_id} was not in the training run; the model only "
            f"knows {sorted(int(k) for k in indices)}"
        )
    return int(indices[clave])


def _verificar_disposicion(metadatos: dict) -> list[str]:
    """Confirm the feature columns are the variables followed by time features.

    The iterative mode rebuilds input rows by concatenating predicted
    variables and freshly computed time features, which is only correct if the
    columns sit in that order. Checking it here turns a silent corruption into
    an error at load time.

    Args:
        metadatos: Metadata dict returned by cargar_modelo().

    Returns:
        The names of the trailing time-feature columns, possibly empty.

    Raises:
        RuntimeError: If the variables are not the leading columns.
    """
    columnas = metadatos["columnas_rasgos"]
    variables = metadatos["variables_activas"]
    if columnas[: len(variables)] != variables:
        raise RuntimeError(
            "the feature layout is not 'variables first, time features after', "
            f"so iterative prediction cannot rebuild a row: {columnas}"
        )
    return columnas[len(variables) :]


def _rasgos_de_tiempo(marcas: pd.Series, nombres: list[str]) -> np.ndarray:
    """Build the cyclical time features for a series of timestamps.

    Delegates to the same function the training pipeline uses, so inference can
    never drift from training on how the time of day is encoded.

    Args:
        marcas: Timestamps to encode.
        nombres: Expected feature names, in order.

    Returns:
        An array shaped (len(marcas), len(nombres)).
    """
    if not nombres:
        return np.empty((len(marcas), 0), dtype=np.float32)
    marco = pd.DataFrame({cfg.COLUMNA_TIEMPO: marcas.reset_index(drop=True)})
    agregar_rasgos_de_tiempo(marco)
    return marco[nombres].to_numpy(dtype=np.float32)


def _a_fisico(
    normalizador: Normalizador,
    bloque: np.ndarray,
    variables: list[str],
    espacio_id: int,
) -> np.ndarray:
    """Convert a normalized block to physical units and clip it.

    Args:
        normalizador: Fitted normalizer.
        bloque: Normalized predictions shaped (pasos, n_variables).
        variables: Variable names matching the last axis.
        espacio_id: Space the block belongs to.

    Returns:
        An array of the same shape in physical units, within the instrument
        range declared in config.LIMITES_FISICOS.
    """
    espacios = np.array([espacio_id]) if normalizador.por_espacio else None
    fisico = normalizador.desnormalizar(bloque[None, :, :], variables, espacios)[0]

    for i, variable in enumerate(variables):
        minimo, maximo = cfg.LIMITES_FISICOS.get(variable, (None, None))
        if minimo is not None:
            fisico[:, i] = np.maximum(fisico[:, i], minimo)
        if maximo is not None:
            fisico[:, i] = np.minimum(fisico[:, i], maximo)
    return fisico


def _a_normalizado(
    normalizador: Normalizador,
    bloque: np.ndarray,
    variables: list[str],
    espacio_id: int,
) -> np.ndarray:
    """Convert a physical block back to normalized space.

    Used only by the iterative mode, to feed clipped values back into the
    context window. It goes through the normalizer's public transform instead
    of repeating the affine arithmetic, so the two can never disagree.

    Args:
        normalizador: Fitted normalizer.
        bloque: Values in physical units shaped (pasos, n_variables).
        variables: Variable names matching the last axis.
        espacio_id: Space the block belongs to.

    Returns:
        An array of the same shape in normalized space.
    """
    marco = pd.DataFrame(bloque, columns=variables)
    marco[cfg.COLUMNA_ESPACIO] = espacio_id
    return normalizador.transformar(marco)[variables].to_numpy(dtype=np.float32)


def _ventana_normalizada(
    ventana: pd.DataFrame,
    normalizador: Normalizador,
    metadatos: dict,
    nombres_tiempo: list[str],
) -> np.ndarray:
    """Turn a window of raw readings into the network's input array.

    Args:
        ventana: Exactly PASOS_ENTRADA rows satisfying the data contract,
            sorted by time, in physical units.
        normalizador: Fitted normalizer.
        metadatos: Metadata dict returned by cargar_modelo().
        nombres_tiempo: Trailing time-feature column names.

    Returns:
        An array shaped (PASOS_ENTRADA, n_rasgos) in normalized space.

    Raises:
        ValueError: If the window has the wrong length, is missing a contract
            column, or holds a missing value.
    """
    pasos_entrada = metadatos["pasos_entrada"]
    variables = metadatos["variables_activas"]

    if len(ventana) != pasos_entrada:
        raise ValueError(
            f"the window needs exactly {pasos_entrada} rows, got {len(ventana)}"
        )

    faltan = {cfg.COLUMNA_TIEMPO, *variables} - set(ventana.columns)
    if faltan:
        raise ValueError(f"the window is missing columns: {sorted(faltan)}")

    if ventana[variables].isna().to_numpy().any():
        raise ValueError(
            "the window holds missing values; the network cannot consume a "
            "window with gaps, so pick a contiguous one"
        )

    marco = ventana.sort_values(cfg.COLUMNA_TIEMPO).reset_index(drop=True).copy()
    if cfg.COLUMNA_ESPACIO not in marco.columns:
        marco[cfg.COLUMNA_ESPACIO] = 0

    normalizado = normalizador.transformar(marco)
    rasgos = normalizado[variables].to_numpy(dtype=np.float32)

    if nombres_tiempo:
        tiempo = _rasgos_de_tiempo(marco[cfg.COLUMNA_TIEMPO], nombres_tiempo)
        rasgos = np.concatenate([rasgos, tiempo], axis=1)
    return rasgos


def ultima_ventana_contigua(filas: pd.DataFrame, pasos: int) -> pd.DataFrame | None:
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


def predecir(
    red: mod.GRUAmbiental,
    normalizador: Normalizador,
    metadatos: dict,
    ventana: pd.DataFrame,
    espacio_id: int,
    horizonte_min: int = 30,
    device: str | None = None,
) -> dict:
    """Forecast one space forward, choosing direct or iterative mode.

    Args:
        red: Network returned by cargar_modelo().
        normalizador: Normalizer returned by cargar_modelo().
        metadatos: Metadata dict returned by cargar_modelo().
        ventana: The last PASOS_ENTRADA readings of that space, in physical
            units, with the timestamp column. The timestamps are what let the
            iterative mode know the real time of day of each predicted step.
        espacio_id: Space identifier as it appears in the data.
        horizonte_min: Minutes ahead, a multiple of 5 up to HORIZONTE_MAX_MIN.
        device: "cuda", "cpu" or None to autodetect.

    Returns:
        A dict with:
          predicciones: one dict per 5-minute step, holding offset_min and one
            entry per variable in physical units.
          horizonte_min: the horizon actually served.
          metodo: "directo" or "iterativo".
          advertencia: a sentence about compounded error, or None.

    Raises:
        ValueError: If the horizon is out of range or not a multiple of the
            sampling interval, or if the window is malformed.
    """
    if horizonte_min <= 0 or horizonte_min > HORIZONTE_MAX_MIN:
        raise ValueError(
            f"horizonte_min must be between {MINUTOS_POR_PASO} and "
            f"{HORIZONTE_MAX_MIN}, got {horizonte_min}"
        )
    if horizonte_min % MINUTOS_POR_PASO != 0:
        raise ValueError(
            f"horizonte_min must be a multiple of {MINUTOS_POR_PASO}, "
            f"got {horizonte_min}"
        )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    variables = metadatos["variables_activas"]
    pasos_salida = metadatos["pasos_salida"]
    nombres_tiempo = _verificar_disposicion(metadatos)
    indice = indice_de_espacio(metadatos, espacio_id)

    rasgos = _ventana_normalizada(ventana, normalizador, metadatos, nombres_tiempo)
    ultima_marca = pd.to_datetime(ventana[cfg.COLUMNA_TIEMPO]).max()

    pasos_pedidos = horizonte_min // MINUTOS_POR_PASO
    minutos_por_bloque = pasos_salida * MINUTOS_POR_PASO
    n_iteraciones = math.ceil(horizonte_min / minutos_por_bloque)

    # Every timestamp the run will predict, computed once so each iteration
    # encodes the real clock time of its own steps.
    paso = pd.Timedelta(seconds=cfg.INTERVALO_SEGUNDOS)
    marcas = pd.Series(
        [ultima_marca + paso * (i + 1) for i in range(n_iteraciones * pasos_salida)]
    )
    tiempo_futuro = _rasgos_de_tiempo(marcas, nombres_tiempo)

    espacio_t = torch.tensor([indice], dtype=torch.long, device=device)
    acumulado: list[np.ndarray] = []

    for iteracion in range(n_iteraciones):
        entrada = torch.from_numpy(rasgos[None, :, :]).to(device)
        with torch.inference_mode():
            salida = red(entrada, espacio_t)

        bloque_norm = salida[0].cpu().numpy()
        bloque_fisico = _a_fisico(normalizador, bloque_norm, variables, espacio_id)
        acumulado.append(bloque_fisico)

        if iteracion + 1 == n_iteraciones:
            break

        # Slide the window: drop the oldest block, append what was just
        # predicted. The clipped physical values go back, not the raw network
        # output, so an impossible value never becomes the next input.
        realimenta = _a_normalizado(
            normalizador, bloque_fisico, variables, espacio_id
        )
        desde = iteracion * pasos_salida
        nuevas = np.concatenate(
            [realimenta, tiempo_futuro[desde : desde + pasos_salida]], axis=1
        )
        rasgos = np.vstack([rasgos[pasos_salida:], nuevas]).astype(np.float32)

    prediccion = np.vstack(acumulado)[:pasos_pedidos]

    pasos = [
        {
            "offset_min": (i + 1) * MINUTOS_POR_PASO,
            **{
                variable: round(float(prediccion[i, j]), 2)
                for j, variable in enumerate(variables)
            },
        }
        for i in range(len(prediccion))
    ]

    iterativo = n_iteraciones > 1
    advertencia = None
    if iterativo:
        advertencia = (
            f"Predicción iterativa en {n_iteraciones} pasos de "
            f"{minutos_por_bloque} min: el error se acumula en cada iteración, "
            "porque las entradas después del primer bloque son salidas del "
            "propio modelo y no lecturas reales."
        )

    return {
        "predicciones": pasos,
        "horizonte_min": horizonte_min,
        "metodo": "iterativo" if iterativo else "directo",
        "advertencia": advertencia,
    }


def main() -> int:
    """Forecast one space from the cleaned sensor data and print the result.

    Exists so the module can be exercised without the API on top, which is how
    a broken checkpoint or a stale metadata file gets caught early.

    Returns:
        Process exit code.
    """
    import argparse

    from comun import datos as dat

    parser = argparse.ArgumentParser(description="Forecast one space.")
    parser.add_argument("--espacio", type=int, default=308)
    parser.add_argument("--horizonte", type=int, default=30)
    parser.add_argument(
        "--datos", default=None, help="contract frame; defaults to the cleaned sensor data"
    )
    args = parser.parse_args()

    red, normalizador, metadatos = cargar_modelo()
    print(
        f"Modelo cargado: implementacion {red.implementacion}, "
        f"{red.n_parametros():,} parametros, "
        f"espacios {sorted(int(k) for k in metadatos['indice_espacios'])}"
    )

    df = dat.cargar_datos(args.datos)
    propio = df[df[cfg.COLUMNA_ESPACIO] == args.espacio].sort_values(
        cfg.COLUMNA_TIEMPO
    ).reset_index(drop=True)
    ventana = ultima_ventana_contigua(propio, metadatos["pasos_entrada"])
    if ventana is None:
        print(f"El espacio {args.espacio} no tiene una hora de lecturas consecutivas.")
        return 1
    print(f"Ventana: {ventana[cfg.COLUMNA_TIEMPO].iloc[0]} a {ventana[cfg.COLUMNA_TIEMPO].iloc[-1]}")

    salida = predecir(
        red,
        normalizador,
        metadatos,
        ventana,
        espacio_id=args.espacio,
        horizonte_min=args.horizonte,
    )

    print(
        f"\nEspacio {args.espacio}, horizonte {salida['horizonte_min']} min, "
        f"metodo {salida['metodo']}"
    )
    ultima = ventana.iloc[-1]
    print(
        "  ahora      "
        + "  ".join(
            f"{v}={float(ultima[v]):.1f}" for v in metadatos["variables_activas"]
        )
    )
    for p in salida["predicciones"]:
        detalle = "  ".join(
            f"{v}={p[v]:.1f}" for v in metadatos["variables_activas"]
        )
        print(f"  +{p['offset_min']:>3} min  {detalle}")

    if salida["advertencia"]:
        print(f"\n  {salida['advertencia']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
