"""Configuration for the GRU prediction model.

Holds the data contract, the physical limits of the sensors and the network
hyperparameters. By design this file contains no means, no standard deviations
and no anomaly thresholds: every value derived from data is computed from the
training split and stored next to the trained model, so that the same code
produces Aulas 3 parameters once the real sensors are deployed.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-01 - Added IMPLEMENTACION_GRU to switch recurrent implementations.

Reference:
    Section 4.2.6 of the research document.
    Sabiri, Y., Houmaidi, W., Bougrine, A., & El Mansour Billah, S. (2025).
    Optimizing indoor environmental quality in smart buildings using deep
    learning. arXiv:2509.26187.
"""

from pathlib import Path

RAIZ = Path(__file__).resolve().parents[2]
DATOS_CRUDOS = RAIZ / "data" / "raw"
DATOS_PROCESADOS = RAIZ / "data" / "processed"
MODELOS = RAIZ / "models"

SEMILLA = 42

# --- Data contract ---------------------------------------------------------
# Every data source (synthetic now, Aulas 3 sensors later) must provide a
# DataFrame with these columns. Nothing downstream knows where the rows
# came from, which is what lets the pipeline survive the switch.
COLUMNA_TIEMPO = "timestamp"
COLUMNA_ESPACIO = "espacio_id"

VARIABLES = ["co2", "temperatura", "humedad", "luz", "movimiento"]

# Variables the model reads and predicts. Narrowing this list to
# ["co2", "temperatura", "humedad"] reproduces the Sabiri (2025) setup.
VARIABLES_ACTIVAS = ["co2", "temperatura", "humedad", "luz", "movimiento"]

# Physical units, used only to print metrics in readable form.
UNIDADES = {
    "co2": "ppm",
    "temperatura": "C",
    "humedad": "%RH",
    "luz": "lux",
    "movimiento": "0/1",
}

# --- Physical limits of the instruments ------------------------------------
# Hardware bounds, not statistics. Used to clip anomaly thresholds and to flag
# physically impossible readings.
LIMITES_FISICOS = {
    "co2": (400.0, 5000.0),  # 400 is the NDIR calibration floor, not a reading
    "temperatura": (-10.0, 60.0),
    "humedad": (0.0, 100.0),
    "luz": (0.0, None),
    "movimiento": (0.0, 1.0),
}

# Binary variables. This matters in three places: they are not z-scored, MAPE
# does not apply to them, and the 3-sigma detector does not either because the
# band falls entirely outside [0, 1].
#
# Reference: docs/hallazgos_robod.md
BINARIAS = {"movimiento"}

# Variables whose minimum is a calibration floor rather than a real
# measurement. The lower tail is censored, so a -3 sigma threshold can never
# fire for them.
CENSURADAS_POR_ABAJO = {"co2": 400.0}

# --- Time windows ----------------------------------------------------------
INTERVALO_SEGUNDOS = 300  # data aggregated to 5-minute buckets
PASOS_ENTRADA = 12  # 1 hour of context
PASOS_SALIDA = 6  # 30 minutes predicted per direct invocation

# Gaps up to this many steps are interpolated; anything longer splits the
# block. Windows never cross a block boundary.
MAX_PASOS_INTERPOLABLES = 3  # 15 minutes

# --- Architecture ----------------------------------------------------------
UNIDADES_OCULTAS = [128, 64]
DROPOUT = 0.2

# Which recurrent implementation to use.
#
#   "cho"     CeldaGRU, equations 1 to 4 of section 4.2.7.2 written literally.
#             Default, so the equations published in the thesis are the ones
#             that produce the reported results. About 15x slower, which on
#             this dataset means minutes rather than seconds.
#   "pytorch" nn.GRU, the fused cuDNN variant. Faster, but it applies the reset
#             gate after the linear transform instead of before, so it is not
#             the formulation the document states.
#
# Reference: see modelo.CeldaGRU for the full comparison.
IMPLEMENTACION_GRU = "cho"

# Width of the space-id embedding. An embedding rather than one-hot because
# Aulas 3 has many rooms across 5 floors: one-hot grows with the number of
# rooms, the embedding does not.
DIM_EMBEDDING_ESPACIO = 8

# Whether to append cyclical time features to the input. Sine and cosine of the
# time of day avoid the artificial jump from 23:55 to 00:00 that an integer
# hour would introduce.
USAR_RASGOS_DE_TIEMPO = True

# --- Training --------------------------------------------------------------
TASA_APRENDIZAJE = 1e-3  # Adam, standard value for recurrent networks
TAMANO_LOTE = 32
EPOCAS_MAXIMAS = 200
PACIENCIA = 15  # epochs without validation improvement before stopping
MIN_MEJORA = 1e-5  # smallest validation gain that counts as an improvement

# Chronological split, not random. A random split would leave step t in train
# and step t+1 in test, so the model would be scored against data nearly
# identical to what it saw: the result looks excellent and means nothing.
FRACCION_ENTRENAMIENTO = 0.70
FRACCION_VALIDACION = 0.15
# The remaining 0.15 is the test split.
