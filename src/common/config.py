"""Configuration for the GRU prediction model.

Holds the data contract, the physical limits of the sensors and the network
hyperparameters. By design this file contains no means, no standard deviations
and no anomaly thresholds: every value derived from data is computed from the
training split and stored next to the trained model, so that the same code
produces Aulas 3 parameters once the real sensors are deployed.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-01 - Added GRU_IMPLEMENTATION to switch recurrent implementations.

Reference:
    Section 4.2.6 of the research document.
    Sabiri, Y., Houmaidi, W., Bougrine, A., & El Mansour Billah, S. (2025).
    Optimizing indoor environmental quality in smart buildings using deep
    learning. arXiv:2509.26187.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DATA = ROOT / "data" / "raw"
PROCESSED_DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# Cleaned sensor data written by ingestion.cleaning. Every entry point reads this
# unless told otherwise.
REAL_DATA = PROCESSED_DATA / "aulas3.parquet"

# Outdoor weather for the same period, written by ingestion.weather.
WEATHER_DATA = PROCESSED_DATA / "weather.parquet"

# Campus location, for weather queries and solar position. The sensor
# timestamps are local time, so weather is requested in the same zone.
LATITUDE = 19.3593
LONGITUDE = -99.2588
TIMEZONE = "America/Mexico_City"

SEED = 42

# --- Data contract ---------------------------------------------------------
# Every data source (synthetic now, Aulas 3 sensors later) must provide a
# DataFrame with these columns. Nothing downstream knows where the rows
# came from, which is what lets the pipeline survive the switch.
TIME_COLUMN = "timestamp"
SPACE_COLUMN = "space_id"

VARIABLES = ["co2", "temperature", "humidity", "light", "motion"]

# Variables the model reads and predicts. Narrowing this list to
# ["co2", "temperatura", "humedad"] reproduces the Sabiri (2025) setup.
ACTIVE_VARIABLES = ["co2", "temperature", "humidity", "light", "motion"]

# Physical units, used only to print metrics in readable form.
UNITS = {
    "co2": "ppm",
    "temperature": "C",
    "humidity": "%RH",
    "light": "%",
    "motion": "0/1",
}

# --- Physical limits of the instruments ------------------------------------
# Hardware bounds, not statistics. Used to clip anomaly thresholds and to flag
# physically impossible readings.
PHYSICAL_LIMITS = {
    "co2": (400.0, 5000.0),  # 400 is the NDIR calibration floor, not a reading
    "temperature": (-10.0, 60.0),
    "humidity": (0.0, 100.0),
    "light": (0.0, 100.0),  # the nodes report light as a percentage
    "motion": (0.0, 1.0),
}

# Binary variables. This matters in three places: they are not z-scored, MAPE
# does not apply to them, and the 3-sigma detector does not either because the
# band falls entirely outside [0, 1].
#
# Reference: docs/hallazgos_robod.md
BINARY = {"motion"}

# Variables whose minimum is a calibration floor rather than a real
# measurement. The lower tail is censored, so a -3 sigma threshold can never
# fire for them.
LOWER_CENSORED = {"co2": 400.0}

# --- Time windows ----------------------------------------------------------
INTERVAL_SECONDS = 300  # data aggregated to 5-minute buckets
INPUT_STEPS = 12  # 1 hour of context
OUTPUT_STEPS = 6  # 30 minutes predicted per direct invocation

# Gaps up to this many steps are interpolated; anything longer splits the
# block. Windows never cross a block boundary.
MAX_INTERPOLABLE_STEPS = 3  # 15 minutes

# --- Architecture ----------------------------------------------------------
HIDDEN_UNITS = [128, 64]
DROPOUT = 0.2

# Which recurrent implementation to use.
#
#   "cho"     GRUCell, equations 1 to 4 of section 4.2.7.2 written literally.
#             Default, so the equations published in the thesis are the ones
#             that produce the reported results. About 15x slower, which on
#             this dataset means minutes rather than seconds.
#   "pytorch" nn.GRU, the fused cuDNN variant. Faster, but it applies the reset
#             gate after the linear transform instead of before, so it is not
#             the formulation the document states.
#
# Reference: see model.GRUCell for the full comparison.
GRU_IMPLEMENTATION = "cho"

# Width of the space-id embedding. An embedding rather than one-hot because
# Aulas 3 has many rooms across 5 floors: one-hot grows with the number of
# rooms, the embedding does not.
SPACE_EMBEDDING_DIM = 8

# Whether to append cyclical time features to the input. Sine and cosine of the
# time of day avoid the artificial jump from 23:55 to 00:00 that an integer
# hour would introduce.
USE_TIME_FEATURES = True

# --- Training --------------------------------------------------------------
LEARNING_RATE = 1e-3  # Adam, standard value for recurrent networks
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 15  # epochs without validation improvement before stopping
MIN_IMPROVEMENT = 1e-5  # smallest validation gain that counts as an improvement

# Chronological split, not random. A random split would leave step t in train
# and step t+1 in test, so the model would be scored against data nearly
# identical to what it saw: the result looks excellent and means nothing.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
# The remaining 0.15 is the test split.

# "by_days" assigns whole calendar days, with validation and test drawn from
# the latest weekdays; "chronological" cuts each space's series at fixed
# fractions. With one week of data the chronological cut leaves Saturday and
# Sunday at the end, so 96% of validation and test fell on an empty building.
# Reference: Bergmeir, Hyndman & Koo (2018), doi:10.1016/j.csda.2017.11.003.
PARTITION_MODE = "by_days"
