"""Configuration for the GRU prediction model.

Holds the data contract, the physical limits of the sensors and the network
hyperparameters. By design this file contains no means, no standard deviations
and no anomaly thresholds: every value derived from data is computed from the
training split and stored next to the trained model, so that the same code
produces Aulas 3 parameters once the real sensors are deployed.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-25 - Removed options nobody used (implementation, split mode).

Reference:
    Section 4.2.6 of the research document.
    Sabiri, Y., Houmaidi, W., Bougrine, A., & El Mansour Billah, S. (2025).
    Optimizing indoor environmental quality in smart buildings using deep
    learning. arXiv:2509.26187.
"""

import os
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

# Shared space registry: names for the classifier, floor clouds and markers
# for the viewer.
SPACES_FILE = ROOT / "config" / "spaces.json"
VIEWER_DIR = ROOT / "web"

# --- Services ----------------------------------------------------------------
# Addresses come from the environment so the same code runs on this PC and in
# a container in the Tec cloud; the defaults are the local setup.
# OLLAMA_URL and not OLLAMA_HOST: Ollama itself reads OLLAMA_HOST as the
# address it binds to, without a scheme, and sharing the name would break it.
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# The scan server is a separate process (src/scans/server.py), so the clouds
# can live on another machine than the API.
SCANS_DIR = Path(os.environ.get("SCANS_DIR", r"D:\escaneos"))
SCANS_URL = os.environ.get("SCANS_URL", "http://localhost:8100")
SCANS_PORT = int(os.environ.get("SCANS_PORT", "8100"))
POTREE_CONVERTER = Path(os.environ.get(
    "POTREE_CONVERTER",
    r"D:\tools\PotreeConverter\PotreeConverter_2.1.5_x64_windows\PotreeConverter.exe",
))

# Pages allowed to read the clouds from another origin. The viewer is served by
# the API, so by default that is the API address.
VIEWER_ORIGINS = os.environ.get(
    "VIEWER_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
).split(",")

# Campus location, for weather queries and solar position. The sensor
# timestamps are local time, so weather is requested in the same zone.
LATITUDE = 19.3593
LONGITUDE = -99.2588
TIMEZONE = "America/Mexico_City"

SEED = 42

# --- Data contract ---------------------------------------------------------
# Every data source (the cleaned sensor file today, the Tec database later)
# must provide a DataFrame with these columns. Nothing downstream knows where
# the rows came from, which is what lets the pipeline survive the switch.
TIME_COLUMN = "timestamp"
SPACE_COLUMN = "space_id"

# Variables the model reads and predicts. Narrowing this list to
# ["co2", "temperature", "humidity"] reproduces the Sabiri (2025) setup.
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

# Width of the space-id embedding. An embedding rather than one-hot because
# Aulas 3 has many rooms across 5 floors: one-hot grows with the number of
# rooms, the embedding does not.
SPACE_EMBEDDING_DIM = 8

# --- Training --------------------------------------------------------------
LEARNING_RATE = 1e-3  # Adam, standard value for recurrent networks
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 15  # epochs without validation improvement before stopping
MIN_IMPROVEMENT = 1e-5  # smallest validation gain that counts as an improvement

# Share of calendar days for each split; the remaining 0.15 is the test split.
# Validation and test take whole weekdays, the latest ones, and weekends stay
# in training: a plain chronological cut over one week left Saturday and
# Sunday at the end, so 96% of validation and test fell on an empty building.
# Reference: Bergmeir, Hyndman & Koo (2018), doi:10.1016/j.csda.2017.11.003.
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15
