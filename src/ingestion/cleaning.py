"""Clean raw sensor exports into the project data contract.

Turns the CSV the ESP32 nodes produce into the frame every model in this
repository expects: one row per space and 5-minute interval, with co2,
temperatura, humedad, luz and movimiento. It accepts several files at once,
because exports arrive week by week and overlap, and writes a report of every
reading it discarded and why, so the cleaning can be audited in the thesis
instead of taken on trust.

Decisions that are not obvious from the code:

  - Time comes from `formattedTimestamp`. The numeric `timestamp` column was
    saved in scientific notation by a spreadsheet and holds 57 distinct values
    for 370,341 rows, so it is unusable.
  - `tvoc` and `rssi` are dropped, by the advisor's decision to work with the
    five variables only.
  - `eco2` becomes `co2`, but it is the equivalent CO2 a metal-oxide sensor
    estimates from volatile compounds, not a measured NDIR value. It rises
    with cleaning products and markers, and stays above 1,000 ppm on weekends
    with no motion. Downstream code must not treat it as true CO2.
  - `lightPercent` becomes `luz` in percent (0-100), not lux.
  - Movement becomes the fraction of readings in the interval that saw motion,
    a value in [0, 1]. Keeping a single 0/1 per interval would discard almost
    everything: the PIR fires in 1.6-3.6% of weekday readings even in full
    classrooms.
  - Nothing is interpolated here. Gaps stay as missing rows, which is what the
    data contract asks for; the training pipeline decides what to bridge.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-12 - Initial implementation.

Reference:
    Section 4.2.6 of the research document; data contract in config.py.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import config as cfg

CONFIG_DIR = cfg.ROOT / "config"

# Raw column -> contract column. Anything not listed here is dropped.
COLUMNS = {
    "temperature": "temperature",
    "humidity": "humidity",
    "lightPercent": "light",
    "motion": "motion",
    "eco2": "co2",
}

TIME_FORMAT = "%d %b %Y, %H:%M:%S"

# The gas sensor reports -1 while it warms up or when it fails.
SENTINEL = -1

# Readings arrive about every 5 seconds, so a full 5-minute interval holds
# around 60. An interval backed by fewer than this many valid readings is
# dropped for that variable: an average of two readings is noise presented as
# a measurement.
MIN_READINGS_PER_INTERVAL = 6


def read_raw(paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    """Read one or more raw exports and merge them.

    Args:
        rutas: CSV files exported from the sensor platform.

    Returns:
        A tuple of (frame with deviceId, tiempo and the five raw variables,
        partial report). Exact repeats of the same device and second, which
        appear when exports overlap, are kept only once.

    Raises:
        ValueError: If a file lacks a required column or holds a timestamp that
            does not match TIME_FORMAT.
    """
    required = ["deviceId", "formattedTimestamp", *COLUMNS]
    parts = []
    for path in paths:
        raw = pd.read_csv(path)
        missing = set(required) - set(raw.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        parts.append(raw[required])

    df = pd.concat(parts, ignore_index=True)
    rows_read = len(df)

    # errors="raise" on purpose: a timestamp in an unexpected format must stop
    # the run, not become NaT and silently vanish with its reading.
    df["ts"] = pd.to_datetime(df.pop("formattedTimestamp"), format=TIME_FORMAT)
    df = df.drop_duplicates(subset=["deviceId", "ts"], keep="first")

    return df, {
        "files": [r.name for r in paths],
        "rows_read": rows_read,
        "duplicates_across_files": rows_read - len(df),
    }


def load_sensors(path: Path = CONFIG_DIR / "sensors.csv") -> pd.DataFrame:
    """Load the device-to-space table.

    Args:
        ruta: CSV with dispositivo, space_id, desde and hasta.

    Returns:
        The table with parsed dates. Empty dates mean an open interval.
    """
    table = pd.read_csv(path, dtype={"device": str})
    table["since"] = pd.to_datetime(table["since"])
    table["until"] = pd.to_datetime(table["until"])
    return table


def assign_spaces(
    df: pd.DataFrame, sensors: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    """Attach the space each reading belongs to.

    The model learns spaces, not devices, so every reading needs the space its
    device sat in at that moment. The table carries dates so that a sensor
    moved to another room splits correctly instead of merging two rooms into
    one identity.

    Args:
        df: Output of read_raw().
        sensores: Output of load_sensors().

    Returns:
        A tuple of (readings with an integer space_id, count of discarded
        readings per device and reason).
    """
    df = df.copy()
    df[cfg.SPACE_COLUMN] = np.nan

    for _, row in sensors.dropna(subset=[cfg.SPACE_COLUMN]).iterrows():
        mask = df["deviceId"] == row["device"]
        if pd.notna(row["since"]):
            mask &= df["ts"] >= row["since"]
        if pd.notna(row["until"]):
            mask &= df["ts"] < row["until"]
        df.loc[mask, cfg.SPACE_COLUMN] = int(row[cfg.SPACE_COLUMN])

    known = set(sensors["device"])
    no_space = df[cfg.SPACE_COLUMN].isna()
    discarded = {}
    for device, n in df.loc[no_space, "deviceId"].value_counts().items():
        reason = (
            "dispositivo sin espacio asignado"
            if device in known
            else "dispositivo no registrado en sensors.csv"
        )
        discarded[device] = {"rows": int(n), "reason": reason}

    df = df[~no_space].copy()
    df[cfg.SPACE_COLUMN] = df[cfg.SPACE_COLUMN].astype(int)
    return df, discarded


def invalidate(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Turn impossible readings into missing values.

    Three rules, each counted separately so the report says which one fired:

      - centinela: the sensor's own error code, -1.
      - lectura_vacia: temperature and humidity both exactly 0, the signature
        of a failed read on this kind of sensor rather than a real reading.
      - fuera_de_rango: outside the instrument limits in config.PHYSICAL_LIMITS.

    Only the affected variable is blanked; the rest of the reading survives.

    Args:
        df: Readings with the contract column names.

    Returns:
        A tuple of (cleaned copy, counts per device, variable and rule).
    """
    df = df.copy()
    counts: dict = {}

    def mark(mask: pd.Series, variable: str, rule: str) -> None:
        for device, n in df.loc[mask, "deviceId"].value_counts().items():
            (
                counts.setdefault(device, {})
                .setdefault(variable, {})
                .setdefault(rule, 0)
            )
            counts[device][variable][rule] += int(n)
        df.loc[mask, variable] = np.nan

    for variable in cfg.ACTIVE_VARIABLES:
        mark(df[variable] == SENTINEL, variable, "sentinel")

    empty = (df["temperature"] == 0) & (df["humidity"] == 0)
    for variable in ("temperature", "humidity"):
        mark(empty, variable, "empty_reading")

    for variable in cfg.ACTIVE_VARIABLES:
        lower, upper = cfg.PHYSICAL_LIMITS[variable]
        outside = pd.Series(False, index=df.index)
        if lower is not None:
            outside |= df[variable] < lower
        if upper is not None:
            outside |= df[variable] > upper
        if variable == "light":
            # Real nodes report light as a percentage, so it is bounded above
            # even though the synthetic lux scale in config is not.
            outside |= df[variable] > 100
        mark(outside & df[variable].notna(), variable, "out_of_range")

    return df, counts


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate readings into 5-minute intervals per space.

    Continuous variables are averaged. Movement is averaged too, which turns a
    stream of 0/1 into the fraction of the interval with motion detected.
    Intervals are labelled by their start.

    Args:
        df: Cleaned readings carrying space_id and tiempo.

    Returns:
        One row per space and interval, satisfying the data contract. Rows
        where every variable is missing are dropped, so gaps stay as absent
        rows.
    """
    interval = f"{cfg.INTERVAL_SECONDS}s"
    df = df.assign(
        **{cfg.TIME_COLUMN: df["ts"].dt.floor(interval)}
    )
    groups = df.groupby([cfg.SPACE_COLUMN, cfg.TIME_COLUMN])

    means = groups[cfg.ACTIVE_VARIABLES].mean()
    valid_counts = groups[cfg.ACTIVE_VARIABLES].count()
    means = means.where(valid_counts >= MIN_READINGS_PER_INTERVAL)

    output = means.dropna(how="all").reset_index()
    output[cfg.SPACE_COLUMN] = output[cfg.SPACE_COLUMN].astype(int)
    return output[[cfg.TIME_COLUMN, cfg.SPACE_COLUMN, *cfg.ACTIVE_VARIABLES]]


def clean(paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    """Run the whole cleaning, from raw exports to the data contract.

    Args:
        rutas: Raw CSV exports.

    Returns:
        A tuple of (contract frame, full report).
    """
    raw, report = read_raw(paths)
    raw = raw.rename(columns=COLUMNS)

    report["rows_by_device"] = {
        k: int(v) for k, v in raw["deviceId"].value_counts().sort_index().items()
    }

    assigned, discarded = assign_spaces(raw, load_sensors())
    report["discarded_by_device"] = discarded

    cleaned, invalid = invalidate(assigned)
    report["invalidated_values"] = invalid

    contract = aggregate(cleaned)

    summary = {}
    for space, group in contract.groupby(cfg.SPACE_COLUMN):
        complete_rows = group[cfg.ACTIVE_VARIABLES].notna().all(axis=1)
        summary[str(space)] = {
            "since": str(group[cfg.TIME_COLUMN].min()),
            "until": str(group[cfg.TIME_COLUMN].max()),
            "intervals": int(len(group)),
            "complete_intervals": int(complete_rows.sum()),
            "complete_hours": round(float(complete_rows.sum()) * cfg.INTERVAL_SECONDS / 3600, 1),
            "missing_by_variable": {
                v: int(group[v].isna().sum()) for v in cfg.ACTIVE_VARIABLES
            },
        }
    report["by_space"] = summary
    report["notes"] = [
        "co2 es eCO2 estimado por un sensor MOX, no CO2 medido (NDIR).",
        "luz esta en porcentaje (0-100), no en lux.",
        "movimiento es la fraccion del intervalo con deteccion, en [0, 1].",
        "tvoc y rssi se descartan por decision de la asesora.",
    ]
    return contract, report


def print_report(report: dict) -> None:
    """Print the parts of the report worth reading in the terminal.

    Args:
        reporte: Report returned by clean().
    """
    print(f"Archivos: {', '.join(report['files'])}")
    print(
        f"Filas leidas: {report['rows_read']:,}  "
        f"(duplicadas entre archivos: {report['duplicates_across_files']:,})"
    )

    if report["discarded_by_device"]:
        print("\nDescartadas por no tener espacio:")
        for device, d in report["discarded_by_device"].items():
            print(f"  {device:15} {d['rows']:>8,}  {d['reason']}")

    if report["invalidated_values"]:
        print("\nValores invalidados (la lectura sigue, sin esa variable):")
        for device, variables in sorted(report["invalidated_values"].items()):
            for variable, rules in variables.items():
                detail = ", ".join(f"{r}={n:,}" for r, n in rules.items())
                print(f"  {device:15} {variable:12} {detail}")

    print(
        f"\n{'space':>8} {'since':>17} {'until':>17} {'interv.':>8} "
        f"{'completos':>9} {'horas':>6}"
    )
    for space, r in report["by_space"].items():
        print(
            f"{space:>8} {r['since'][5:16]:>17} {r['until'][5:16]:>17} "
            f"{r['intervals']:>8} {r['complete_intervals']:>9} "
            f"{r['complete_hours']:>6}"
        )


def main() -> int:
    """Clean the given exports and write the contract frame and the report.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Clean raw sensor exports.")
    parser.add_argument("inputs", nargs="+", type=Path, help="raw CSV exports")
    parser.add_argument(
        "--output",
        type=Path,
        default=cfg.PROCESSED_DATA / "aulas3.parquet",
        help="where to write the contract frame",
    )
    args = parser.parse_args()

    contract, report = clean(args.inputs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    contract.to_parquet(args.output, index=False)
    report_path = args.output.with_name(args.output.stem + "_report.json")
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print_report(report)
    print(f"\nGuardado: {args.output}\nReporte:  {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
