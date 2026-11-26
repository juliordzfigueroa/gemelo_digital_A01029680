"""How numbers, units, ranges and trends are written in prompts and answers.

Shared by the single-space prompts (llm/context.py) and the floor prompts
(llm/floor.py), so a value reads the same wherever it appears. Every
judgment a person could read into a number ("alto", "sube 62.8 ppm") is
written here, by code, and handed to the model as text: the model is never
asked to make it.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Moved here from context.py; added the anomaly lines.

Reference:
    Section 4.3.4 of the research document.
"""

from __future__ import annotations

UNIT_LABELS = {
    "co2": "ppm",
    "temperature": "°C",
    "humidity": "%RH",
    "light": "%",
    "motion": "(1 = hay movimiento, 0 = no)",
}

VARIABLE_NAMES = {
    "co2": "CO₂",
    "temperature": "temperatura",
    "humidity": "humedad",
    "light": "luz",
    "motion": "movimiento",
}

# Comfort ranges used to annotate the readings so the model can say "normal" or
# "alto" without judging it on its own. CO2 has its own bands, below.
#
# These are working values, not taken from a standard: section 4.3.4 does not
# fix them, and the building's own criteria are still pending. They belong in
# config.py once they are decided, next to the physical limits.
HEALTHY_RANGE = {
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


def number(value: object) -> str:
    """Format a number for the prompt, or mark it as unavailable.

    Args:
        value: The value to render.

    Returns:
        The number with one decimal, or "no disponible" when it is missing.
    """
    if isinstance(value, (int, float)):
        return f"{value:.1f}"
    return "no disponible"


def short_unit(variable: str) -> str:
    """Return the unit written after a value in a sentence.

    Motion gets none: "movimiento 0.2" reads better than repeating what the
    scale means after every room.

    Args:
        variable: Variable name.

    Returns:
        The unit, or an empty string.
    """
    if variable == "motion":
        return ""
    return UNIT_LABELS.get(variable, "")


def readings(values: dict) -> str:
    """Render one reading per line, with units.

    Args:
        values: Variable name to value.

    Returns:
        The formatted block, or a line saying there is no data.
    """
    if not values:
        return "  Sin datos."
    lines = []
    for variable, value in values.items():
        lines.append(f"  {variable}: {number(value)} {UNIT_LABELS.get(variable, '')}".rstrip())
    return "\n".join(lines)


def compact(values: dict) -> str:
    """Render several readings on one line.

    Args:
        values: Variable name to value.

    Returns:
        A string such as "co2=680.0 ppm, temperature=22.1 °C".
    """
    parts = []
    for variable, value in values.items():
        parts.append(f"{variable}={number(value)} {UNIT_LABELS.get(variable, '')}".rstrip())
    return ", ".join(parts)


def threshold_notes(values: dict) -> str:
    """Annotate readings that fall outside their comfort range.

    Giving the model the band a value sits in, instead of letting it decide,
    is what keeps two identical readings from being described as "normal" in
    one answer and "elevado" in the next.

    Args:
        values: Variable name to value, in physical units.

    Returns:
        One line per annotated variable, empty when everything is in range.
    """
    notes = []

    co2 = values.get("co2")
    if isinstance(co2, (int, float)):
        for threshold, label, note in CO2_LEVELS:
            if co2 > threshold:
                notes.append(f"  CO₂ {co2:.0f} ppm: {label} ({note})")
                break

    for variable, (lower, upper) in HEALTHY_RANGE.items():
        value = values.get(variable)
        if not isinstance(value, (int, float)):
            continue
        unit = UNIT_LABELS[variable]
        comfort = f"rango de confort ({lower:.0f}-{upper:.0f})"
        if value < lower:
            notes.append(f"  {variable} {value:.1f} {unit}: por debajo del {comfort}")
        elif value > upper:
            notes.append(f"  {variable} {value:.1f} {unit}: por encima del {comfort}")

    return "\n".join(notes)


def anomaly(item: dict) -> str:
    """Describe one anomaly the detector found, for a prompt.

    Only the limit that was crossed is named: the lower end of the CO2 band
    is negative, and printing it would only confuse.

    Args:
        item: One entry of anomaly.detector.check().

    Returns:
        Text such as "co2 2400.0 ppm, 3.3σ por encima de lo normal (límite
        normal 2251.1 ppm)".
    """
    variable = item["variable"]
    unit = UNIT_LABELS.get(variable, "")
    low, high = item["band"]
    limit = high if item["direction"] == "por encima" else low
    return (
        f"{variable} {number(item['value'])} {unit}, {abs(item['sigma']):.1f}σ "
        f"{item['direction']} de lo normal (límite normal {number(limit)} {unit})"
    )


def anomaly_short(item: dict) -> str:
    """Describe one anomaly in a few words, for the room-by-room detail.

    Args:
        item: One entry of anomaly.detector.check().

    Returns:
        Text such as "CO₂ 2400.0 ppm (3.3σ por encima)".
    """
    variable = item["variable"]
    value = f"{VARIABLE_NAMES.get(variable, variable)} {number(item['value'])} {short_unit(variable)}"
    return f"{value.rstrip()} ({abs(item['sigma']):.1f}σ {item['direction']})"


def trend(variable: str, now: object, later: object) -> str:
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
    word = "sube" if change > 0 else "baja"
    return f"{word} {abs(change):.1f} {short_unit(variable)}".rstrip()
