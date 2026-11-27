"""The facts the explanations of the labels quote, taken from the data.

llm/glossary.py writes what "medido", "simulado" and "estimado" mean; this
file gives it what those sentences name: the rooms with a sensor, the days
they measured, the days the generator covers and its typical error, and,
for one space, which of the labels it gets and why.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-27 - Initial implementation.

Reference:
    Sections 4.3.4 and 4.4 of the research document.
"""

from __future__ import annotations

from api import retrieval as ret
from api import sources as src
from common import config as cfg
from common import spaces
from llm import formatting as fmt
from llm import replies


def facts() -> dict:
    """What the welcome and the definitions quote.

    Returns:
        {"rooms": the spaces with readings, already written as a list,
         "measured_span": the days they cover, "generated": whether the
         generator's rows are loaded, "generated_span": the days those
         cover, "error": the typical error of a generated value}.
    """
    named = []
    for room in ret.rooms_with_data():
        named.append(room if room.isdigit() else replies.with_article(room))
    measured = ret.STATE["data"][cfg.TIME_COLUMN]
    covered = src.covered()
    return {
        "rooms": replies.spoken_list(named) or "ningún salón todavía",
        "measured_span": fmt.date_span(measured.min(), measured.max()) if len(measured) else "",
        "generated": covered is not None,
        "generated_span": (
            fmt.date_span(*covered) if covered
            else "sin generar todavía (falta correr python -m generator.generate)"
        ),
        "error": fmt.expected_error(src.STATE["error"]),
    }


def space_status(space_id: int) -> tuple[str | None, dict]:
    """Which labels one space gets, and what the sentence about it quotes.

    Args:
        space_id: Space identifier.

    Returns:
        A key of glossary.SPACE_STATUS ("own", "few", "none",
        "only_measured" or "nothing"), or None without the generator, and
        the values the sentence quotes: subject, measured_span, based_on.
    """
    calibration = src.STATE["calibration"]
    if calibration is None:
        return None, {}

    rows = ret.space_rows(space_id)
    has_sensor = not rows.empty
    origin = calibration["rooms"].get(str(space_id), {}).get("origin")
    if origin == "simulado":
        status = "own"
    elif origin == "estimado":
        status = "few" if has_sensor else "none"
    else:
        status = "only_measured" if has_sensor else "nothing"

    values = {
        "subject": replies.capitalized(str(spaces.labels().get(space_id, space_id))),
        "measured_span": "",
        "based_on": "",
    }
    if has_sensor:
        times = rows[cfg.TIME_COLUMN]
        values["measured_span"] = fmt.date_span(times.min(), times.max())
    if origin == "estimado":
        rooms = src.provenance(space_id, "estimado")["based_on"]
        values["based_on"] = replies.with_article(replies.spoken_list(rooms))
    return status, values
