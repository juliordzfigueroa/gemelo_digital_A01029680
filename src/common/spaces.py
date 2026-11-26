"""The space registry, config/spaces.json, read in one place.

Every part that needs to know the spaces of the building asks here: the
classifier for the words that name each one, the API for the name a person
uses and the floor each space is on, and the viewer, through the API, for the
floor clouds and the markers.

The file is read on every call instead of once at startup, so a marker
measured in Potree and saved to spaces.json shows up without restarting
anything. It is a few kilobytes; reading it again costs nothing.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Spaces without a room number take their "label".

Reference:
    Section 4.3.4 of the research document.
"""

from __future__ import annotations

import json

from common import config as cfg


def read() -> dict:
    """Read the registry as it is on disk.

    Returns:
        The parsed file, with its "floors" and "spaces" lists.
    """
    return json.loads(cfg.SPACES_FILE.read_text(encoding="utf-8"))


def labels() -> dict[int, str]:
    """Return the name a person uses for each space.

    The room number when there is one ("3308"); otherwise the "label" of the
    registry, which the offices and meeting rooms of floor 4 carry ("oficina
    mentores"). A space with neither gets a name built from the registry name
    ("space_4_2" would become "espacio 4-2"). Answers use these instead of
    ids, so they never say "espacio 5" or "space_4_2".

    Returns:
        A dict from space id to its label.
    """
    output = {}
    for space in read()["spaces"]:
        if space.get("room"):
            label = space["room"]
        elif space.get("label"):
            label = space["label"]
        else:
            number = space["name"].removeprefix("space_").replace("_", "-")
            label = f"espacio {number}"
        output[int(space["space_id"])] = label
    return output


def names() -> dict[int, list[str]]:
    """Return the words a user might type to name each space.

    Returns:
        A dict from space id to its lowercase names, longest first: the
        registry name, the room number or label, and every alias. Every name
        an answer shows can also be typed.
    """
    output = {}
    for space in read()["spaces"]:
        words = {space["name"], *space.get("alias", [])}
        if space.get("room"):
            words.add(space["room"])
        if space.get("label"):
            words.add(space["label"])
        output[int(space["space_id"])] = sorted(
            {str(w).lower() for w in words if w}, key=len, reverse=True
        )
    return output


def floor_spaces() -> dict[int, list[int]]:
    """Group the spaces by floor.

    Returns:
        A dict from floor number to the space ids on that floor.
    """
    output: dict[int, list[int]] = {}
    for space in read()["spaces"]:
        output.setdefault(int(space["floor"]), []).append(int(space["space_id"]))
    return output


def aliases(space_id: int) -> list[str]:
    """Return the other numbers a space is known by.

    Args:
        space_id: Space identifier.

    Returns:
        Its aliases, such as ["3302"] for the room that is also 3301, or an
        empty list.
    """
    for space in read()["spaces"]:
        if int(space["space_id"]) == space_id:
            return space.get("alias", [])
    return []
