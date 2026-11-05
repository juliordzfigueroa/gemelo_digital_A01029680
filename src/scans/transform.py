"""Move each floor LAS into the shared building frame.

The floors were registered separately in the Leica software and exported
each in its own plan frame: heights agree, but in plan every floor sits at a
different offset and two of them are rotated. The combined export of all five
floors (Aulas_3_Completo 1.e57) has them in one frame. The plan rotation and
translation that carry each floor into that frame were measured once by
registering a horizontal section of each floor against the same section of
the combined cloud (median residual 2-5 mm) and are stored per floor in
config/spaces.json under "transform".

The plan values say nothing about height. Floor 1 and floor 2 disagree
vertically (the ceiling of floor 1 sits above the floor of floor 2), and
stairwells cannot settle it automatically: flights repeat every storey, so a
match can lock onto the wrong one. A vertical correction, once known, goes in
the optional "dz".

This module applies them: p_building = R(rotation_deg) @ p_floor + (dx, dy),
z_building = z_floor + dz. It writes new LAS files and never touches the
originals.

Usage:
    python -m scans.transform --floor all --input-dir D:\\las_A3 --output-dir D:\\las_A3_alineado

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-05 - Added the optional vertical shift dz.

Reference:
    Besl, P. J., & McKay, N. D. (1992). A method for registration of 3-D
    shapes. IEEE Transactions on Pattern Analysis and Machine Intelligence,
    14(2), 239-256.
    ASPRS (2011). LAS specification, version 1.2.
"""

from __future__ import annotations

import argparse
import shutil
import struct
import time
from pathlib import Path

import numpy as np

from scans.convert import find_input, load_floors

# LAS 1.2 public header offsets (bytes).
OFFSET_TO_POINTS = 96
RECORD_LENGTH = 105
POINT_COUNT = 107
SCALE = 131
OFFSET = 155
BOUNDS = 179

CHUNK = 20_000_000


def read_header(path: Path) -> dict:
    """Read the fields of a LAS header this module needs.

    Args:
        path: LAS file.

    Returns:
        The raw header bytes up to the first point, and the parsed fields.
    """
    with open(path, "rb") as f:
        head = f.read(227)
        if head[:4] != b"LASF":
            raise ValueError(f"{path} no es un archivo LAS")
        start = struct.unpack_from("<I", head, OFFSET_TO_POINTS)[0]
        f.seek(0)
        raw = bytearray(f.read(start))
    return {
        "raw": raw,
        "start": start,
        "record_length": struct.unpack_from("<H", raw, RECORD_LENGTH)[0],
        "count": struct.unpack_from("<I", raw, POINT_COUNT)[0],
        "scale": np.array(struct.unpack_from("<3d", raw, SCALE)),
        "offset": np.array(struct.unpack_from("<3d", raw, OFFSET)),
        "bounds": np.array(struct.unpack_from("<6d", raw, BOUNDS)),
    }


def plan_matrix(rotation_deg: float) -> np.ndarray:
    """Return the 2x2 rotation for a plan angle in degrees, counterclockwise."""
    t = np.radians(rotation_deg)
    return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])


def transform_file(
    source: Path, target: Path, rotation_deg: float, dx: float, dy: float, dz: float = 0.0
) -> None:
    """Write a copy of one LAS with its plan coordinates moved.

    The integer coordinates are re-quantized on the original scale (1 mm for
    these exports), so the added error is at most half a millimetre.

    Args:
        source: Original LAS of one floor.
        target: LAS to write.
        rotation_deg: Plan rotation about the origin, counterclockwise.
        dx: Translation along x after the rotation, in metres.
        dy: Translation along y after the rotation, in metres.
        dz: Vertical shift in metres. Applied through the header offset, so
            the stored integers of z do not change.
    """
    h = read_header(source)
    scale, offset = h["scale"], h["offset"]
    rot = plan_matrix(rotation_deg)
    shift = np.array([dx, dy])

    # New bounds from the corners of the old ones: exact for the rotated box,
    # then tightened with the real points below.
    xmax, xmin, ymax, ymin, zmax, zmin = h["bounds"]
    corners = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymin], [xmax, ymax]]) @ rot.T + shift
    new_offset = offset.copy()
    new_offset[:2] = np.floor(corners.min(0))
    new_offset[2] = offset[2] + dz

    n, reclen = h["count"], h["record_length"]
    src = np.memmap(source, dtype=np.uint8, mode="r", offset=h["start"], shape=(n, reclen))
    target.parent.mkdir(parents=True, exist_ok=True)
    lo = np.full(2, np.inf)
    hi = np.full(2, -np.inf)
    start = time.time()
    with open(target, "wb") as out:
        out.write(h["raw"])
        for first in range(0, n, CHUNK):
            block = np.array(src[first:first + CHUNK])
            ints = block[:, :8].copy().view("<i4").reshape(-1, 2)
            xy = ints * scale[:2] + offset[:2]
            xy = xy @ rot.T + shift
            lo = np.minimum(lo, xy.min(0))
            hi = np.maximum(hi, xy.max(0))
            new_ints = np.round((xy - new_offset[:2]) / scale[:2]).astype("<i4")
            block[:, :8] = new_ints.view(np.uint8).reshape(-1, 8)
            out.write(block.tobytes())
            done = min(first + CHUNK, n)
            print(f"\r  {done / n:5.1%}  {done / 1e6:,.0f} M puntos", end="", flush=True)

    raw = bytearray(h["raw"])
    struct.pack_into("<3d", raw, OFFSET, *new_offset)
    struct.pack_into("<6d", raw, BOUNDS, hi[0], lo[0], hi[1], lo[1], zmax + dz, zmin + dz)
    with open(target, "r+b") as out:
        out.write(raw)
    print(f"\n  listo en {(time.time() - start) / 60:.1f} min")


def main() -> None:
    """Parse the command line and transform the requested floors."""
    parser = argparse.ArgumentParser(description="Lleva el LAS de cada piso al marco común del edificio.")
    parser.add_argument("--floor", required=True, help="número de piso (0 a 4) o 'all'")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.input_dir.resolve() == args.output_dir.resolve():
        raise SystemExit("La salida debe ser otra carpeta: los originales no se sobrescriben.")

    floors = load_floors()
    chosen = sorted(floors) if args.floor == "all" else [int(args.floor)]
    free = shutil.disk_usage(args.output_dir.anchor).free
    print(f"Espacio libre en {args.output_dir.anchor}: {free / 1e9:,.0f} GB")

    for floor in chosen:
        entry = floors[floor]
        tr = entry.get("transform")
        if tr is None:
            raise SystemExit(f"El piso {floor} no tiene 'transform' en spaces.json")
        source = find_input(args.input_dir, entry["source"])
        target = args.output_dir / source.name
        dz = tr.get("dz", 0.0)
        print(f"Piso {floor}: {source.name}  giro {tr['rotation_deg']:+.4f}°  "
              f"dx {tr['dx']:+.3f}  dy {tr['dy']:+.3f}  dz {dz:+.3f}")
        transform_file(source, target, tr["rotation_deg"], tr["dx"], tr["dy"], dz)


if __name__ == "__main__":
    main()
