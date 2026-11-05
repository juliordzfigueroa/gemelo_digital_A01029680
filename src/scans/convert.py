"""Convert the LAS export of each floor into a Potree octree.

Wraps PotreeConverter 2 so every floor lands where config/spaces.json says the
viewer will look for it. The LAS files are exported from the Leica software,
one per floor, from the single registered project; that shared project is what
makes the five floors line up in the viewer without any manual alignment.

PotreeConverter 2 reads LAS and LAZ only, not E57, which is why the export
step exists.

Usage:
    python -m scans.convert --floor 1 --input D:\\las_A3\\Aulas_3.1.las
    python -m scans.convert --floor all --input-dir D:\\las_A3

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-03 - Initial implementation.

Reference:
    Schütz, M., Ohrhallinger, S., & Wimmer, M. (2020). Fast out-of-core
    octree generation for massive point clouds. Computer Graphics Forum,
    39(7), 155-167.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from common import config as cfg


def load_floors(path: Path = cfg.SPACES_FILE) -> dict[int, dict]:
    """Read the floor table from the space registry.

    Args:
        path: The shared space registry.

    Returns:
        A dict from floor number to its entry (cloud path and source name).
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    return {int(f["floor"]): f for f in data["floors"]}


def find_input(input_dir: Path, source: str) -> Path:
    """Locate the export of one floor by the name of its E57.

    Args:
        input_dir: Folder with the LAS or LAZ exports.
        source: E57 name recorded in spaces.json, e.g. "Aulas_3.1.e57".

    Returns:
        The matching .las or .laz file.

    Raises:
        FileNotFoundError: If neither extension exists.
    """
    stem = Path(source).stem
    for extension in (".laz", ".las"):
        candidate = input_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No encontré {stem}.las ni {stem}.laz en {input_dir}")


def convert(source: Path, output: Path, converter: Path = cfg.POTREE_CONVERTER) -> None:
    """Run PotreeConverter on one file.

    Poisson sampling (the default) keeps an even spacing at every level of
    detail, so walls read as surfaces when zoomed out instead of as noise.

    Args:
        source: LAS or LAZ file of one floor.
        output: Folder that will hold metadata.json, hierarchy.bin and
            octree.bin.
        converter: PotreeConverter executable.

    Raises:
        FileNotFoundError: If the executable is missing.
        subprocess.CalledProcessError: If the conversion fails.
    """
    if not converter.exists():
        raise FileNotFoundError(
            f"No encontré PotreeConverter en {converter}. "
            "Define POTREE_CONVERTER con la ruta del .exe."
        )
    output.mkdir(parents=True, exist_ok=True)
    print(f"{source.name} -> {output}", flush=True)
    start = time.time()
    subprocess.run(
        [str(converter), str(source), "-o", str(output), "-m", "poisson"],
        check=True,
    )
    print(f"Listo en {(time.time() - start) / 60:.1f} min")


def main() -> None:
    """Parse the command line and convert the requested floors."""
    parser = argparse.ArgumentParser(description="Convierte el LAS de cada piso a formato Potree.")
    parser.add_argument("--floor", required=True,
                        help="número de piso (0 a 4) o 'all'")
    parser.add_argument("--input", type=Path,
                        help="archivo LAS/LAZ de un piso")
    parser.add_argument("--input-dir", type=Path,
                        help="carpeta con los LAS/LAZ nombrados como el e57 (Aulas_3.N.las)")
    parser.add_argument("--output-dir", type=Path, default=cfg.SCANS_DIR)
    args = parser.parse_args()

    floors = load_floors()
    chosen = sorted(floors) if args.floor == "all" else [int(args.floor)]

    for floor in chosen:
        entry = floors[floor]
        if args.input and len(chosen) == 1:
            source = args.input
        elif args.input_dir:
            source = find_input(args.input_dir, entry["source"])
        else:
            raise SystemExit("Indica --input (un piso) o --input-dir.")
        # The cloud path in spaces.json ends in metadata.json; its folder is
        # the converter output.
        output = args.output_dir / Path(entry["cloud"]).parent
        convert(source, output)


if __name__ == "__main__":
    main()
