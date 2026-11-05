"""Static file server for the converted point clouds.

Serves the PotreeConverter output (one folder per floor with metadata.json,
hierarchy.bin and octree.bin) to the viewer. It is a separate process from the
API on purpose: the clouds weigh tens of gigabytes and can live on the machine
with the disk, while the API and the models move to the Tec cloud.

Two things a plain file server does not do and Potree needs:

- HTTP Range. Potree 2 never downloads octree.bin whole; it asks for the byte
  span of each octree node. `python -m http.server` ignores the Range header
  and sends the full file, which with a 40 GB floor means the viewer never
  loads. Starlette's FileResponse answers 206 Partial Content.
- CORS. The page comes from the API origin (port 8000) and the clouds from
  this one (port 8100), so the browser blocks the reads unless this server
  allows that origin and exposes Content-Range.

Usage:
    python -m scans.server
    python -m scans.server --dir D:\\escaneos --port 8100

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-03 - Initial implementation.

Reference:
    RFC 9110, section 14 (Range requests).
    Schütz, M. (2016). Potree: Rendering large point clouds in web browsers.
    Master's thesis, TU Wien.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from common import config as cfg


def list_clouds(root: Path) -> list[dict]:
    """Find every converted cloud under the served folder.

    Args:
        root: Folder served by this process.

    Returns:
        One entry per metadata.json, with its path relative to the root and
        the size of its octree in bytes.
    """
    clouds = []
    for metadata in sorted(root.glob("*/metadata.json")):
        octree = metadata.parent / "octree.bin"
        clouds.append({
            "cloud": metadata.relative_to(root).as_posix(),
            "octree_bytes": octree.stat().st_size if octree.exists() else None,
        })
    return clouds


def build_app(root: Path, origins: list[str]) -> Starlette:
    """Assemble the server for one folder.

    Args:
        root: Folder with one subfolder per converted cloud.
        origins: Pages allowed to read the clouds.

    Returns:
        The Starlette application.
    """

    async def index(request: Request) -> JSONResponse:
        return JSONResponse({"root": str(root), "clouds": list_clouds(root)})

    middleware = [
        Middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "HEAD"],
            # Range is what Potree sends; without it in the preflight answer
            # the browser drops every node request.
            allow_headers=["Range"],
            expose_headers=["Content-Range", "Content-Length", "Accept-Ranges"],
        )
    ]
    return Starlette(
        routes=[
            Route("/", index),
            Mount("/", app=StaticFiles(directory=root, check_dir=True)),
        ],
        middleware=middleware,
    )


def main() -> None:
    """Parse the command line and serve until interrupted."""
    parser = argparse.ArgumentParser(description="Servidor de nubes de puntos para Potree.")
    parser.add_argument("--dir", type=Path, default=cfg.SCANS_DIR,
                        help=f"carpeta con las nubes convertidas (por defecto {cfg.SCANS_DIR})")
    parser.add_argument("--port", type=int, default=cfg.SCANS_PORT)
    parser.add_argument("--host", default="127.0.0.1",
                        help="0.0.0.0 para aceptar conexiones de otras máquinas")
    args = parser.parse_args()

    if not args.dir.is_dir():
        raise SystemExit(f"No existe la carpeta de escaneos: {args.dir}")

    clouds = list_clouds(args.dir)
    print(f"Sirviendo {args.dir} en http://{args.host}:{args.port}")
    print(f"Nubes encontradas: {[c['cloud'] for c in clouds] or 'ninguna'}")
    print(f"Orígenes permitidos: {cfg.VIEWER_ORIGINS}")
    uvicorn.run(build_app(args.dir, cfg.VIEWER_ORIGINS), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
