"""Training loop for the GRU: Adam at lr=1e-3, MSE loss and early stopping.

Saves the best state by validation loss into models/, together with the
normalization statistics and the metadata needed to reproduce inference. The
model is only useful alongside its normalizer: without the mean and sigma it
was trained with, its outputs cannot be turned back into physical units.

Usage:
    python entrenar.py                      # synthetic data
    python entrenar.py --datos ruta.parquet # once real readings exist
    python entrenar.py --prueba-sintetica   # quick implementation check

Reference:
    Sections 4.2.6 and 5.5 of the research document.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

import config as cfg
import datos as dat
import metricas as met
import modelo as mod


def dispositivo() -> torch.device:
    """Pick the compute device.

    Returns:
        The CUDA device when available, otherwise CPU.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fijar_semilla(semilla: int = cfg.SEMILLA) -> None:
    """Seed every random source so a run can be reproduced.

    Args:
        semilla: Seed shared by torch, numpy and CUDA.
    """
    torch.manual_seed(semilla)
    np.random.seed(semilla)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semilla)


def una_epoca(
    red: nn.Module,
    cargador: DataLoader,
    perdida_fn: nn.Module,
    optimizador: torch.optim.Optimizer | None,
    dev: torch.device,
) -> float:
    """Run one full pass over a data loader.

    Args:
        red: The network.
        cargador: Batches to iterate over.
        perdida_fn: Loss function.
        optimizador: Optimizer to step, or None to evaluate without training.
        dev: Device to run on.

    Returns:
        Mean loss per sample over the pass.
    """
    entrenando = optimizador is not None
    red.train(entrenando)

    total, n = 0.0, 0
    with torch.set_grad_enabled(entrenando):
        for rasgos, espacio, objetivo in cargador:
            rasgos = rasgos.to(dev)
            espacio = espacio.to(dev)
            objetivo = objetivo.to(dev)

            prediccion = red(rasgos, espacio)
            perdida = perdida_fn(prediccion, objetivo)

            if entrenando:
                optimizador.zero_grad()
                perdida.backward()
                # Gradient clipping: recurrent networks can produce very large
                # gradients on sequences with abrupt jumps, and a single runaway
                # step ruins weights that were already learned.
                nn.utils.clip_grad_norm_(red.parameters(), max_norm=1.0)
                optimizador.step()

            total += perdida.item() * len(rasgos)
            n += len(rasgos)
    return total / max(n, 1)


def predecir(
    red: nn.Module, cargador: DataLoader, dev: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference over a whole loader.

    Args:
        red: Trained network.
        cargador: Batches to predict.
        dev: Device to run on.

    Returns:
        A tuple of (ground truth, predictions, space indices), all still in
        normalized space.

    Raises:
        RuntimeError: If the loader yielded no windows at all.
    """
    red.eval()
    reales, predichos, espacios = [], [], []
    with torch.no_grad():
        for rasgos, espacio, objetivo in cargador:
            prediccion = red(rasgos.to(dev), espacio.to(dev))
            reales.append(objetivo.numpy())
            predichos.append(prediccion.cpu().numpy())
            espacios.append(espacio.numpy())
    if not reales:
        raise RuntimeError("the loader produced no windows")
    return (
        np.concatenate(reales),
        np.concatenate(predichos),
        np.concatenate(espacios),
    )


def entrenar(
    df: pd.DataFrame,
    por_espacio: bool = False,
    epocas: int = cfg.EPOCAS_MAXIMAS,
    paciencia: int = cfg.PACIENCIA,
    silencioso: bool = False,
) -> dict:
    """Train the network and evaluate it on the test split.

    Args:
        df: Frame satisfying the data contract.
        por_espacio: Whether to normalize per space instead of globally.
        epocas: Maximum number of epochs.
        paciencia: Epochs without validation improvement before stopping.
        silencioso: Suppress progress output.

    Returns:
        A dict holding the trained network, its normalizer, the training
        history and the test metrics.

    Raises:
        RuntimeError: If any split ended up with no windows.
    """
    fijar_semilla()
    dev = dispositivo()

    conjuntos, normalizador, columnas, indice_espacios = dat.preparar(
        df, por_espacio=por_espacio
    )
    cargadores = dat.cargadores(conjuntos)

    if not silencioso:
        print(f"Dispositivo: {dev}")
        if dev.type == "cuda":
            print(f"  {torch.cuda.get_device_name(0)}")
        print(f"Rasgos de entrada ({len(columnas)}): {columnas}")
        print(
            f"Variables a predecir ({len(cfg.VARIABLES_ACTIVAS)}): "
            f"{cfg.VARIABLES_ACTIVAS}"
        )
        print("Ventanas por particion:")
        for nombre, conjunto in conjuntos.items():
            print(f"  {nombre:14} {len(conjunto):7,}")

    vacias = [n for n, c in conjuntos.items() if len(c) == 0]
    if vacias:
        raise RuntimeError(
            f"splits {vacias} produced no windows; "
            "more data or longer contiguous blocks are needed"
        )

    red = mod.construir(
        n_espacios=len(indice_espacios), n_rasgos_continuos=len(columnas)
    ).to(dev)
    if not silencioso:
        print(f"\nParametros entrenables: {red.n_parametros():,}")

    # MSE because this is regression over continuous values with no activation
    # on the output layer: the standard pairing from Table 1, section 5.3.
    perdida_fn = nn.MSELoss()
    optimizador = torch.optim.Adam(red.parameters(), lr=cfg.TASA_APRENDIZAJE)

    mejor_perdida = float("inf")
    mejor_estado = None
    mejor_epoca = 0
    sin_mejora = 0
    historia = []
    t0 = time.time()

    for epoca in range(1, epocas + 1):
        p_tren = una_epoca(
            red, cargadores["entrenamiento"], perdida_fn, optimizador, dev
        )
        p_val = una_epoca(red, cargadores["validacion"], perdida_fn, None, dev)
        historia.append({"epoca": epoca, "entrenamiento": p_tren, "validacion": p_val})

        if p_val < mejor_perdida - cfg.MIN_MEJORA:
            mejor_perdida = p_val
            mejor_epoca = epoca
            # Copy to CPU: a state dict left on the GPU carries the device with
            # it and fails to load on a machine without CUDA.
            mejor_estado = {
                k: v.detach().cpu().clone() for k, v in red.state_dict().items()
            }
            sin_mejora = 0
            marca = " *"
        else:
            sin_mejora += 1
            marca = ""

        if not silencioso and (epoca <= 5 or epoca % 10 == 0 or marca):
            print(
                f"  epoca {epoca:3d}  tren {p_tren:.6f}  val {p_val:.6f}"
                f"  (sin mejora: {sin_mejora}){marca}"
            )

        if sin_mejora >= paciencia:
            if not silencioso:
                print(
                    f"\nEarly stopping en la epoca {epoca}: "
                    f"{paciencia} epocas sin mejorar validacion."
                )
            break

    duracion = time.time() - t0
    if mejor_estado is not None:
        red.load_state_dict(mejor_estado)
    if not silencioso:
        print(
            f"Mejor epoca: {mejor_epoca} (val {mejor_perdida:.6f}) en {duracion:.1f} s"
        )

    # Evaluate in physical units, never in z-score space.
    real_n, pred_n, espacios = predecir(red, cargadores["prueba"], dev)
    inverso = {v: k for k, v in indice_espacios.items()}
    ids = np.array([inverso[int(i)] for i in espacios])
    real = normalizador.desnormalizar(real_n, cfg.VARIABLES_ACTIVAS, ids)
    pred = normalizador.desnormalizar(pred_n, cfg.VARIABLES_ACTIVAS, ids)

    resultados = met.evaluar(real, pred)
    return {
        "red": red,
        "normalizador": normalizador,
        "columnas": columnas,
        "indice_espacios": indice_espacios,
        "historia": historia,
        "mejor_epoca": mejor_epoca,
        "mejor_perdida_val": mejor_perdida,
        "segundos": duracion,
        "resultados": resultados,
        "horizonte": met.por_horizonte(real, pred),
    }


def guardar(salida: dict, carpeta: Path = cfg.MODELOS, etiqueta: str = "gru") -> None:
    """Persist the weights, the normalizer and the run metadata.

    Args:
        salida: Dict returned by entrenar().
        carpeta: Destination directory.
        etiqueta: Filename prefix for the three files written.
    """
    carpeta.mkdir(parents=True, exist_ok=True)
    torch.save(salida["red"].state_dict(), carpeta / f"{etiqueta}.pt")
    salida["normalizador"].guardar(carpeta / f"{etiqueta}_normalizador.json")

    metadatos = {
        "variables_activas": cfg.VARIABLES_ACTIVAS,
        "columnas_rasgos": salida["columnas"],
        "indice_espacios": {str(k): v for k, v in salida["indice_espacios"].items()},
        "pasos_entrada": cfg.PASOS_ENTRADA,
        "pasos_salida": cfg.PASOS_SALIDA,
        "unidades_ocultas": cfg.UNIDADES_OCULTAS,
        "dropout": cfg.DROPOUT,
        "mejor_epoca": salida["mejor_epoca"],
        "mejor_perdida_val": salida["mejor_perdida_val"],
        "metricas_prueba": salida["resultados"],
        "mae_por_horizonte": salida["horizonte"],
    }
    (carpeta / f"{etiqueta}_metadatos.json").write_text(
        json.dumps(metadatos, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"\nGuardado en {carpeta.relative_to(cfg.RAIZ)}/: {etiqueta}.pt, "
        f"{etiqueta}_normalizador.json, {etiqueta}_metadatos.json"
    )


def prueba_de_implementacion() -> int:
    """Check that the network can learn something whose answer is known.

    This is the test that stands in for reproducing a published benchmark: if
    the GRU cannot fit a clean, deterministic series, the bug is in the code
    and not in the data. It is not a performance result to report, only
    evidence that the pipeline works.

    Returns:
        Process exit code: 0 when the network learns, 1 when it does not.
    """
    import sintetico

    print("== Prueba de implementacion: serie sintetica sin huecos, 20 dias ==")
    df = sintetico.genera(dias=20, n_espacios=2, con_huecos=False)
    salida = entrenar(df, epocas=30, paciencia=8, silencioso=False)
    met.imprimir(salida["resultados"], "Prueba de implementacion")

    r2_global = salida["resultados"]["global"]["r2"]
    continuas = [v for v in cfg.VARIABLES_ACTIVAS if v not in cfg.BINARIAS]
    r2_continuas = [salida["resultados"]["por_variable"][v]["r2"] for v in continuas]
    print(
        f"\n  R2 global {r2_global:.4f} | R2 de las continuas: "
        + ", ".join(f"{v}={r:.3f}" for v, r in zip(continuas, r2_continuas))
    )

    if min(r2_continuas) < 0.5:
        print("\n  FALLO: hay variables continuas con R2 < 0.5 sobre una serie")
        print("  limpia y predecible. El problema esta en el codigo, no en los datos.")
        return 1
    print("\n  OK: la red aprende la estructura de una serie conocida.")
    return 0


def main() -> int:
    """Parse arguments, train, report and save.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Train the environmental GRU.")
    parser.add_argument(
        "--datos",
        type=Path,
        default=None,
        help="parquet or csv satisfying the data contract; synthetic if omitted",
    )
    parser.add_argument(
        "--dias", type=int, default=60, help="days to generate when --datos is absent"
    )
    parser.add_argument("--epocas", type=int, default=cfg.EPOCAS_MAXIMAS)
    parser.add_argument("--paciencia", type=int, default=cfg.PACIENCIA)
    parser.add_argument(
        "--normalizar-por-espacio",
        action="store_true",
        help="fit mean and sigma per space instead of globally",
    )
    parser.add_argument("--etiqueta", default="gru")
    parser.add_argument(
        "--prueba-sintetica",
        action="store_true",
        help="run only the implementation check and exit",
    )
    args = parser.parse_args()

    if args.prueba_sintetica:
        return prueba_de_implementacion()

    if args.datos is None:
        import sintetico

        print(f"Sin --datos: generando {args.dias} dias sinteticos.")
        df = sintetico.genera(dias=args.dias)
    elif args.datos.suffix == ".parquet":
        df = pd.read_parquet(args.datos)
    else:
        df = pd.read_csv(args.datos, parse_dates=[cfg.COLUMNA_TIEMPO])

    salida = entrenar(
        df,
        por_espacio=args.normalizar_por_espacio,
        epocas=args.epocas,
        paciencia=args.paciencia,
    )
    met.imprimir(salida["resultados"], "Conjunto de prueba")

    print("\n== MAE por horizonte de prediccion ==")
    variables = list(next(iter(salida["horizonte"].values())).keys())
    print(f"  {'horizonte':10} " + " ".join(f"{v:>12}" for v in variables))
    for horizonte, valores in salida["horizonte"].items():
        print(f"  {horizonte:10} " + " ".join(f"{valores[v]:12.4f}" for v in variables))

    guardar(salida, etiqueta=args.etiqueta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
