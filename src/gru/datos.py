"""Data pipeline: from the data contract to the tensors the GRU consumes.

Responsibilities, in order:

  1. Split each space into contiguous time blocks. Windows never cross a cut:
     joining two stretches separated by days would manufacture examples the
     model would learn as if they were real.
  2. Reindex each block onto the regular 5-minute grid and interpolate short
     gaps. Long gaps already split the block.
  3. Split chronologically into train, validation and test.
  4. Normalize with z-score, fitting mean and sigma on the training split only.
  5. Build the sliding windows of 12 input steps and 6 output steps.

Steps 3 and 4 are where the most common leakage in time-series pipelines
creeps in, so both are deliberately explicit here.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-01 - Applied the project documentation standard.

Reference:
    Section 4.2.6 of the research document.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

import config as cfg


# --------------------------------------------------------------------------
# Contiguous blocks and regular grid
# --------------------------------------------------------------------------
def marcar_bloques(df: pd.DataFrame) -> pd.DataFrame:
    """Label contiguous time blocks within each space.

    A gap longer than MAX_PASOS_INTERPOLABLES ends the current block.

    Args:
        df: Data contract frame, with timestamp and space columns.

    Returns:
        The same rows plus a 'bloque_id' column.
    """
    paso = pd.Timedelta(seconds=cfg.INTERVALO_SEGUNDOS)
    tolerancia = paso * (cfg.MAX_PASOS_INTERPOLABLES + 1)

    partes = []
    for espacio, grupo in df.groupby(cfg.COLUMNA_ESPACIO, sort=True):
        grupo = grupo.sort_values(cfg.COLUMNA_TIEMPO).copy()
        salto = grupo[cfg.COLUMNA_TIEMPO].diff()
        corte = (salto > tolerancia) | salto.isna()
        numeros = corte.cumsum().astype(int)
        grupo["bloque_id"] = [f"E{espacio}-B{n:03d}" for n in numeros]
        partes.append(grupo)
    return pd.concat(partes, ignore_index=True)


def regularizar(df: pd.DataFrame) -> pd.DataFrame:
    """Snap each block onto the 5-minute grid and fill short gaps.

    Interpolation is inside-only, meaning gaps surrounded by valid data.
    Extrapolating at the edges of a block would invent values with nothing to
    support them. Binary variables are forward-filled instead of interpolated,
    so no meaningless intermediate value such as 0.5 is produced for a PIR.

    Args:
        df: Frame already carrying a 'bloque_id' column.

    Returns:
        Frame on a regular grid, with short gaps filled and long ones left as
        missing values.
    """
    paso = f"{cfg.INTERVALO_SEGUNDOS}s"
    partes = []
    for _, grupo in df.groupby("bloque_id", sort=True):
        grupo = grupo.sort_values(cfg.COLUMNA_TIEMPO).set_index(cfg.COLUMNA_TIEMPO)
        rejilla = pd.date_range(grupo.index.min(), grupo.index.max(), freq=paso)
        grupo = grupo.reindex(rejilla)
        grupo[cfg.COLUMNA_ESPACIO] = grupo[cfg.COLUMNA_ESPACIO].ffill().bfill()
        grupo["bloque_id"] = grupo["bloque_id"].ffill().bfill()

        for variable in cfg.VARIABLES_ACTIVAS:
            if variable in cfg.BINARIAS:
                grupo[variable] = grupo[variable].ffill(
                    limit=cfg.MAX_PASOS_INTERPOLABLES
                )
            else:
                grupo[variable] = grupo[variable].interpolate(
                    method="linear",
                    limit=cfg.MAX_PASOS_INTERPOLABLES,
                    limit_area="inside",
                )
        partes.append(grupo.rename_axis(cfg.COLUMNA_TIEMPO).reset_index())

    salida = pd.concat(partes, ignore_index=True)
    salida[cfg.COLUMNA_ESPACIO] = salida[cfg.COLUMNA_ESPACIO].astype(int)
    return salida


def agregar_rasgos_de_tiempo(df: pd.DataFrame) -> list[str]:
    """Add time of day and day of week as sine/cosine pairs.

    Encoded cyclically rather than as integers: 23:55 is one step away from
    00:00, but as numbers they sit 23 apart. Sine and cosine together express
    that circularity without a discontinuity.

    Args:
        df: Frame to extend in place.

    Returns:
        Names of the columns that were added, in order.
    """
    t = df[cfg.COLUMNA_TIEMPO]
    minutos = t.dt.hour * 60 + t.dt.minute
    ang_dia = 2 * np.pi * minutos / (24 * 60)
    ang_sem = 2 * np.pi * t.dt.dayofweek / 7

    df["hora_sin"] = np.sin(ang_dia)
    df["hora_cos"] = np.cos(ang_dia)
    df["sem_sin"] = np.sin(ang_sem)
    df["sem_cos"] = np.cos(ang_sem)
    return ["hora_sin", "hora_cos", "sem_sin", "sem_cos"]


# --------------------------------------------------------------------------
# Chronological split
# --------------------------------------------------------------------------
def particionar(df: pd.DataFrame) -> pd.DataFrame:
    """Tag every row as train, validation or test by time.

    Cuts run along each space's time axis, never at random. A random split
    would leave step t in training and step t+1 in test, scoring the model
    against data practically identical to what it already saw: the result
    looks excellent and means nothing.

    Args:
        df: Frame on a regular grid.

    Returns:
        The same rows plus a 'particion' column.
    """
    partes = []
    for _, grupo in df.groupby(cfg.COLUMNA_ESPACIO, sort=True):
        grupo = grupo.sort_values(cfg.COLUMNA_TIEMPO).copy()
        n = len(grupo)
        fin_tren = int(n * cfg.FRACCION_ENTRENAMIENTO)
        fin_val = int(n * (cfg.FRACCION_ENTRENAMIENTO + cfg.FRACCION_VALIDACION))
        particion = np.array(["prueba"] * n, dtype=object)
        particion[:fin_tren] = "entrenamiento"
        particion[fin_tren:fin_val] = "validacion"
        grupo["particion"] = particion
        partes.append(grupo)
    return pd.concat(partes, ignore_index=True)


# --------------------------------------------------------------------------
# Normalization
# --------------------------------------------------------------------------
class Normalizador:
    """Per-variable z-score fitted on the training split only.

    The variables live on wildly different scales (CO2 in hundreds of ppm,
    movement in 0/1), and without normalization the network would weight the
    large-magnitude ones out of proportion.

    Two explicit decisions:

      - Binary variables are NOT normalized. Applying (x - 0.4) / 0.49 to
        something that only takes 0 or 1 improves nothing and makes the output
        unreadable.
      - `por_espacio` chooses between global per-variable statistics, as the
        research document specifies, and per-space statistics. With light
        levels differing by two orders of magnitude between spaces, global
        statistics collapse the dark spaces near zero. The default follows the
        document; this is a parameter, not a decision buried in the code.

    Reference:
        Section 4.2.6 of the research document.
    """

    def __init__(self, por_espacio: bool = False) -> None:
        """Create an unfitted normalizer.

        Args:
            por_espacio: If True, fit one mean and sigma per space and
                variable instead of one per variable.
        """
        self.por_espacio = por_espacio
        self.parametros: dict[str, dict[str, float]] = {}

    @staticmethod
    def _clave(variable: str, espacio: int | None) -> str:
        """Build the lookup key for a variable, optionally scoped to a space.

        Args:
            variable: Variable name.
            espacio: Space id, or None for global statistics.

        Returns:
            The dictionary key under which the statistics are stored.
        """
        return variable if espacio is None else f"{variable}@{espacio}"

    def ajustar(self, entrenamiento: pd.DataFrame) -> "Normalizador":
        """Compute mean and sigma from the training split.

        Args:
            entrenamiento: Rows belonging to the training partition only.
                Passing the full frame here would leak the future into
                preprocessing and inflate every metric.

        Returns:
            This normalizer, so calls can be chained.
        """
        for variable in cfg.VARIABLES_ACTIVAS:
            if variable in cfg.BINARIAS:
                continue
            if self.por_espacio:
                for espacio, grupo in entrenamiento.groupby(cfg.COLUMNA_ESPACIO):
                    self._ajustar_uno(variable, grupo[variable], espacio)
            else:
                self._ajustar_uno(variable, entrenamiento[variable], None)
        return self

    def _ajustar_uno(
        self, variable: str, serie: pd.Series, espacio: int | None
    ) -> None:
        """Store mean and sigma for one variable, optionally per space.

        Args:
            variable: Variable name.
            serie: Training values for that variable.
            espacio: Space id, or None for global statistics.
        """
        valores = serie.dropna()
        sigma = float(valores.std())
        # A variable that is constant in training would give sigma 0 and a
        # division by zero. Falling back to 1 turns normalization into a plain
        # shift, which is the harmless behaviour.
        self.parametros[self._clave(variable, espacio)] = {
            "media": float(valores.mean()),
            "sigma": sigma if sigma > 1e-8 else 1.0,
        }

    def transformar(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted z-score to a frame.

        Args:
            df: Frame holding the active variables.

        Returns:
            A copy with the continuous variables normalized.
        """
        salida = df.copy()
        for variable in cfg.VARIABLES_ACTIVAS:
            if variable in cfg.BINARIAS:
                continue
            if self.por_espacio:
                for espacio in salida[cfg.COLUMNA_ESPACIO].unique():
                    p = self.parametros[self._clave(variable, int(espacio))]
                    filas = salida[cfg.COLUMNA_ESPACIO] == espacio
                    salida.loc[filas, variable] = (
                        salida.loc[filas, variable] - p["media"]
                    ) / p["sigma"]
            else:
                p = self.parametros[self._clave(variable, None)]
                salida[variable] = (salida[variable] - p["media"]) / p["sigma"]
        return salida

    def desnormalizar(
        self,
        valores: np.ndarray,
        variables: list[str],
        espacios: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return values to their original physical units.

        Args:
            valores: Array shaped (n, pasos, n_variables) in normalized space.
            variables: Variable names in the order of the last axis.
            espacios: Space ids shaped (n,). Required when por_espacio is True.

        Returns:
            An array of the same shape, in the original units.

        Raises:
            ValueError: If por_espacio is True and espacios is None.
        """
        salida = np.asarray(valores, dtype=float).copy()
        for i, variable in enumerate(variables):
            if variable in cfg.BINARIAS:
                continue
            if self.por_espacio:
                if espacios is None:
                    raise ValueError("por_espacio=True requires the space ids")
                for espacio in np.unique(espacios):
                    p = self.parametros[self._clave(variable, int(espacio))]
                    filas = espacios == espacio
                    salida[filas, :, i] = salida[filas, :, i] * p["sigma"] + p["media"]
            else:
                p = self.parametros[self._clave(variable, None)]
                salida[:, :, i] = salida[:, :, i] * p["sigma"] + p["media"]
        return salida

    def guardar(self, ruta: Path) -> None:
        """Write the fitted statistics to disk as JSON.

        The model is useless without this file: without the mean and sigma it
        was trained with, its outputs cannot be turned back into ppm or
        degrees.

        Args:
            ruta: Destination path.
        """
        ruta.write_text(
            json.dumps(
                {"por_espacio": self.por_espacio, "parametros": self.parametros},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def cargar(cls, ruta: Path) -> "Normalizador":
        """Rebuild a normalizer from a saved JSON file.

        Args:
            ruta: Path written by guardar().

        Returns:
            A fitted Normalizador.
        """
        datos = json.loads(ruta.read_text(encoding="utf-8"))
        norm = cls(por_espacio=datos["por_espacio"])
        norm.parametros = datos["parametros"]
        return norm


# --------------------------------------------------------------------------
# Windows
# --------------------------------------------------------------------------
class VentanasAmbientales(Dataset):
    """Sliding windows of 12 input steps and 6 output steps.

    A window is valid only if all three conditions hold:
      - its 18 steps fall inside the same contiguous block;
      - none of its steps has a missing value in the active variables;
      - all 18 steps belong to the same partition.

    The third condition is what stops a test window from being fed by training
    steps.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        particion: str,
        columnas_rasgos: list[str],
        indice_espacios: dict[int, int],
    ) -> None:
        """Materialize every valid window of one partition as tensors.

        Args:
            df: Normalized frame carrying 'bloque_id' and 'particion'.
            particion: Which partition to build, e.g. "entrenamiento".
            columnas_rasgos: Input column names, in order.
            indice_espacios: Mapping from space id to embedding index.
        """
        self.columnas_rasgos = columnas_rasgos
        self.variables = cfg.VARIABLES_ACTIVAS
        self.indice_espacios = indice_espacios

        entradas, salidas, espacios = [], [], []
        largo = cfg.PASOS_ENTRADA + cfg.PASOS_SALIDA

        for _, bloque in df.groupby("bloque_id", sort=True):
            bloque = bloque.sort_values(cfg.COLUMNA_TIEMPO)
            if len(bloque) < largo:
                continue

            rasgos = bloque[columnas_rasgos].to_numpy(dtype=np.float32)
            objetivo = bloque[self.variables].to_numpy(dtype=np.float32)
            completa = ~np.isnan(rasgos).any(axis=1) & ~np.isnan(objetivo).any(axis=1)
            en_particion = (bloque["particion"] == particion).to_numpy()
            espacio = int(bloque[cfg.COLUMNA_ESPACIO].iloc[0])

            valida = completa & en_particion
            for inicio in range(len(bloque) - largo + 1):
                fin = inicio + largo
                if not valida[inicio:fin].all():
                    continue
                corte = inicio + cfg.PASOS_ENTRADA
                entradas.append(rasgos[inicio:corte])
                salidas.append(objetivo[corte:fin])
                espacios.append(indice_espacios[espacio])

        if entradas:
            self.X = torch.from_numpy(np.stack(entradas))
            self.y = torch.from_numpy(np.stack(salidas))
            self.espacio = torch.tensor(espacios, dtype=torch.long)
        else:
            n_rasgos, n_vars = len(columnas_rasgos), len(self.variables)
            self.X = torch.empty(0, cfg.PASOS_ENTRADA, n_rasgos)
            self.y = torch.empty(0, cfg.PASOS_SALIDA, n_vars)
            self.espacio = torch.empty(0, dtype=torch.long)

    def __len__(self) -> int:
        """Count the windows this partition produced.

        Returns:
            Number of valid windows available for training or evaluation.
        """
        return len(self.X)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return one training example.

        Args:
            i: Window index.

        Returns:
            A tuple of (features, space index, target).
        """
        return self.X[i], self.espacio[i], self.y[i]


# --------------------------------------------------------------------------
def preparar(
    df: pd.DataFrame, por_espacio: bool = False
) -> tuple[dict[str, VentanasAmbientales], Normalizador, list[str], dict[int, int]]:
    """Run the whole pipeline, from the data contract to the three splits.

    Args:
        df: Frame satisfying the data contract described in config.py.
        por_espacio: Whether to normalize per space instead of globally.

    Returns:
        A tuple of (splits, normalizer, feature column names, space index map).
        `splits` is keyed by "entrenamiento", "validacion" and "prueba".

    Raises:
        ValueError: If the frame does not satisfy the data contract.
    """
    faltan = {cfg.COLUMNA_TIEMPO, cfg.COLUMNA_ESPACIO, *cfg.VARIABLES_ACTIVAS} - set(
        df.columns
    )
    if faltan:
        raise ValueError(f"frame violates the data contract, missing: {sorted(faltan)}")

    df = marcar_bloques(df)
    df = regularizar(df)
    df = marcar_bloques(df)  # the grid may have renumbered things; relabel
    df = particionar(df)

    columnas_tiempo = agregar_rasgos_de_tiempo(df) if cfg.USAR_RASGOS_DE_TIEMPO else []
    columnas_rasgos = list(cfg.VARIABLES_ACTIVAS) + columnas_tiempo

    normalizador = Normalizador(por_espacio=por_espacio)
    normalizador.ajustar(df[df["particion"] == "entrenamiento"])
    df = normalizador.transformar(df)

    espacios = sorted(df[cfg.COLUMNA_ESPACIO].unique())
    indice_espacios = {int(e): i for i, e in enumerate(espacios)}

    conjuntos = {
        nombre: VentanasAmbientales(df, nombre, columnas_rasgos, indice_espacios)
        for nombre in ("entrenamiento", "validacion", "prueba")
    }
    return conjuntos, normalizador, columnas_rasgos, indice_espacios


def cargadores(
    conjuntos: dict[str, VentanasAmbientales], lote: int = cfg.TAMANO_LOTE
) -> dict[str, DataLoader]:
    """Wrap the splits in DataLoaders.

    Only the training split is shuffled. Shuffling window order there is both
    correct and desirable: each window is already a complete, self-contained
    example, and the order they are shown in should not influence the weights.
    Validation and test keep chronological order so predictions can be plotted
    against time.

    Args:
        conjuntos: Splits returned by preparar().
        lote: Batch size.

    Returns:
        One DataLoader per split, under the same keys.
    """
    return {
        nombre: DataLoader(
            conjunto,
            batch_size=lote,
            shuffle=(nombre == "entrenamiento"),
            drop_last=False,
        )
        for nombre, conjunto in conjuntos.items()
    }
