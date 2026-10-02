# Estándar de código

Convención para todos los archivos `.py` del proyecto. Comentarios y docstrings
en **inglés**; los identificadores siguen en español, igual que la tesis.

## Encabezado de archivo

La primera línea es el título. Después, un párrafo de qué hace y por qué existe.
Al final, en este orden: autoría, fecha y descripción de la última modificación,
y la referencia si el archivo implementa algo publicado o una sección del
documento de investigación.

```python
"""Short title of the file.

What this module does and why it exists. Enough context that someone opening
the file cold knows whether this is the file they are looking for.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-01 - What changed in that edit, in one line.

Reference:
    Author, A. (year). Title. DOI or URL.
"""
```

`Last modified` describe **qué** cambió, no solo que cambió: "Fixed MAPE to use
a scale-relative zero threshold" sirve, "Updated file" no.

> **Advertencia.** Este campo se escribe a mano, así que se desactualiza en
> cuanto alguien edita el archivo sin acordarse de tocarlo. Git ya lleva ese
> dato de forma confiable: `git log -1 --format='%ad %s' -- ruta/al/archivo`
> siempre dice la verdad. El encabezado existe para que el archivo se pueda leer
> fuera del repositorio, por ejemplo impreso como anexo de la tesis.

## Docstring de función

Estilo Google. El resumen va en una línea, en modo imperativo. Las secciones
`Args`, `Returns` y `Raises` solo aparecen cuando aplican: una función sin
argumentos no lleva `Args`, y una que no devuelve nada no lleva `Returns`.

```python
def normalizar(valores: np.ndarray, media: float, sigma: float) -> np.ndarray:
    """Apply z-score normalization to a single variable.

    Optional paragraph for reasoning that is not obvious from the code: why
    this approach and not the alternative, what breaks if it changes.

    Args:
        valores: Raw sensor readings.
        media: Mean computed on the training split only.
        sigma: Standard deviation computed on the training split only.

    Returns:
        Normalized values with the same shape as the input.

    Raises:
        ValueError: If sigma is zero.

    Reference:
        Section 4.2.6 of the research document.
    """
```

## Comentarios dentro del código

En inglés, y solo donde expliquen **por qué**, no **qué**. El código ya dice
qué hace; el comentario existe para lo que el código no puede decir: una
decisión entre alternativas, una restricción de la librería, un defecto
conocido del sensor.

```python
# Two separate nn.GRU modules instead of num_layers=2: PyTorch forces every
# layer inside one nn.GRU to share hidden_size, and the spec asks for [128, 64].
self.gru1 = nn.GRU(dim_entrada, 128, batch_first=True)
```

No esto:

```python
# Create the first GRU layer
self.gru1 = nn.GRU(dim_entrada, 128, batch_first=True)
```

## Referencias

Cuando el código implementa algo de la literatura o del documento de
investigación, se cita en el docstring, no en un comentario suelto. Dos formas:

- `Reference: Section 4.2.6 of the research document.`
- `Reference: Cho et al. (2014). arXiv:1406.1078.`

## Tipado

Anotaciones de tipo en toda firma pública. `from __future__ import annotations`
al inicio del archivo para poder usar la sintaxis moderna (`list[str]`,
`int | None`) sin depender de la versión de Python.
