# Estándar de código

Convención para todos los archivos `.py` del proyecto: cómo se documentan (encabezado,
docstrings y comentarios) y cómo se mantienen simples. Comentarios, docstrings e
identificadores en **inglés**: variables, funciones, clases, módulos, columnas
de datos, opciones de línea de comandos y campos de la API. Lo que lee una
persona se queda en **español**: mensajes en pantalla, prompts y respuestas del
asistente, y la documentación.

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

El bloque va **dentro** de la función, justo después de la firma. Es la única
posición en que Python lo reconoce como docstring.

```python
def normalize(values: np.ndarray, mean: float, sigma: float) -> np.ndarray:
    """Apply z-score normalization to a single variable.

    Optional paragraph for reasoning that is not obvious from the code: why
    this approach and not the alternative, what breaks if it changes.

    Args:
        values: Raw sensor readings.
        mean: Mean computed on the training split only.
        sigma: Standard deviation computed on the training split only.

    Returns:
        Normalized values with the same shape as the input.

    Raises:
        ValueError: If sigma is zero.

    Reference:
        Section 4.2.6 of the research document.
    """
```

> **Por qué no va arriba del `def`.** Un bloque de triple comilla colocado antes
> de la definición deja de ser un docstring: Python lo evalúa como una expresión
> suelta y lo descarta. El código sigue corriendo, pero `funcion.__doc__` queda
> en `None`, `help(funcion)` sale vacío, los tooltips de VS Code no muestran
> nada y cualquier generador de documentación produce páginas en blanco. Dentro
> de la función se conserva todo eso.

## Comentarios dentro del código

En inglés, y solo donde expliquen **por qué**, no **qué**. El código ya dice
qué hace; el comentario existe para lo que el código no puede decir: una
decisión entre alternativas, una restricción de la librería, un defecto
conocido del sensor.

```python
# Non-training steps become NaN instead of being cut out: cutting would splice
# distant days together and shift the phase of the Fourier terms.
training = np.where(is_train, series, np.nan)
```

No esto:

```python
# Replace the non-training values with NaN
training = np.where(is_train, series, np.nan)
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

## Simplicidad

El código tiene que poder leerlo y explicarlo quien lo defiende. Cuando hay
dos formas de escribir algo, gana la que se entiende a la primera.

**Sin opciones que nadie usa.** Un modo, un interruptor o un argumento que no
se usa se quita, aunque "pueda servir después". Si sirvió para medir algo, el
resultado se escribe en el README del módulo y el código se va. Cada opción es
un camino más que hay que leer, probar y mantener.

**Cada regla en un solo lugar.** Si dos partes del código necesitan la misma
regla (por ejemplo, qué ventanas son válidas), una la calcula y la otra la usa.
Dos copias de la misma regla terminan diciendo cosas distintas.

**Funciones cortas, archivos con una sola responsabilidad.** Como referencia,
una función de más de unas 40 líneas o un archivo de más de unas 400 se parten.
Un archivo de entrenamiento entrena; las pruebas van en otro.

**Bucles claros antes que construcciones compactas.** Se prefiere un `for`
con su comentario a una comprensión anidada, al operador `:=` o a
desempaquetar diccionarios dentro de una comprensión.

```python
# Sí
totals = {}
for variable in variables:
    totals[variable] = sum(values[variable])

# No
totals = {v: sum(x) for v, x in ((v, values[v]) for v in variables)}
```

**Un nombre por concepto.** Si la partición de prueba se llama `test` en el
código, se llama `test` en los docstrings, en los mensajes y en los
diccionarios. Los nombres de `Args` del docstring son exactamente los
parámetros de la función.

**Nada que apunte a algo que ya no existe.** Un mensaje que dice "corre
`python -m modulo`" tiene que nombrar un comando que funcione hoy.
