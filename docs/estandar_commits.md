# Estándar de commits

Tomado de los commits hechos desde GitHub (`Initial commit`, `Fix formatting in
README.md header`, `Update README steps for training process`): una línea, en
inglés, que dice qué cambia.

## Formato

```
<Verbo> <qué cambia>

<Cuerpo opcional: por qué, en 1 a 3 líneas.>
```

## Reglas del asunto

| Regla | Bien | Mal |
|---|---|---|
| En inglés, como los comentarios del código | `Add SARIMAX baseline` | `Agrega el baseline SARIMAX` |
| Imperativo: completa "this commit will…" | `Fix MAPE zero threshold` | `Fixed MAPE` / `Fixes MAPE` |
| Primera letra mayúscula, sin punto final | `Update README` | `update README.` |
| Máximo 60 caracteres | `Add GRU iterative prediction` | una oración que explica todo |
| Un solo cambio lógico | `Add LLM client` | `Add LLM client and fix README typo` |

Los nombres de archivos, funciones y variables se escriben tal cual, aunque
estén en español: `Fix truncated header in sarimax/modelo.py`.

## Verbos

| Verbo | Cuándo |
|---|---|
| `Add` | Algo nuevo: archivo, función, sección, dependencia |
| `Fix` | Algo que estaba mal y ahora no |
| `Update` | Cambio a algo que ya funcionaba |
| `Remove` | Quitar algo |
| `Move` / `Rename` | Cambiar de lugar o de nombre sin cambiar comportamiento |
| `Refactor` | Reorganizar código sin cambiar lo que hace |
| `Document` | Solo documentación, cuando `Update README` no basta |

## Cuerpo

Opcional. Solo cuando el **por qué** no se deduce del diff: una decisión entre
alternativas, una medición que la justifica, un efecto que no se ve. Separado
del asunto por una línea en blanco, de 1 a 3 líneas.

```
Detect GRU implementation from checkpoint keys

The saved metadata has no implementacion field, and loading nn.GRU
weights into the hand-written cell fails.
```

El detalle largo va al README del módulo o a un comentario en el código, no al
commit: ahí se encuentra cuando se necesita.

## Merges

```
Merge branch <nombre-de-la-rama>
```

## Lo que no va

- Líneas de atribución como `Co-Authored-By`.
- Mensajes genéricos: `Update files`, `Changes`, `WIP`, `fix`.
