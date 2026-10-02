# Gemelo Digital Aulas 3

Prototipo técnico del gemelo digital del edificio Aulas 3, Tec de Monterrey
Campus Santa Fe. Modelo GRU para predicción de variables ambientales.

Julio César Rodríguez Figueroa — A01029680

---

## Qué hay ahora

Implementado el modelo predictivo (parte 1 de la investigación): un GRU de dos
capas [128, 64] que predice CO₂, temperatura, humedad, luz y movimiento 30
minutos hacia adelante a partir de una hora de contexto, con un ID de espacio
como rasgo para que un solo modelo sirva a todo el edificio.

La celda recurrente está escrita a mano siguiendo las ecuaciones 1 a 4 de la
sección 4.2.7.2 del documento de investigación, en vez de usar `nn.GRU`, que
implementa una variante distinta. Así las ecuaciones que publica la tesis son
las que producen los resultados.

Los sensores de Aulas 3 todavía no existen: se recolectan durante dos meses. El
pipeline se desarrolla contra **datos sintéticos** generados para tener la
misma forma y los mismos defectos que tendrán los reales. El día que lleguen,
lo único que cambia es el argumento `--datos`.

---

## Requisitos

- Python con el entorno virtual del repo (`.venv`)
- GPU NVIDIA con CUDA (opcional; corre en CPU, solo más lento)

```bash
.venv/Scripts/python.exe check_gpu.py
```

Debe reportar `CUDA disponible: True` y el nombre de la GPU.

### Dependencias

```bash
.venv/Scripts/python.exe -m pip install torch pandas numpy scikit-learn statsmodels matplotlib pyarrow
```

PyTorch se instala con el índice de CUDA correspondiente; en esta máquina es
`cu128`.

---

## Cómo correrlo

Todos los comandos se ejecutan **desde `src/gru`**, porque los módulos se
importan como hermanos.

```bash
cd src/gru
```

### Verificación (~2.5 min)

Confirma que la red aprende una serie limpia cuya respuesta se conoce. Si falla,
el problema está en el código y no en los datos.

```bash
../../.venv/Scripts/python.exe entrenar.py --prueba-sintetica
```

### Entrenamiento completo (~30 min en RTX 4090)

```bash
../../.venv/Scripts/python.exe entrenar.py --dias 60
```

Con `--implementacion pytorch` baja a unos 2 minutos, útil mientras iteras. La
celda escrita a mano es ~14× más lenta porque un bucle de Python sobre los pasos
de tiempo no puede usar el kernel fusionado de cuDNN.

Deja en `models/` tres archivos: los pesos (`gru.pt`), los parámetros de
normalización (`gru_normalizador.json`) y las métricas de la corrida
(`gru_metadatos.json`). **Los tres son necesarios**: sin el normalizador, las
salidas del modelo no se pueden convertir a ppm ni a grados.

### Comparar las dos implementaciones recurrentes

Entrena la celda escrita a mano y la de PyTorch con la misma semilla y los
mismos datos, y verifica que lleguen a métricas equivalentes.

```bash
../../.venv/Scripts/python.exe comparar_implementaciones.py
```

### Generar solo los datos

```bash
../../.venv/Scripts/python.exe sintetico.py --dias 60
```

### Con datos reales, cuando existan

```bash
../../.venv/Scripts/python.exe entrenar.py --datos ../../data/raw/aulas3.parquet
```

### Opciones útiles

| Opción | Para qué |
|---|---|
| `--epocas N` | Tope de épocas (por omisión 200, pero el early stopping suele cortar antes) |
| `--paciencia N` | Épocas sin mejora antes de detener |
| `--normalizar-por-espacio` | Media y sigma por espacio en vez de globales |
| `--etiqueta nombre` | Prefijo de los archivos guardados, para comparar corridas |
| `--implementacion pytorch` | Usa `nn.GRU` en vez de la celda escrita a mano: ~15× más rápido, útil para iterar |

---

## El contrato de datos

El pipeline no sabe de dónde salen los datos. Cualquier fuente debe entregar un
DataFrame con estas columnas:

| Columna | Tipo | Nota |
|---|---|---|
| `timestamp` | datetime | hora local, intervalos de 5 min |
| `espacio_id` | int | identificador del espacio |
| `co2` | float | ppm |
| `temperatura` | float | °C |
| `humedad` | float | %RH |
| `luz` | float | lux |
| `movimiento` | float | 0 o 1 |

Los huecos se expresan como **filas ausentes**, no como NaN: es lo que pasa
cuando un nodo pierde la red.

---

## Estructura

```
src/gru/          modelo predictivo (ver src/gru/README.md)
docs/             guía de demo, estándar de código, hallazgos de ROBOD
data/raw/         datos de entrada (no versionados)
data/processed/   datos derivados (no versionados)
models/           pesos y metadatos (no versionados)
check_gpu.py      diagnóstico de CUDA
```

Pendientes: `src/sarimax/` (baseline), `src/anomalias/` (detector 3σ),
`src/llm/` (Llama 3.1) y `src/api/` (FastAPI).

---

## Documentación

| Documento | Contenido |
|---|---|
| [docs/guia_demo_gru.md](docs/guia_demo_gru.md) | Guion para presentar el GRU, conceptos clave y preguntas difíciles |
| [src/gru/README.md](src/gru/README.md) | Decisiones de diseño del modelo y qué queda abierto |
| [docs/estandar_codigo.md](docs/estandar_codigo.md) | Convención de documentación del código |
| [docs/hallazgos_robod.md](docs/hallazgos_robod.md) | Evaluación descartada de ROBOD y qué defectos reaparecerán con los sensores propios |

---

## Advertencia sobre los resultados

Las métricas que produce el pipeline hoy se calculan sobre datos sintéticos.
Miden qué tan bien el GRU aprende un generador escrito para este repositorio,
así que son **circulares por construcción** y no constituyen un resultado de
desempeño. Sirven para verificar que el pipeline corre y que la red aprende
estructura temporal. El desempeño real se mide con los datos de Aulas 3.
