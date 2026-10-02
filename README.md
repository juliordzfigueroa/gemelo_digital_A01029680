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

### Dependencias

Solo la primera vez, con el entorno activado:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install pandas numpy scikit-learn statsmodels matplotlib pyarrow
```

PyTorch se instala con el índice de CUDA correspondiente; en esta máquina es
`cu128`.

| Paquete | Versión | Para qué |
|---|---|---|
| `torch` | 2.11.0+cu128 | La red neuronal y el cómputo en GPU |
| `pandas` | 3.0.6 | Series de tiempo, agregación, ventanas |
| `numpy` | 2.5.2 | Operaciones numéricas y métricas |
| `statsmodels` | 0.15.0 | SARIMAX (filtro de Kalman) |
| `scikit-learn` | 1.9.1 | Utilidades de partición |
| `pyarrow` | 25.0.1 | Leer y escribir Parquet |
| `matplotlib` | 3.11.2 | Gráficas |

---

## Demo completa, en orden

Siete pasos, unos 90 segundos de cómputo en total.

### Paso 1 — Activar el entorno virtual

```powershell
cd C:\Users\A01029680\Desktop\RepositorioJCRF\gemelo_digital_A01029680
.\.venv\Scripts\Activate.ps1
```

Con Git Bash en lugar de PowerShell: `source .venv/Scripts/activate`.

Sabes que funcionó porque aparece `(.venv)` al inicio del prompt. Si prefieres
no activar nada, cada `python` de aquí en adelante se sustituye por
`.venv\Scripts\python.exe`.

### Paso 2 — Verificar la GPU

```powershell
python check_gpu.py
```

Debe reportar `CUDA disponible: True` y el nombre de la GPU. El script no solo
pregunta si hay GPU: multiplica dos matrices de 5000×5000 para confirmar que
**realmente ejecuta**, porque `is_available()` puede dar `True` y aun así fallar
al lanzar kernels.

### Paso 3 — Entrar a `src/`

```powershell
cd src
```

Todo lo demás corre desde aquí, con `python -m`, porque las carpetas son
paquetes de Python.

### Paso 4 — Generar los datos sintéticos (~5 s)

```powershell
python -m comun.sintetico --dias 20 --espacios 2
```

Imprime un resumen por espacio y escribe `data/raw/sintetico.parquet`. Los
pasos siguientes generan sus propios datos con la misma semilla, así que este
paso sirve para inspeccionarlos; no es un requisito de los demás.

### Paso 5 — Verificar que la red aprende (~17 s)

```powershell
python -m gru.entrenar --prueba-sintetica --implementacion pytorch
```

Debe terminar con `OK: la red aprende la estructura de una serie conocida.`
Confirma que la red aprende una serie limpia cuya respuesta se conoce: si falla,
el problema está en el código y no en los datos.

**Usa siempre `--implementacion pytorch` en vivo.** Sin esa bandera corre la
celda escrita a mano, que tarda unos 2.5 minutos.

### Paso 6 — El baseline SARIMAX (~26 s)

```powershell
python -m sarimax.entrenar --dias 20 --espacios 2
```

Ajusta un modelo univariado por variable y espacio, así que 5 variables en 2
espacios son 10 ajustes independientes.

### Paso 7 — La comparación (~40 s)

```powershell
python comparar_modelos.py --dias 20 --espacios 2
```

Entrena los dos modelos y los compara sobre las mismas ventanas de prueba. Es
el resultado principal del prototipo.

Los dos se entrenan por separado y con criterios distintos: el GRU minimiza MSE
por backpropagation, SARIMAX maximiza verosimilitud por filtro de Kalman.
Ninguno alimenta al otro. Lo que comparten es el conjunto de prueba, las
particiones, las ventanas y el código de métricas, y eso es lo que hace
comparables los números. `verificar_contra_gru()` reconstruye los objetivos del
GRU y aborta si difieren, así que esa igualdad no se asume.

---

## Otros comandos

### Entrenamiento completo (~30 min en RTX 4090)

```powershell
python -m gru.entrenar --dias 60
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

```powershell
python -m gru.comparar_implementaciones
```

### Con datos reales, cuando existan

```powershell
python -m gru.entrenar --datos ../data/raw/aulas3.parquet
```

### Opciones útiles

| Opción | Para qué |
|---|---|
| `--dias N` | Días de datos sintéticos a generar |
| `--espacios N` | Espacios a simular (no aplica a `gru.entrenar`) |
| `--epocas N` | Tope de épocas (por omisión 200, pero el early stopping suele cortar antes) |
| `--paciencia N` | Épocas sin mejora antes de detener |
| `--normalizar-por-espacio` | Media y sigma por espacio en vez de globales |
| `--etiqueta nombre` | Prefijo de los archivos guardados, para comparar corridas |
| `--implementacion pytorch` | Usa `nn.GRU` en vez de la celda escrita a mano: ~14× más rápido |
| `--estacionalidad estacional` | SARIMAX con `s=288` en vez de Fourier. Inviable: 2,147 s por modelo |

### Si algo falla

| Síntoma | Qué hacer |
|---|---|
| `ModuleNotFoundError` | Estás fuera de `src/`. Los comandos van con `python -m` desde ahí |
| El prompt no dice `(.venv)` | No activaste el entorno. Vuelve al paso 1 |
| CUDA no disponible | Corre igual en CPU, solo más lento. `python check_gpu.py` diagnostica |
| Tarda demasiado | Te faltó `--implementacion pytorch` |
| SARIMAX no termina | Usaste `--estacionalidad estacional`. Quítalo |

---

## Dónde se muestran los resultados

### En pantalla

| Qué | Dónde se imprime |
|---|---|
| Tabla de métricas por variable | `metricas.imprimir()` |
| MAE por horizonte | `entrenar.main()` |
| GRU contra SARIMAX | `comparar_modelos.imprimir_comparacion()` |

### En archivos

| Archivo | Qué guarda |
|---|---|
| `models/gru.pt` | Los pesos entrenados |
| `models/gru_normalizador.json` | Media y sigma por variable |
| `models/gru_metadatos.json` | Métricas, mejor época, hiperparámetros |
| `models/sarimax_metadatos.json` | Métricas y AIC de los modelos ajustados |
| `data/processed/comparacion_gru_sarimax.json` | La comparación completa |

Ninguno de esos archivos se versiona.

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
src/comun/        contrato de datos, ventanas, normalización, métricas
src/gru/          modelo predictivo (ver src/gru/README.md)
src/sarimax/      baseline estadístico
src/comparar_modelos.py   GRU contra SARIMAX sobre las mismas ventanas
docs/             estándar de código, hallazgos de ROBOD
data/raw/         datos de entrada (no versionados)
data/processed/   datos derivados (no versionados)
models/           pesos y metadatos (no versionados)
check_gpu.py      diagnóstico de CUDA
```

Pendientes: `src/anomalias/` (detector 3σ), `src/llm/` (Llama 3.1) y
`src/api/` (FastAPI).

---

## Documentación

| Documento | Contenido |
|---|---|
| [src/gru/README.md](src/gru/README.md) | Decisiones de diseño del modelo y qué queda abierto |
| [src/sarimax/README.md](src/sarimax/README.md) | Qué es y qué no es el baseline, y por qué no `s=288` |
| [docs/estandar_codigo.md](docs/estandar_codigo.md) | Convención de documentación del código |
| [docs/hallazgos_robod.md](docs/hallazgos_robod.md) | Evaluación descartada de ROBOD y qué defectos reaparecerán con los sensores propios |

---

## Advertencia sobre los resultados

Las métricas que produce el pipeline hoy se calculan sobre datos sintéticos.
Miden qué tan bien el GRU aprende un generador escrito para este repositorio,
así que son **circulares por construcción** y no constituyen un resultado de
desempeño. Sirven para verificar que el pipeline corre y que la red aprende
estructura temporal. El desempeño real se mide con los datos de Aulas 3.
