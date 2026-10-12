# Gemelo Digital Aulas 3

Prototipo técnico del gemelo digital del edificio Aulas 3, Tec de Monterrey
Campus Santa Fe.

Julio César Rodríguez Figueroa — A01029680

---

## Qué hay ahora

| Componente | Qué hace | Estado |
|---|---|---|
| **GRU** | Predice CO₂, temperatura, humedad, luz y movimiento hasta 2 h adelante | listo |
| **SARIMAX** | Baseline estadístico contra el que se compara el GRU | listo |
| **LLM** | Llama 3.1 8B vía Ollama: redacta respuestas con los datos del GRU | listo, falta probar con Ollama |
| **API** | FastAPI: recibe la pregunta, consulta al GRU, le pide la frase al LLM | lista, con datos sintéticos |
| Detector de anomalías 3σ | | pendiente |

El GRU es de dos capas [128, 64], con un ID de espacio como rasgo para que un
solo modelo sirva a todo el edificio. Su celda recurrente está escrita a mano
siguiendo las ecuaciones 1 a 4 de la sección 4.2.7.2 del documento de
investigación.

Los sensores de Aulas 3 todavía no existen. Todo se desarrolla contra **datos
sintéticos** con la misma forma y los mismos defectos que tendrán los reales; el
día que lleguen, lo que cambia es el argumento `--datos` y la fuente de la API.

---

## Instalación

Tres partes, en orden. La primera es para todo; las otras dos, según lo que
quieras correr.

| Para correr… | Necesitas |
|---|---|
| Entrenamiento, predicción, SARIMAX | 1 y 2 |
| La API sin LLM (`/contexto`) | 1 y 2 |
| La API completa (`/chat`) | 1, 2 y 3 |

### 1. Base: Python, repositorio y entorno

**Python 3.10 a 3.14** (desarrollado con 3.12.10). Instalador de
[python.org](https://www.python.org/downloads/windows/): en la primera pantalla
**marca `Add python.exe to PATH`**. Comprueba en una terminal nueva:

```powershell
py --list
```

Muévete a donde quieras el repositorio y clónalo. En PowerShell, `cd D:\ruta`
cambia de disco y de carpeta en un solo paso.

```powershell
cd D:\Proyectos
git clone git@github.com:juliordzfigueroa/gemelo_digital_A01029680.git
cd gemelo_digital_A01029680
```

Crea y activa el entorno virtual:

```powershell
py -3.12 -m venv .venv           # o py -3.14, según la que tengas
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Funcionó si el prompt empieza con `(.venv)`. Si PowerShell contesta
`running scripts is disabled on this system`:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 2. Componente GRU

**Dependencias.** PyTorch va aparte porque su versión con CUDA vive en otro
índice:

```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

`cu128` corresponde a CUDA 12.8. Si la otra máquina tiene otro driver,
`nvidia-smi` dice qué versión de CUDA soporta y
[pytorch.org](https://pytorch.org/get-started/locally/) da el índice. **Sin GPU
NVIDIA**, usa `pip install torch` a secas: todo corre igual en CPU, más lento.

**Comprobar la GPU:**

```powershell
python check_gpu.py
```

**Entrenar el modelo.** Este paso es obligatorio en un clon nuevo: los pesos no
se versionan, así que `models/` llega vacío y sin modelo no hay predicción ni
API.

```powershell
cd src
python -m gru.entrenar --dias 60 --implementacion pytorch
```

Deja tres archivos en `models/`: `gru.pt`, `gru_normalizador.json` y
`gru_metadatos.json`. Los tres son necesarios: sin el normalizador las salidas
no se pueden convertir a ppm ni a grados.

### 3. Componente LLM (Llama 3.1 vía Ollama)

Ollama es un programa aparte, no un paquete de pip. Instálalo desde
[ollama.com/download/windows](https://ollama.com/download/windows); queda
corriendo como servicio en segundo plano.

```powershell
ollama --version
ollama pull llama3.1:8b          # ~5 GB, una sola vez
```

Comprueba que responde, desde `src/`:

```powershell
python -m llm.cliente_llm
```

Debe terminar en `OK: Ollama respondió` seguido de una frase en español. Si
Ollama no está corriendo, el mismo comando dice qué falta y termina sin
tronar. Para arrancarlo a mano: `ollama serve`.

El modelo corre local: ninguna lectura sale de la máquina.

---

## Cómo probar que predice

Todo desde `src/`, con el entorno activado y el modelo ya entrenado.

### 1. Que la red aprende

```powershell
python -m gru.entrenar --prueba-sintetica --implementacion pytorch
```

Entrena con una serie limpia cuya respuesta se conoce. Debe terminar con
`OK: la red aprende la estructura de una serie conocida.` Si falla, el problema
está en el código, no en los datos.

### 2. Qué tan bien predice

El entrenamiento completo ya lo mide: al terminar imprime el MAE, RMSE y R² por
variable sobre el conjunto de prueba —ventanas que la red nunca vio— y el MAE
por horizonte, de +5 a +30 min. Esos números también quedan en
`models/gru_metadatos.json`.

### 3. Una predicción concreta

```powershell
python -m gru.predecir --espacio 1 --horizonte 30      # directa, 30 min
python -m gru.predecir --espacio 3 --horizonte 120     # iterativa, 2 h
```

Imprime la lectura actual y la predicción paso a paso. Esto muestra que el
modelo **corre**, no que acierta: para eso está el paso 2.

### 4. A través de la API, sin LLM

```powershell
uvicorn api.api:app --port 8000
```

Abre [localhost:8000/docs](http://localhost:8000/docs) y prueba `POST /contexto`
con:

```json
{"pregunta": "¿Cómo va a estar el CO2 del 3302 en una hora?", "espacio_id": 1}
```

Devuelve la intención detectada, la predicción del GRU y el prompt que se le
mandaría a Llama. No necesita Ollama.

---

## Pasos de prueba para el entrenamiento

Desde `src/`.

```powershell
python -m comun.sintetico --dias 20 --espacios 2                     # generar e inspeccionar datos
python -m gru.entrenar --prueba-sintetica --implementacion pytorch   # la red aprende
python -m sarimax.entrenar --dias 20 --espacios 2                    # baseline
python comparar_modelos.py --dias 20 --espacios 2                    # GRU contra SARIMAX
```

La comparación entrena los dos modelos y los evalúa sobre las mismas ventanas
de prueba. Se entrenan con criterios distintos —el GRU minimiza MSE, SARIMAX
maximiza verosimilitud— y ninguno alimenta al otro; lo que comparten son las
particiones, las ventanas y el código de métricas.
`verificar_contra_gru()` aborta si las ventanas no coinciden.

**Usa `--implementacion pytorch`** salvo que quieras la celda escrita a mano,
que es ~14× más lenta.

---

## Datos reales

Las exportaciones de los sensores van a `data/raw/` (no se versionan). Desde
`src/`:

```powershell
python -m ingesta.limpieza "../data/raw/1 oct a 11 octubre 2026.csv"   # acepta varios archivos
python -m ingesta.horarios "../data/raw/horarios.csv"                  # horario contra sensores
python -m gru.entrenar --datos ../data/processed/aulas3.parquet        # entrenar con datos reales
```

La limpieza escribe `data/processed/aulas3.parquet` en el formato del contrato
de datos y un `aulas3_reporte.json` con cada lectura descartada y por qué. Qué
dispositivo mide qué espacio está en [config/sensores.csv](config/sensores.csv),
y los espacios del edificio en [config/espacios.json](config/espacios.json).

Diferencias con el contrato que hay que tener presentes:

| Variable | En los sensores reales |
|---|---|
| `co2` | eCO₂ **estimado** por un sensor MOX, no CO₂ medido: sube con productos de limpieza y no baja a ~420 ppm de noche |
| `luz` | porcentaje (0–100), no lux |
| `movimiento` | fracción del intervalo de 5 min con detección, entre 0 y 1 |

---

## La API

```powershell
uvicorn api.api:app --port 8000
```

| Endpoint | Para qué |
|---|---|
| `GET /health` | Estado del GRU, de los datos y de Ollama |
| `POST /chat` | Pregunta en lenguaje natural, respuesta redactada por Llama |
| `POST /contexto` | Lo mismo **sin** llamar a Llama: devuelve el prompt |
| `GET /docs` | Interfaz para probar los endpoints desde el navegador |

Qué está conectado y qué no:

| Pieza | Estado |
|---|---|
| GRU, predicción directa e iterativa | real, desde `models/gru.pt` |
| Clasificación de la pregunta y prompts | real |
| Fuente de datos | **sintética**, en memoria (`TODO E5` en `api/api.py`) |
| Detector de anomalías | **no existe**; la API lo dice en vez de responder que no hay nada |

Diseño de esta capa en [src/llm/README.md](src/llm/README.md).

---

## Otros comandos

```powershell
python -m gru.comparar_implementaciones                  # celda a mano contra nn.GRU
python -m gru.entrenar --datos ../data/raw/aulas3.parquet  # con datos reales
python -m llm.clasificador                               # cómo se rutean las preguntas
python -m llm.contexto                                   # los prompts por tipo
```

| Opción | Para qué |
|---|---|
| `--dias N` | Días de datos sintéticos |
| `--espacios N` | Espacios a simular (no aplica a `gru.entrenar`, que usa 5) |
| `--epocas N` / `--paciencia N` | Tope de épocas y épocas sin mejora antes de parar |
| `--implementacion pytorch` | `nn.GRU` en vez de la celda a mano |
| `--normalizar-por-espacio` | Media y sigma por espacio en vez de globales |
| `--etiqueta nombre` | Prefijo de los archivos guardados |
| `--estacionalidad estacional` | SARIMAX con `s=288`. Inviable: 2,147 s por modelo |

---

## Si algo falla

| Síntoma | Qué hacer |
|---|---|
| `python` no se reconoce | No marcaste `Add python.exe to PATH`, o no reabriste la terminal |
| `running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `ModuleNotFoundError` | Estás fuera de `src/`, o el prompt no dice `(.venv)` |
| `No matching distribution found for torch` | Python fuera del rango 3.10–3.14 |
| `CUDA NO disponible` | Normal con la build de CPU; si esperabas GPU, revisa el índice `cuXXX` |
| `models/gru.pt does not exist` | Clon nuevo: entrena primero (Instalación, paso 2) |
| `gru_cargado: false` en `/health` | Lo mismo: falta entrenar |
| `No se puede conectar con Ollama` | Ollama no está instalado o no corre: `ollama serve` |
| `Ollama no tiene el modelo` | `ollama pull llama3.1:8b` |
| Tarda demasiado | Te faltó `--implementacion pytorch` |

---

## El contrato de datos

Cualquier fuente de datos debe entregar un DataFrame con estas columnas:

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
src/gru/          modelo, entrenamiento y predicción
src/sarimax/      baseline estadístico
src/llm/          clasificador, prompts y cliente de Ollama
src/api/          orquestador FastAPI
src/comparar_modelos.py   GRU contra SARIMAX
docs/             estándares y hallazgos de ROBOD
data/, models/    datos y pesos (no versionados)
requirements.txt  dependencias, menos torch
check_gpu.py      diagnóstico de CUDA
```

## Documentación

| Documento | Contenido |
|---|---|
| [src/gru/README.md](src/gru/README.md) | Decisiones de diseño del GRU y qué queda abierto |
| [src/llm/README.md](src/llm/README.md) | La capa conversacional: reparto de trabajo con el LLM |
| [src/sarimax/README.md](src/sarimax/README.md) | El baseline, y por qué no `s=288` |
| [docs/estandar_codigo.md](docs/estandar_codigo.md) | Convención de documentación del código |
| [docs/estandar_commits.md](docs/estandar_commits.md) | Convención de mensajes de commit |
| [docs/hallazgos_robod.md](docs/hallazgos_robod.md) | Evaluación descartada de ROBOD |

---

## Advertencia sobre los resultados

Las métricas de hoy se calculan sobre datos sintéticos: miden qué tan bien el
GRU aprende un generador escrito para este repositorio, así que son
**circulares por construcción** y no son un resultado de desempeño. Sirven para
verificar que el pipeline corre y que la red aprende estructura temporal. El
desempeño real se mide con los datos de Aulas 3.
