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
| **API** | FastAPI: recibe la pregunta, consulta al GRU, le pide la frase al LLM | lista, con datos reales limpios |
| Detector de anomalías 3σ | | pendiente |

El GRU es de dos capas [128, 64], con un ID de espacio como rasgo para que un
solo modelo sirva a todo el edificio. Su celda recurrente está escrita a mano
siguiendo las ecuaciones 1 a 4 de la sección 4.2.7.2 del documento de
investigación.

Todo se entrena y se evalúa con **datos reales** de los sensores de Aulas 3
(desde el 1 de octubre de 2026): cuatro salones útiles, 3206, 3208, 3305 y
3308. La API los lee del archivo limpio mientras no haya acceso a la base de
datos del Tec.

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

**Datos y modelo.** Obligatorio en un clon nuevo: ni los datos de los sensores
ni los pesos se versionan, así que `data/` y `models/` llegan vacíos. Copia la
exportación de los sensores y el horario a `data/raw/`, y luego:

```powershell
cd src
python -m ingesta.limpieza "../data/raw/1 oct a 11 octubre 2026.csv"
python -m gru.entrenar --implementacion pytorch
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

## Prueba local completa

Todo en orden, desde `src/` con el entorno activado. Necesita la exportación
de los sensores y el horario en `data/raw/`.

```powershell
cd src
```

### 1. Limpiar los datos

```powershell
python -m ingesta.limpieza "../data/raw/1 oct a 11 octubre 2026.csv"
```

Acepta varios archivos a la vez. Escribe `data/processed/aulas3.parquet` y
`aulas3_reporte.json`, con cada valor descartado y por qué. Al final imprime
una tabla por espacio; hoy hay cuatro útiles: 206, 208, 305 y 308.

Después, el clima exterior de esas mismas fechas:

```powershell
python -m ingesta.clima
```

Descarga de [Open-Meteo](https://open-meteo.com/) (gratuito, sin clave) la
temperatura, humedad, radiación solar y nubosidad del campus, hora por hora,
y lo deja en intervalos de 5 minutos en `data/processed/clima.parquet`. Imprime
la comparación interior contra exterior por salón. Necesita internet.

### 2. Que la red aprende

```powershell
python -m gru.entrenar --prueba-sintetica --implementacion pytorch
```

Entrena con una serie determinista, sin ruido ni huecos, cuya respuesta se
conoce. Debe terminar en `OK: la red aprende la estructura de una serie
conocida.` Si falla, el problema está en el código, no en los datos.

### 3. Entrenar y medir

```powershell
python -m gru.entrenar --implementacion pytorch
```

Imprime MAE, RMSE y R² por variable sobre el día de prueba —un día entre
semana que la red nunca vio— y el MAE por horizonte, de +5 a +30 min. Deja el
modelo en `models/`.

### 4. Una predicción concreta

```powershell
python -m gru.predecir --espacio 308 --horizonte 30     # directa, 30 min
python -m gru.predecir --espacio 206 --horizonte 120    # iterativa, 2 h
```

Toma la última hora de lecturas consecutivas del archivo y predice. "Ahora"
es la última lectura del archivo, no la hora actual. Esto muestra que el
modelo **corre**; qué tan bien acierta lo dice el paso 3.

### 5. Contra el baseline

```powershell
python comparar_modelos.py
```

Entrena el GRU y ajusta SARIMAX sobre las mismas ventanas de prueba, y aborta
si no coinciden. Con una semana de datos SARIMAX gana; ver la advertencia al
final.

### 6. El horario contra los sensores

```powershell
python -m ingesta.horarios ../data/raw/horarios.csv
```

Compara movimiento y eCO₂ dentro y fuera de los bloques de clase, por salón y
día.

### 7. La API

```powershell
uvicorn api.api:app --port 8000
```

Abre [localhost:8000/docs](http://localhost:8000/docs) en el navegador y prueba
desde ahí:

- `POST /contexto` muestra la intención detectada, los datos y el prompt que
  recibiría Llama. **No necesita Ollama.**
- `POST /chat` devuelve la respuesta redactada por Llama. Necesita Ollama
  corriendo.

Con un cuerpo como:

```json
{"pregunta": "¿Cómo va a estar el CO2 del 3308 en una hora?"}
```

Los salones con datos son 3206, 3208, 3305 y 3308. Para comprobar desde la
terminal que la API está arriba:

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Para preguntas, usa `/docs`: en PowerShell 5.1, `curl` es otro comando y los
acentos de la pregunta se corrompen al mandarlos desde la terminal.

---

## Datos reales

Qué dispositivo mide qué espacio está en [config/sensores.csv](config/sensores.csv),
y los espacios del edificio en [config/espacios.json](config/espacios.json).
Diferencias con el contrato de datos que hay que tener presentes:

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
| Fuente de datos | archivo limpio `aulas3.parquet`; la base de datos del Tec queda pendiente (`TODO E5` en `api/api.py`) |
| Detector de anomalías | **no existe**; la API lo dice en vez de responder que no hay nada |

Diseño de esta capa en [src/llm/README.md](src/llm/README.md).

---

## Otros comandos

```powershell
python -m gru.comparar_implementaciones                  # celda a mano contra nn.GRU
python -m llm.clasificador                               # cómo se rutean las preguntas
python -m llm.contexto                                   # los prompts por tipo
```

| Opción | Para qué |
|---|---|
| `--datos ruta` | Otro archivo con el contrato de datos; por omisión `data/processed/aulas3.parquet` |
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
| `luz` | float | % (0–100) |
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

Las métricas de hoy salen de **una semana de datos reales**: unos 4 días de
entrenamiento, un jueves de validación y un viernes de prueba. Sirven para
verificar que el pipeline funciona con sensores reales, no como resultado de
desempeño. Con esta cantidad de datos SARIMAX le gana al GRU en las cinco
variables; la comparación que vale es la que se haga con varias semanas.

La validación y la prueba se toman de días completos entre semana
(`MODO_PARTICION = "por_dias"` en `config.py`), porque un corte cronológico
sobre una sola semana dejaba el fin de semana al final y calificaba al modelo
con el edificio vacío.
