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

- **Python 3.10 a 3.14.** Desarrollado y verificado con 3.12.10
- Git
- GPU NVIDIA con CUDA (opcional; todo corre en CPU, solo más lento)

---

## Instalación desde cero

Para una máquina donde el repositorio nunca se ha clonado.

### 1. Instalar Python

Descarga el instalador de Windows desde
[python.org/downloads](https://www.python.org/downloads/windows/).

En la primera pantalla del instalador, **marca la casilla
`Add python.exe to PATH`** antes de darle a instalar. Es la causa más común de
que después `python` no se encuentre en la terminal, y corregirlo luego es más
molesto que marcarla a tiempo. Deja también activado el `py launcher`, que viene
marcado por omisión y es lo que permite tener varias versiones conviviendo.

Comprueba que quedó, en una terminal nueva:

```powershell
py --list
python --version
```

`py --list` muestra todas las versiones instaladas. Si abriste la terminal antes
de instalar, ciérrala y abre otra: el `PATH` se lee al arrancar.

**Sobre la versión.** Este proyecto se desarrolló y se verificó con 3.12.10,
pero **3.14 también sirve**: PyTorch 2.11 soporta Python 3.10 a 3.14 en Windows,
y CUDA 12.8 —el `cu128` del paso 6— está en su lista de versiones estables.

Si alguna dependencia se queja de que no hay *wheel* para tu versión e intenta
compilar desde el código fuente, no pelees con eso en Windows: instala 3.12
junto a la que ya tienes y crea el entorno con ella. Las versiones de Python
conviven sin estorbarse, y el paso 4 explica cómo elegir cuál se usa.

> Referencia: [matriz de compatibilidad de PyTorch](https://github.com/pytorch/pytorch/blob/main/RELEASE.md#release-compatibility-matrix)

### 2. Elegir dónde va a vivir el repositorio

Muévete a la carpeta donde lo quieras antes de clonar, porque `git clone` crea
la carpeta del proyecto **dentro de donde estés parado**.

| Qué quieres | Comando |
|---|---|
| Ver en qué carpeta estás | `pwd` |
| Listar lo que hay ahí | `ls` |
| Entrar a una carpeta | `cd Proyectos` |
| Subir un nivel | `cd ..` |
| **Cambiar de disco** | `cd D:\` |
| Cambiar de disco y carpeta de una vez | `cd D:\Proyectos\Tec` |
| Ver qué discos existen | `Get-PSDrive -PSProvider FileSystem` |

**En PowerShell, `cd D:\Proyectos` cambia de disco y de carpeta en un solo
paso.** Esto es distinto de `cmd`, donde `cd` no cambia de unidad y hay que usar
`cd /d D:\Proyectos` o escribir `D:` solo en una línea. Si alguna vez te topaste
con que `cd` "no hacía nada" al cambiar de disco, era `cmd`, no PowerShell.

Si la ruta tiene espacios, va entre comillas: `cd "D:\Mis Proyectos"`.

### 3. Clonar el repositorio

```powershell
git clone git@github.com:juliordzfigueroa/gemelo_digital_A01029680.git
cd gemelo_digital_A01029680
```

Con HTTPS en lugar de SSH, si esa máquina no tiene llave configurada:

```powershell
git clone https://github.com/juliordzfigueroa/gemelo_digital_A01029680.git
```

Las carpetas `data/raw/`, `data/processed/` y `models/` ya vienen en el clon
—vacías, con un `.gitkeep`— así que no hay que crearlas a mano. Su contenido no
se versiona.

### 4. Crear el entorno virtual

```powershell
py -3.12 -m venv .venv
```

El `-3.12` elige explícitamente esa versión de entre las que tengas instaladas.
Para usar la más reciente, `py -m venv .venv`; para 3.14 en concreto,
`py -3.14 -m venv .venv`. Con `py --list` ves cuáles hay.

El `.venv` vive dentro del repositorio pero está en `.gitignore`, así que nunca
se sube. Si algo sale mal en los pasos siguientes, se puede borrar la carpeta
`.venv` y volver a empezar desde aquí sin tocar nada más.

### 5. Activar el entorno

```powershell
.\.venv\Scripts\Activate.ps1
```

| Shell | Comando |
|---|---|
| PowerShell | `.\.venv\Scripts\Activate.ps1` |
| `cmd` | `.venv\Scripts\activate.bat` |
| Git Bash | `source .venv/Scripts/activate` |

Sabes que funcionó porque aparece `(.venv)` al inicio del prompt.

**Si PowerShell contesta `running scripts is disabled on this system`**, está
bloqueando la ejecución de scripts. La salida sin tocar nada es activar desde
`cmd` o Git Bash con los comandos de la tabla. Si prefieres arreglarlo de forma
permanente, esto lo habilita solo para tu usuario y solo para scripts locales:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### 6. Instalar las dependencias

Con el entorno ya activado:

```powershell
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install pandas numpy scikit-learn statsmodels matplotlib pyarrow
```

**Cuidado con `cu128`.** Ese índice corresponde a CUDA 12.8, que es lo que
soporta el driver de la máquina donde se desarrolló; en otra computadora puede no
ser el correcto. `nvidia-smi` reporta en la esquina superior derecha la versión
de CUDA que soporta el driver instalado, y con ese número se elige el índice en
[pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

**Si la máquina no tiene GPU NVIDIA**, omite el `--index-url` por completo:

```powershell
pip install torch
```

Eso instala la build de CPU. Todo el pipeline corre igual; el entrenamiento
completo tarda bastante más, pero los siete pasos de la demo siguen siendo
cuestión de minutos.

Versiones con las que se desarrolló, por si hace falta reproducir el entorno
exacto:

| Paquete | Versión | Para qué |
|---|---|---|
| `torch` | 2.11.0+cu128 | La red neuronal y el cómputo en GPU |
| `pandas` | 3.0.6 | Series de tiempo, agregación, ventanas |
| `numpy` | 2.5.2 | Operaciones numéricas y métricas |
| `statsmodels` | 0.15.0 | SARIMAX (filtro de Kalman) |
| `scikit-learn` | 1.9.1 | Utilidades de partición |
| `pyarrow` | 25.0.1 | Leer y escribir Parquet |
| `matplotlib` | 3.11.2 | Gráficas |

### 7. Comprobar que quedó bien

```powershell
python check_gpu.py
```

Debe reportar `CUDA disponible: True` y el nombre de la GPU, o terminar en
`CUDA NO disponible` si instalaste la build de CPU, que también es un resultado
válido.

---

## Demo completa, en orden

Siete pasos, unos 90 segundos de cómputo en total.

### Paso 1 — Activar el entorno virtual

```powershell
cd <la carpeta donde está el repositorio>
.\.venv\Scripts\Activate.ps1
```

Con Git Bash: `source .venv/Scripts/activate`. Si prefieres no activar nada,
cada `python` de aquí en adelante se sustituye por `.venv\Scripts\python.exe`.

### Paso 2 — Verificar la GPU

```powershell
python check_gpu.py
```

El script no solo pregunta si hay GPU: multiplica dos matrices de 5000×5000 para
confirmar que **realmente ejecuta**, porque `is_available()` puede dar `True` y
aun así fallar al lanzar kernels.

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

Los tiempos de arriba se midieron en una RTX 4090. En CPU o en otra GPU cambian,
pero el orden de magnitud se sostiene.

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
| `python` no se reconoce | No marcaste `Add python.exe to PATH`, o no reabriste la terminal. Prueba `py --version` |
| `ModuleNotFoundError` | Estás fuera de `src/`. Los comandos van con `python -m` desde ahí |
| El prompt no dice `(.venv)` | No activaste el entorno. Vuelve al paso 1 |
| `running scripts is disabled` | PowerShell bloquea scripts. Ve a **Instalación desde cero**, paso 5 |
| `cd` no cambia de disco | Estás en `cmd`, no en PowerShell. Usa `cd /d D:\ruta` o abre PowerShell |
| `No matching distribution found for torch` | Tu versión de Python queda fuera del rango 3.10–3.14. Instala 3.12 y crea el entorno con `py -3.12` |
| `CUDA NO disponible` | Normal si instalaste la build de CPU. Si esperabas GPU, revisa que el índice `cuXXX` corresponda a tu driver |
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
