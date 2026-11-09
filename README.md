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
| **LLM** | Llama 3.1 8B vía Ollama: redacta respuestas con los datos del GRU | listo; con fallas de redacción conocidas |
| **API** | FastAPI: recibe la pregunta, consulta al GRU, le pide la frase al LLM | lista, con datos reales limpios |
| **Visor** | Potree en el navegador: un escaneo por piso, marcadores por salón y chat | listo; faltan los marcadores de los salones y corregir la altura entre pisos |
| **Servidor de escaneos** | Sirve las nubes convertidas, aparte de la API | listo |
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

Cuatro partes, en orden. La primera es para todo; las demás, según lo que
quieras correr.

| Para correr… | Necesitas |
|---|---|
| Entrenamiento, predicción, SARIMAX | 1 y 2 |
| La API sin LLM (`/context`) | 1 y 2 |
| La API completa (`/chat`) | 1, 2 y 3 |
| El visor con los escaneos | 1, 2, 3 y 4 |

### La máquina

| Pieza | Mínimo | Para qué |
|---|---|---|
| Windows 10 u 11 | | todo lo probado es en Windows |
| GPU NVIDIA, 8 GB de VRAM o más | recomendada | Llama 3.1 8B ocupa unos 5 GB. Sin GPU todo corre en CPU, pero cada respuesta de Llama tarda mucho más |
| RAM | 16 GB | el entrenamiento y Ollama al mismo tiempo |
| Disco | ~15 GB, más 134 GB si las nubes van en esa máquina | entorno con torch (~5 GB), modelo de Llama (~5 GB), nubes convertidas |
| Navegador | Chrome o Edge | el visor usa WebGL |

### Lo que no viene con `git clone`

El repositorio trae el código y la configuración. Esto hay que copiarlo,
descargarlo o generarlo en cada máquina:

| Qué | Dónde va | Cómo se consigue |
|---|---|---|
| Exportación de los sensores (CSV) y horario (`schedule.csv`) | `data/raw/` | copiarlos de la máquina de desarrollo |
| Datos limpios y clima | `data/processed/` | se generan: `ingestion.cleaning` e `ingestion.weather` (paso 2) |
| Modelo entrenado | `models/` | se entrena en unos minutos (paso 2), o se copia la carpeta |
| Modelo de Llama | dentro de Ollama | `ollama pull llama3.1:8b` (paso 3) |
| Librería de Potree | `web/potree/` | se descarga (paso 4) |
| Nubes convertidas (134 GB) | `D:\escaneos` o la carpeta de `SCANS_DIR` | copiar la carpeta, o servirlas desde otra máquina por red (ver *Nubes en otra máquina*, en [Variables de entorno](#variables-de-entorno)) |
| LAS de cada piso (136 GB) | cualquier carpeta | solo si se van a volver a convertir |

### PowerShell o Git Bash

Funciona en las dos terminales. Los comandos de Python, `pip`, `ollama` y
`uvicorn` son **idénticos**; solo cambia lo de esta tabla. Donde un paso es
distinto, el README trae las dos versiones; donde no, un solo bloque sirve
para ambas.

| Para… | PowerShell | Git Bash |
|---|---|---|
| Cambiar de disco y carpeta | `cd D:\Proyectos` | `cd /d/Proyectos` |
| Activar el entorno | `.\.venv\Scripts\Activate.ps1` | `source .venv/Scripts/activate` |
| Acentos en los mensajes | no hace falta | `export PYTHONUTF8=1` (una vez por consola) |
| Definir una variable de entorno | `$env:SCANS_DIR = "E:\escaneos"` | `export SCANS_DIR=E:/escaneos` |
| Pasar una ruta a un comando | `D:/las_A3` o `D:\las_A3` | `D:/las_A3` (con `\` y sin comillas se rompe) |
| Consultar la API | `Invoke-RestMethod http://localhost:8000/health` | `curl http://localhost:8000/health` |

Las rutas de los comandos de este README van con `/`, que funcionan igual en
las dos.

### 1. Base: Python, repositorio y entorno

**Git** desde [git-scm.com](https://git-scm.com/download/win); con las
opciones por omisión instala también Git Bash.

**Python 3.10 a 3.14** (desarrollado con 3.12.10). Instalador de
[python.org](https://www.python.org/downloads/windows/): en la primera pantalla
**marca `Add python.exe to PATH`**. Con 3.14 todas las versiones de
`requirements.txt` y torch con CUDA tienen instalador para Windows (comprobado
el 5 de noviembre de 2026). Comprueba en una terminal nueva:

```powershell
git --version
py --list
```

**Clonar.** Los comandos de abajo usan SSH (`git@github.com:…`), que necesita
una clave SSH de esa máquina registrada en GitHub. Si no la tiene, clona por
HTTPS y Git pide iniciar sesión en el navegador la primera vez:

```
git clone https://github.com/juliordzfigueroa/gemelo_digital_A01029680.git
```

Muévete a donde quieras el repositorio, clónalo, y crea y activa el entorno
virtual.

**PowerShell:**

```powershell
cd D:\Proyectos
git clone git@github.com:juliordzfigueroa/gemelo_digital_A01029680.git
cd gemelo_digital_A01029680
py -3.12 -m venv .venv           # o py -3.14, según la que tengas
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Si contesta `running scripts is disabled on this system`, una sola vez:
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

**Git Bash:**

```bash
cd /d/Proyectos
git clone git@github.com:juliordzfigueroa/gemelo_digital_A01029680.git
cd gemelo_digital_A01029680
py -3.12 -m venv .venv           # o py -3.14, según la que tengas
source .venv/Scripts/activate
export PYTHONUTF8=1
python -m pip install --upgrade pip
```

Funcionó si el prompt muestra `(.venv)`.

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

`requirements.txt` ya incluye lo que usa el módulo de clima (`httpx`), así que
no hay que instalar nada más para él. Lo único que necesita es **acceso a
internet** para consultar [Open-Meteo](https://open-meteo.com/), que es
gratuito y no pide registro ni clave. Si la red del campus bloquea el acceso,
el resto del sistema funciona igual; solo faltarán los datos de clima.

**Comprobar la GPU:**

```powershell
python check_gpu.py
```

**Datos y modelo.** Obligatorio en un clon nuevo: ni los datos de los sensores
ni los pesos se versionan, así que `data/` y `models/` llegan vacíos. Copia la
exportación de los sensores y el horario a `data/raw/`, y luego:

```powershell
cd src
python -m ingestion.cleaning "../data/raw/1 oct a 11 octubre 2026.csv"
python -m gru.train --implementation pytorch
```

Deja tres archivos en `models/`: `gru.pt`, `gru_normalizer.json` y
`gru_metadata.json`. Los tres son necesarios: sin el normalizador las salidas
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
python -m llm.llm_client
```

Debe terminar en `OK: Ollama respondió` seguido de una frase en español. Si
Ollama no está corriendo, el mismo comando dice qué falta y termina sin
tronar. Para arrancarlo a mano: `ollama serve`.

El modelo corre local: ninguna lectura sale de la máquina.

### 4. Componente visor (Potree)

Dos descargas, ninguna por pip y ninguna necesita permisos de administrador.

**La librería de Potree 1.8.2**, que usa la página del visor. No se versiona
(pesa 65 MB); va en `web/potree/`. Desde la raíz del repo.

PowerShell:

```powershell
Invoke-WebRequest https://github.com/potree/potree/releases/download/1.8.2/Potree_1.8.2.zip -OutFile $env:TEMP\potree.zip
Expand-Archive $env:TEMP\potree.zip $env:TEMP\potree -Force
New-Item -ItemType Directory -Force web\potree
Copy-Item $env:TEMP\potree\Potree_1.8.2\build, $env:TEMP\potree\Potree_1.8.2\libs, $env:TEMP\potree\Potree_1.8.2\LICENSE web\potree -Recurse -Force
```

Git Bash:

```bash
curl -L -o /tmp/potree.zip https://github.com/potree/potree/releases/download/1.8.2/Potree_1.8.2.zip
unzip -q -o /tmp/potree.zip -d /tmp/potree
mkdir -p web/potree
cp -r /tmp/potree/Potree_1.8.2/build /tmp/potree/Potree_1.8.2/libs /tmp/potree/Potree_1.8.2/LICENSE web/potree/
```

Debe quedar `web/potree/build/potree/potree.js`.

**PotreeConverter 2.1.5**, que convierte los LAS a formato Potree. Solo hace
falta en la máquina donde se convierten los escaneos.

PowerShell:

```powershell
Invoke-WebRequest https://github.com/potree/PotreeConverter/releases/download/2.1.5/PotreeConverter_2.1.5_x64_windows.zip -OutFile $env:TEMP\pc.zip
Expand-Archive $env:TEMP\pc.zip D:\tools\PotreeConverter -Force
D:\tools\PotreeConverter\PotreeConverter_2.1.5_x64_windows\PotreeConverter.exe --help
```

Git Bash:

```bash
curl -L -o /tmp/pc.zip https://github.com/potree/PotreeConverter/releases/download/2.1.5/PotreeConverter_2.1.5_x64_windows.zip
unzip -q -o /tmp/pc.zip -d /d/tools/PotreeConverter
/d/tools/PotreeConverter/PotreeConverter_2.1.5_x64_windows/PotreeConverter.exe --help
```

Si lo pones en otra ruta, define `POTREE_CONVERTER` (ver
[Variables de entorno](#variables-de-entorno)).

Los pasos para exportar, convertir y servir los escaneos están en
[Escaneos y visor](#escaneos-y-visor).

---

## Arrancar el sistema

Con todo instalado, el uso diario son **dos consolas** (más una opcional
para ver a Llama) y el navegador. Ollama
no necesita consola: corre como servicio de Windows.

```
Navegador (visor) ──► API :8000 ──► clasifica la pregunta ──► GRU (pronóstico) o última lectura
                         │                                 └─► Llama vía Ollama redacta la respuesta
                         └── nubes de puntos ◄── servidor de escaneos :8100
```

Las rutas de abajo son las de la máquina de desarrollo; en otra, usa la
carpeta donde clonaste el repositorio.

**Consola 1, servidor de escaneos.** PowerShell:

```powershell
cd C:\Users\A01029680\Desktop\RepositorioJCRF\gemelo_digital_A01029680
.\.venv\Scripts\Activate.ps1
cd src
python -m scans.server
```

Git Bash:

```bash
cd /c/Users/A01029680/Desktop/RepositorioJCRF/gemelo_digital_A01029680
source .venv/Scripts/activate
export PYTHONUTF8=1
cd src
python -m scans.server
```

Debe listar las cinco nubes. Con el visor abierto verás muchas líneas
`206 Partial Content`: son las peticiones por rangos de Potree, es normal.

**Consola 2, API y visor.** Igual que la consola 1, cambiando la última línea:

```
uvicorn api.api:app --port 8000
```

Debe decir `GRU cargado` y `Datos en memoria`. Ahí aparece cada pregunta que
llega al chat.

**Consola 3, Llama (opcional).** Ollama ya corre como servicio, así que no hay
que arrancarlo; esta consola solo muestra su registro en vivo: cada respuesta
que redacta (`POST "/api/chat"`), cuánto tardó y a cuántos tokens por segundo.
No necesita el entorno activado. Git Bash:

```bash
tail -f "$LOCALAPPDATA/Ollama/server.log"
```

PowerShell:

```powershell
Get-Content "$env:LOCALAPPDATA\Ollama\server.log" -Wait -Tail 20
```

Para correr Ollama **en** esa consola en lugar de como servicio: cierra el
ícono de Ollama en la bandeja de Windows (clic derecho, *Quit Ollama*) y corre
`ollama serve`. Lo que Llama recibe exactamente (el prompt) no sale en este
registro; para verlo está `POST /context` en [localhost:8000/docs](http://localhost:8000/docs).

**Navegador:** [localhost:8000](http://localhost:8000). Pregunta nombrando
el salón, por ejemplo "¿cómo está el CO₂ del 3308?" (lectura actual) o "¿cómo
va a estar el CO₂ del 3308 en 30 minutos?" (pronóstico del GRU). Si no nombras
salón ni piso, te pregunta cuál. También por piso:
"¿cómo está el piso 3?" responde con un resumen del piso y después salón por
salón, y deja visible solo ese piso; "todos los salones" hace lo mismo con el
edificio completo. Bajo cada respuesta aparece qué tipo de
pregunta detectó, el salón o piso y, si hubo pronóstico, el método. Todo lo
que se le puede preguntar, y lo que todavía no, está en
[src/llm/README.md](src/llm/README.md#qué-se-le-puede-preguntar). Para apagar,
`Ctrl+C` en cada consola.

---

## Prueba local completa

Todo en orden, desde `src/` con el entorno activado. Necesita la exportación
de los sensores y el horario en `data/raw/`.

```powershell
cd src
```

### 1. Limpiar los datos

```powershell
python -m ingestion.cleaning "../data/raw/1 oct a 11 octubre 2026.csv"
```

Acepta varios archivos a la vez. Escribe `data/processed/aulas3.parquet` y
`aulas3_report.json`, con cada valor descartado y por qué. Al final imprime
una tabla por espacio; hoy hay cuatro útiles: 206, 208, 305 y 308.

Después, el clima exterior de esas mismas fechas:

```powershell
python -m ingestion.weather
```

Descarga de [Open-Meteo](https://open-meteo.com/) (gratuito, sin clave) la
temperatura, humedad, radiación solar y nubosidad del campus, hora por hora,
y lo deja en intervalos de 5 minutos en `data/processed/weather.parquet`. Imprime
la comparación interior contra exterior por salón. Necesita internet.

### 2. Que la red aprende

```powershell
python -m gru.train --known-series-test --implementation pytorch
```

Entrena con una serie determinista, sin ruido ni huecos, cuya respuesta se
conoce. Debe terminar en `OK: la red aprende la estructura de una serie
conocida.` Si falla, el problema está en el código, no en los datos.

### 3. Entrenar y medir

```powershell
python -m gru.train --implementation pytorch
```

Imprime MAE, RMSE y R² por variable sobre el día de prueba —un día entre
semana que la red nunca vio— y el MAE por horizonte, de +5 a +30 min. Deja el
modelo en `models/`.

### 4. Una predicción concreta

```powershell
python -m gru.predict --space 308 --horizon 30     # directa, 30 min
python -m gru.predict --space 206 --horizon 120    # iterativa, 2 h
```

Toma la última hora de lecturas consecutivas del archivo y predice. "Ahora"
es la última lectura del archivo, no la hora actual. Esto muestra que el
modelo **corre**; qué tan bien acierta lo dice el paso 3.

### 5. Contra el baseline

```powershell
python compare_models.py
```

Entrena el GRU y ajusta SARIMAX sobre las mismas ventanas de prueba, y aborta
si no coinciden. Con una semana de datos SARIMAX gana; ver la advertencia al
final.

### 6. El horario contra los sensores

```powershell
python -m ingestion.schedule ../data/raw/schedule.csv
```

Compara movimiento y eCO₂ dentro y fuera de los bloques de clase, por salón y
día.

### 7. La API

```powershell
uvicorn api.api:app --port 8000
```

Abre [localhost:8000/docs](http://localhost:8000/docs) en el navegador y prueba
desde ahí:

- `POST /context` muestra la intención detectada, los datos y el prompt que
  recibiría Llama. **No necesita Ollama.**
- `POST /chat` devuelve la respuesta redactada por Llama. Necesita Ollama
  corriendo.

Con un cuerpo como:

```json
{"question": "¿Cómo va a estar el CO2 del 3308 en una hora?"}
```

Los salones con datos son 3206, 3208, 3305 y 3308. Para comprobar desde la
terminal que la API está arriba: `Invoke-RestMethod http://localhost:8000/health`
en PowerShell, `curl http://localhost:8000/health` en Git Bash.

Para preguntas, usa `/docs` o el visor: desde la terminal los acentos de la
pregunta se corrompen con facilidad.

### 8. El visor

En otra consola, también desde `src/` y con el entorno activado:

```powershell
python -m scans.server
```

Con la API corriendo, abre [localhost:8000](http://localhost:8000). La línea
bajo el título dice cuántos pisos cargó; los que no están convertidos aparecen
como "sin convertir". Pregunta en el panel nombrando un salón o un piso: la
pregunta va a `/chat`. Detalle en
[Arrancar el sistema](#arrancar-el-sistema).

---

## Datos reales

Qué dispositivo mide qué espacio está en [config/sensors.csv](config/sensors.csv),
y los espacios del edificio en [config/spaces.json](config/spaces.json).
Diferencias con el contrato de datos que hay que tener presentes:

| Variable | En los sensores reales |
|---|---|
| `co2` | eCO₂ **estimado** por un sensor MOX, no CO₂ medido: sube con productos de limpieza y no baja a ~420 ppm de noche |
| `light` | porcentaje (0–100), no lux |
| `motion` | fracción del intervalo de 5 min con detección, entre 0 y 1 |

---

## La API

```powershell
uvicorn api.api:app --port 8000
```

| Endpoint | Para qué |
|---|---|
| `GET /health` | Estado del GRU, de los datos y de Ollama |
| `POST /chat` | Pregunta en lenguaje natural, respuesta redactada por Llama |
| `POST /context` | Lo mismo **sin** llamar a Llama: devuelve el prompt |
| `GET /spaces` | Pisos, salones con marcador y dimensiones, y la última lectura de cada uno; lo usa el visor |
| `GET /spaces/registry` | `config/spaces.json` tal cual, para la descarga del visor |
| `GET /viewer/` | La página del visor (`/` redirige aquí) |
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

## Escaneos y visor

```
Leica ──► un LAS por piso ──► scans.transform ──► PotreeConverter ──► D:\escaneos\A3-P0 … A3-P4
           (D:\las_A3)        (D:\las_A3_alineado)                     (metadata.json, hierarchy.bin, octree.bin)

Navegador ── página, /chat, /spaces ──► API               :8000
         └── nubes de puntos ─────────► servidor escaneos  :8100   (proceso aparte)
```

El servidor de escaneos va **aparte de la API** a propósito: las nubes pesan
decenas de GB y se quedan en la máquina con el disco, mientras la API y los
modelos pueden irse a la nube del Tec.

### 1. Exportar cada piso desde Leica

Los e57 de `D:\e57_A3` no sirven directo: PotreeConverter solo lee LAS y LAZ.
Desde el software de Leica, exporta **cada piso por separado** como LAS:

- **Con color (RGB).**
- Sin submuestrear: Potree arma los niveles de detalle solo.
- Con el nombre del e57 de ese piso, `Aulas_3.0.las` … `Aulas_3.4.las`, en una
  sola carpeta, por ejemplo `D:\las_A3`.

Un LAS pesa más o menos lo mismo que el e57 del que sale (de 12 a 41 GB por
piso, 136 GB los cinco). La copia alineada ocupa otro tanto y la conversión
otro más: cuenta con unos 420 GB libres en el disco D.

> **Si el proceso de Leica escribe en C.** El software de Leica usa C para sus
> temporales aunque el destino sea D. Si C no tiene espacio, exporta piso por
> piso en vez de todo junto.

### 2. Alinear los pisos

Cada piso se registró por separado en Leica y salió con su propio marco en
planta: las alturas coinciden, pero en planta cada piso queda desplazado y
dos de ellos girados unos 65°. El e57 con los cinco pisos juntos
(`D:\e57_A3\Aulas_3_Completo 1.e57`) sí los tiene en un marco común. El giro y
el desplazamiento que llevan cada piso a ese marco están medidos y guardados
en el campo `transform` de cada piso en [config/spaces.json](config/spaces.json).
Se midieron registrando un corte horizontal de cada piso contra el mismo corte
del combinado; el error mediano (`residual_cm`) es de 2 a 5 mm.

Desde `src/`:

```powershell
python -m scans.transform --floor all --input-dir D:/las_A3 --output-dir D:/las_A3_alineado
```

Escribe copias nuevas y no toca los originales; tarda unos 11 minutos para
los cinco. Si se vuelve a exportar un piso desde Leica **sin cambiar el
registro**, el mismo `transform` sigue sirviendo. Si se vuelve a registrar, hay
que medirlo de nuevo.

**La altura no está corregida.** El combinado no la resuelve: los pisos 1 y 2
no cuadran entre sí (el techo del piso 1 queda arriba del suelo del piso 2).
Las escaleras tampoco lo resuelven solas, porque los tramos se repiten en cada
piso y un ajuste automático puede empatar el tramo equivocado. Cuando se
conozca la corrección, va en el campo opcional `dz` del `transform` de cada
piso.

> **Coordenadas locales.** El proyecto se registró nube contra nube, sin
> dianas, así que el marco común tiene un origen arbitrario y no está
> georreferenciado. Para el visor no importa. Los marcadores de los salones se
> miden en ese marco.

### 3. Convertir

Desde `src/`, con los LAS alineados:

```powershell
python -m scans.convert --floor all --input-dir D:/las_A3_alineado
```

Unos 10 millones de puntos por segundo: menos de 10 minutos los cinco pisos.
Para uno solo, `--floor 1`.

Cada piso queda en la carpeta que dice su campo `cloud` en
[config/spaces.json](config/spaces.json), dentro de `D:\escaneos`. Para un
archivo con otro nombre: `--floor 2 --input D:/otra/ruta.las`.

### 4. Servir

```powershell
python -m scans.server
```

Al arrancar lista las nubes que encontró. Abrir
[localhost:8100](http://localhost:8100) devuelve esa misma lista en JSON.

Por qué no basta `python -m http.server`: Potree no descarga `octree.bin`
entero, pide solo los bytes de cada nodo (peticiones HTTP Range). Ese servidor
ignora los rangos y manda el archivo completo, y con un piso de 40 GB el visor
nunca termina de cargar. Además, la página viene de otro puerto, así que hace
falta CORS. `scans.server` hace las dos cosas.

### 5. Medir los salones

Separar la nube por salón es difícil, así que cada salón se representa con un
**marcador** (un punto) y sus **dimensiones**, guardados en
`config/spaces.json`. El marcador pone en la escena una etiqueta con el número
del salón y sus lecturas; las dimensiones dan el volumen *V* que usa el
balance de masa del CO₂ del generador físico. Se miden en el mismo visor,
con un panel que **no aparece en la vista normal** para no confundir a quien
solo quiere preguntar: se abre con
[localhost:8000/viewer/?medir](http://localhost:8000/viewer/?medir).

1. Abre el menú de Potree (botón arriba a la izquierda) → *Herramientas* →
   medir un punto, y haz clic en el centro del salón.
2. Abre *Medir salones*, elige el salón en su lista y pulsa
   *Marcador ← último punto*.
3. Para el largo, el ancho y el alto: mide una distancia con Potree (la
   herramienta de altura para el alto) y pulsa *← última distancia* junto al
   campo que corresponde.
4. *Guardar en este espacio*. El marcador aparece en la escena.
5. Al terminar, *Descargar spaces.json* y reemplaza `config/spaces.json` con
   el archivo descargado. Recarga el visor: no hace falta reiniciar la API.

Lo medido se guarda en el navegador hasta que lo descargas, así que recargar
la página no lo borra.

### Variables de entorno

Las direcciones salen del entorno, para que el mismo código corra en esta PC
y en un contenedor. Sin definir nada, valen los valores locales:

| Variable | Por omisión | Para qué |
|---|---|---|
| `SCANS_DIR` | `D:\escaneos` | Carpeta que sirve `scans.server` y donde escribe `scans.convert` |
| `SCANS_URL` | `http://localhost:8100` | Dónde busca el visor las nubes |
| `SCANS_PORT` | `8100` | Puerto de `scans.server` |
| `VIEWER_ORIGINS` | `http://localhost:8000,http://127.0.0.1:8000` | Páginas que pueden leer las nubes (CORS), separadas por coma |
| `POTREE_CONVERTER` | `D:\tools\PotreeConverter\…\PotreeConverter.exe` | Ruta del convertidor |
| `OLLAMA_URL` | `http://localhost:11434` | Dónde está Ollama. No es `OLLAMA_HOST`: ese nombre lo usa Ollama para otra cosa |

Para la consola actual, en PowerShell `$env:SCANS_DIR = "E:\escaneos"` y en
Git Bash `export SCANS_DIR=E:/escaneos`. **Para que quede fija** en esa
máquina, una sola vez y desde cualquiera de las dos terminales:

```
setx SCANS_DIR "E:\escaneos"
```

Vale para las consolas que se abran después, no para la actual.

**Nubes en otra máquina.** Dos formas:

- **Copiarlas.** Copia la carpeta `D:\escaneos` completa (134 GB) a un disco de
  la otra máquina y define `SCANS_DIR` con su ruta si no es `D:\escaneos`.
- **Servirlas por red** desde la máquina que las tiene. Ahí, arranca el
  servidor con `python -m scans.server --host 0.0.0.0`, abre el puerto 8100 en
  el firewall de Windows y agrega a `VIEWER_ORIGINS` la dirección desde la que
  se abre el visor. En la máquina de la API, define `SCANS_URL` con la IP del
  servidor de escaneos.

---

## Otros comandos

```powershell
python -m gru.compare_implementations                  # celda a mano contra nn.GRU
python -m llm.classifier                               # cómo se rutean las preguntas
python -m llm.context                                   # los prompts por tipo
```

| Opción | Para qué |
|---|---|
| `--data ruta` | Otro archivo con el contrato de datos; por omisión `data/processed/aulas3.parquet` |
| `--epochs N` / `--patience N` | Tope de épocas y épocas sin mejora antes de parar |
| `--implementation pytorch` | `nn.GRU` en vez de la celda a mano |
| `--normalize-by-space` | Media y sigma por espacio en vez de globales |
| `--label nombre` | Prefijo de los archivos guardados |
| `--seasonality seasonal` | SARIMAX con `s=288`. Inviable: 2,147 s por modelo |

---

## Si algo falla

| Síntoma | Qué hacer |
|---|---|
| `Permission denied (publickey)` al clonar | Esa máquina no tiene clave SSH en GitHub: clona por HTTPS (Instalación, paso 1) |
| `python` no se reconoce | No marcaste `Add python.exe to PATH`, o no reabriste la terminal |
| `running scripts is disabled` | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| `..venvScriptsActivate.ps1: command not found` | Estás en Git Bash: `source .venv/Scripts/activate` |
| `UnicodeEncodeError` o acentos rotos en Git Bash | `export PYTHONUTF8=1` en esa consola |
| `ModuleNotFoundError` | Estás fuera de `src/`, o el prompt no dice `(.venv)` |
| `No matching distribution found for torch` | Python fuera del rango 3.10–3.14 |
| `CUDA NO disponible` | Normal con la build de CPU; si esperabas GPU, revisa el índice `cuXXX` |
| `models/gru.pt does not exist` | Clon nuevo: entrena primero (Instalación, paso 2) |
| `gru_loaded: false` en `/health` | Lo mismo: falta entrenar |
| `No se puede conectar con Ollama` | Ollama no está instalado o no corre: `ollama serve` |
| `Ollama no tiene el modelo` | `ollama pull llama3.1:8b` |
| Tarda demasiado | Te faltó `--implementation pytorch` |
| `/viewer/` da 404 | Falta la librería de Potree en `web/potree/` (Instalación, paso 4) |
| "Ningún piso disponible" en el visor | `scans.server` no corre, o `SCANS_DIR` no apunta a las nubes |
| Un piso dice "sin convertir" | No existe `D:\escaneos\A3-PN\metadata.json`: falta convertirlo |
| Errores de CORS en la consola del navegador | La dirección del visor no está en `VIEWER_ORIGINS` del servidor de escaneos |
| El visor carga pero la nube nunca aparece | Se sirve con un servidor sin soporte de Range; usa `scans.server` |
| Los pisos no quedan encimados | Se convirtieron los LAS originales y no los de `D:\las_A3_alineado` (paso 2 de Escaneos y visor) |

---

## El contrato de datos

Cualquier fuente de datos debe entregar un DataFrame con estas columnas:

| Columna | Tipo | Nota |
|---|---|---|
| `timestamp` | datetime | hora local, intervalos de 5 min |
| `space_id` | int | identificador del espacio |
| `co2` | float | ppm |
| `temperature` | float | °C |
| `humidity` | float | %RH |
| `light` | float | % (0–100) |
| `motion` | float | fracción del intervalo con detección, 0 a 1 |

Los huecos se expresan como **filas ausentes**, no como NaN: es lo que pasa
cuando un nodo pierde la red.

---

## Estructura

```
src/common/        contrato de datos, ventanas, normalización, métricas
src/gru/          modelo, entrenamiento y predicción
src/sarimax/      baseline estadístico
src/llm/          clasificador, prompts y cliente de Ollama
src/api/          orquestador FastAPI; también sirve el visor
src/scans/        alineación y conversión de escaneos, servidor de nubes
web/              página del visor Potree (la librería, en web/potree/, no se versiona)
src/compare_models.py   GRU contra SARIMAX
docs/             estándares y hallazgos de ROBOD
data/, models/    datos y pesos (no versionados)
requirements.txt  dependencias, menos torch
check_gpu.py      diagnóstico de CUDA
```

## Documentación

| Documento | Contenido |
|---|---|
| [src/gru/README.md](src/gru/README.md) | Decisiones de diseño del GRU y qué queda abierto |
| [src/llm/README.md](src/llm/README.md) | La capa conversacional: qué se le puede preguntar, y el reparto de trabajo con el LLM |
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
(`PARTITION_MODE = "by_days"` en `config.py`), porque un corte cronológico
sobre una sola semana dejaba el fin de semana al final y calificaba al modelo
con el edificio vacío.
