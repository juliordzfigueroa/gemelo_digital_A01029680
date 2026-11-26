# Mapa del código

Guía para leer el repositorio por partes. Cada sección explica primero las
ideas que hacen falta, luego el orden de lectura de los archivos y al final
unas preguntas para comprobar que se entendió. Los números de línea son del
26 de noviembre de 2026; si el código cambia, se pueden mover un poco.

| Parte | Carpetas | Estado de esta guía |
|---|---|---|
| Escaneos y visor (Potree) | `src/scans/`, `web/`, partes de `src/api/` | **completa** |
| Capa conversacional | `src/llm/`, `src/api/`, `src/common/spaces.py` | **completa** |
| Modelo predictivo y su preparación de datos | `src/gru/`, `src/sarimax/`, `src/common/` | **completa** |
| Ingesta (limpieza, horario, clima) | `src/ingestion/` | **completa** |

---

## Escaneos y visor (Potree)

### Las ideas

**Nube de puntos.** El escáner BLK360 mide millones de puntos de las
superficies del edificio, cada uno con posición (x, y, z) y color. Un piso
tiene entre 455 y 1,576 millones de puntos.

**LAS.** Formato de archivo estándar para nubes de puntos. Empieza con un
encabezado de 227 bytes (cuántos puntos hay, escala, origen, límites) y
luego cada punto ocupa 26 bytes. Las coordenadas se guardan como enteros:
la posición real es `entero × escala + origen`. Esto importa en
`transform.py`, que cambia el origen y los enteros.

**Por qué no se puede abrir la nube directo en el navegador.** Un piso pesa
de 12 a 41 GB; el navegador no puede descargar ni dibujar eso. Potree
resuelve el problema con un **octree**: divide el espacio en cubos, y cada
cubo en ocho cubos más chicos, y guarda en cada nivel una muestra de los
puntos. Desde lejos se dibujan los cubos grandes con pocos puntos; al
acercarse, se piden los cubos chicos con más detalle. Es la misma idea que
los niveles de detalle de un videojuego.

**PotreeConverter.** Programa que convierte un LAS en ese octree. Deja tres
archivos por piso: `metadata.json` (la descripción), `hierarchy.bin` (qué
cubos existen) y `octree.bin` (los puntos de todos los cubos, uno tras otro).

**HTTP Range.** Para no descargar `octree.bin` completo (decenas de GB),
Potree pide solo los bytes de cada cubo: "dame del byte 1,200,000 al
1,450,000". El servidor contesta con código **206** (contenido parcial). Un
servidor que no entiende rangos manda el archivo entero y el visor nunca
termina de cargar. Por eso existe `src/scans/server.py`.

**CORS.** Por seguridad, el navegador no deja que una página lea datos de
otro "origen" (otro puerto o dominio) a menos que ese otro servidor lo
permita. La página del visor viene de la API (puerto 8000) y las nubes del
servidor de escaneos (puerto 8100), así que el servidor de escaneos tiene
que decir "el puerto 8000 sí puede leerme". Antes de pedir un rango, el
navegador manda una pregunta previa (`OPTIONS`, el *preflight*) para
confirmar que está permitido.

**Alineación.** Cada piso salió de Leica con su propio sistema de
coordenadas en planta. `transform.py` gira y mueve cada piso para que
queden uno sobre otro, con los valores guardados en `spaces.json`.

### Cómo viaja todo

```
Leica ──► LAS por piso ──► transform.py ──► LAS alineado ──► convert.py ──► octree en D:\escaneos
                           (gira y mueve)                    (PotreeConverter)

Navegador ── index.html + viewer.js ◄── API :8000 (sirve la página y /spaces)
     └── pide cubos con Range ──────► server.py :8100 (lee D:\escaneos)
```

### Orden de lectura

1. **[config/spaces.json](../config/spaces.json), la lista `floors`.**
   Qué buscar: para cada piso, `cloud` (dónde está su octree) y `transform`
   (cuánto se gira y mueve). Es la única "base de datos" de los pisos.

2. **[src/common/config.py](../src/common/config.py), líneas 40 a 67.**
   Qué buscar: las rutas y direcciones (`SCANS_DIR`, `SCANS_URL`,
   `VIEWER_ORIGINS`). Todas se leen de variables de entorno con un valor por
   omisión; así el mismo código corre en otra máquina sin tocarlo.

3. **[src/scans/transform.py](../src/scans/transform.py).** Alinea un piso.
   - `read_header` (línea 61): lee el encabezado del LAS con `struct`, que
     convierte bytes en números.
   - `plan_matrix` (línea 88): la matriz de giro en planta, la misma de
     geometría de bachillerato: `[[cos, −sen], [sen, cos]]`.
   - `transform_file` (línea 94): la parte central. Lee el archivo por
     bloques de 20 millones de puntos (no cabe en memoria completo), gira y
     mueve x, y, y vuelve a escribir los enteros. La altura z no se toca.
   - `main` (línea 153): lee los argumentos de la línea de comandos.

4. **[src/scans/convert.py](../src/scans/convert.py).** Solo llama a
   PotreeConverter con las rutas correctas. `convert` (línea 72) arma el
   comando y lo ejecuta con `subprocess.run`.

5. **[src/scans/server.py](../src/scans/server.py).** El servidor de nubes.
   - `build_app` (línea 71): arma la aplicación con dos piezas de la
     librería Starlette: `StaticFiles`, que sirve archivos y ya entiende
     rangos, y `CORSMiddleware`, que agrega los permisos de CORS.
   - `index` (línea 82): la ruta `/`, que lista las nubes encontradas.
   - Comparado con tu `server_traffic.py`: es lo mismo que una app Flask con
     dos rutas. Se usó Starlette porque ya viene instalada con FastAPI (la API
     está construida sobre ella); con Flask también se podría, usando
     `send_file`, que igual entiende rangos.

6. **[src/api/api.py](../src/api/api.py), solo las partes del visor.**
   - Línea 125: `app.mount("/viewer", ...)` sirve la carpeta `web/` como
     página.
   - `root` (línea 202): abrir `localhost:8000` manda al visor.
   - `get_spaces` (línea 208): la ruta `/spaces`, que le da al visor la dirección
     del servidor de escaneos, la lista de pisos y los salones con su última
     lectura.

7. **[web/index.html](../web/index.html).** La página. Las líneas 16 a 27
   cargan la librería Potree y lo que necesita; la línea 30 es el área donde
   se dibuja la nube; desde la 34, el panel de la derecha (pisos, chat y el
   panel oculto de medir salones). Al final carga `viewer.js`.

8. **[web/viewer.js](../web/viewer.js).** El cliente, como tu cliente WebGL:
   - Arriba (líneas 11 a 30): constantes, entre ellas `SURVEY_MODE`, que
     solo vale verdadero si la dirección termina en `?medir`.
   - `main` (línea 55): pide `/spaces`, carga los pisos y dibuja marcadores.
   - `loadFloors` (línea 78): para cada piso revisa que exista su
     `metadata.json`, lo carga con `Potree.loadPointCloud`, lo agrega a la
     escena y crea su casilla para mostrarlo u ocultarlo.
   - `showOnlyFloor` / `showAllFloors` (líneas 120 y 129): lo que pasa cuando
     el chat responde sobre un piso o sobre el edificio.
   - `drawMarkers` (línea 156): pone una etiqueta por salón con marcador
     (hoy ninguno tiene).
   - El bloque de la línea 199: el chat. Manda la pregunta a `/chat`, muestra
     la respuesta y la línea de abajo ("lectura actual • piso 2").
   - `setupSurvey` (línea 285): el panel de medir salones; solo corre con
     `?medir`.

### Preguntas para comprobar

1. Si abres el visor y el servidor de escaneos está apagado, ¿qué archivo
   decide qué mensaje sale, y en qué función?
2. ¿Por qué `transform.py` lee el LAS por bloques en lugar de completo?
3. ¿Qué pasaría si sirves `D:\escaneos` con `python -m http.server`?
4. Si mañana la API se mueve a otra computadora, ¿qué variable de entorno
   cambias para que el visor siga encontrando las nubes, y en qué máquina?
5. ¿Dónde está guardado cuánto se gira el piso 1, y quién usa ese valor?

Respuestas: 1) `web/viewer.js`, en `loadFloors` (el mensaje "Ningún piso
disponible…"). 2) Porque un piso pesa hasta 41 GB y no cabe en memoria.
3) No soporta rangos: el visor pediría un cubo y recibiría el archivo entero,
y nunca terminaría de cargar. 4) `SCANS_URL`, en la máquina de la API; y en
la de escaneos, agregar la dirección del visor a `VIEWER_ORIGINS`. 5) En
`config/spaces.json`, `floors[1].transform`; lo usa `src/scans/transform.py`.

---

## Modelo predictivo (GRU y SARIMAX)

### Las ideas

**Qué se predice.** Con la última hora de lecturas de un salón (12 pasos de 5
minutos), predecir los siguientes 30 minutos (6 pasos) de las 5 variables.
Para más de 30 minutos se repite el proceso con lo predicho (modo iterativo,
hasta 2 h).

**Ventana.** Un ejemplo de entrenamiento: 12 pasos de entrada y los 6 que
siguen como respuesta correcta. Una ventana solo vale si sus 18 pasos son del
mismo tramo continuo, no tienen huecos y caen en la misma partición.

**Partición por días.** Los datos se reparten en entrenamiento, validación y
prueba por días completos. Prueba y validación son los últimos días entre
semana; los fines de semana se quedan en entrenamiento. Validación decide
cuándo dejar de entrenar; prueba es el examen final que la red nunca vio.

**Normalización (z-score).** El CO₂ anda en cientos y el movimiento entre 0 y
1; para que la red no le dé más peso al CO₂ solo por ser grande, cada variable
se transforma a `(x − media) / sigma`. La media y la sigma salen solo de
entrenamiento, y se guardan junto al modelo para regresar las predicciones a
ppm y grados.

**GRU.** Una red recurrente: lee los pasos en orden y lleva una "memoria"
(el estado oculto `h`). Las compuertas deciden qué olvidar (reset, `r`) y
cuánto actualizar (update, `z`). Son las ecuaciones 1 a 4 de tu documento, y
el código las escribe una por una.

**Embedding del salón.** Un solo modelo para todo el edificio: cada salón
tiene un vector de 8 números que la red aprende y que le dice "este es el
3308". Es lo que deja que un modelo global se comporte distinto por salón.

**SARIMAX.** El modelo estadístico de comparación. Uno por variable y salón
con ventanas de prueba (hoy 20: 5 variables × 4 salones), con el ciclo diario
como senos y cosenos (términos de Fourier). Se evalúa sobre exactamente las
mismas ventanas que el GRU.

### Cómo viaja todo

```
aulas3.parquet ─► data.preprocess ─► tramos continuos, rejilla de 5 min,
                                     partición por días, hora como seno/coseno
       │
       ├─ GRU:     data.prepare ─► normaliza ─► EnvironmentalWindows ─► train.py ─► models/gru.pt
       │                                                                   │
       │                                                 predict.py ◄──────┘ (lo usa la API)
       │
       └─ SARIMAX: sarimax/train.py ─► las mismas EnvironmentalWindows, en unidades físicas
                                       ─► 20 modelos ─► mismas métricas (common/metrics.py)
```

### Orden de lectura

1. **[src/common/config.py](../src/common/config.py).** Los números que
   definen todo: variables, 12 y 6 pasos, [128, 64], tasa de aprendizaje,
   fracciones de la partición. Sin medias ni sigmas: esas se calculan.

2. **[src/common/data.py](../src/common/data.py)**, de arriba abajo:
   - `mark_blocks` (línea 43) y `regularize` (línea 68): parten cada salón en
     tramos continuos y rellenan huecos de hasta 15 minutos.
   - `add_time_features` (línea 110): la hora y el día como seno y coseno.
   - `choose_days` (línea 138) y `split` (línea 181): la partición por días.
   - `Normalizer` (línea 206): `fit` calcula media y sigma, `transform`
     normaliza, `denormalize` regresa a unidades físicas.
   - `EnvironmentalWindows` (línea 315): arma las ventanas válidas. Lee el
     bucle de la línea ~372: es la regla de las tres condiciones.
   - `preprocess` (línea 413) y `prepare` (línea 445): juntan todo lo
     anterior en el orden correcto.

3. **[src/gru/model.py](../src/gru/model.py).**
   - `GRUCell.forward` (línea 100): las ecuaciones 1 a 4, una por línea, con
     el número de ecuación en el comentario.
   - `GRULayer.forward` (línea 136): aplica la celda paso por paso, como un
     `for` sobre el tiempo.
   - `EnvironmentalGRU.forward` (línea 199): embedding del salón, dos capas,
     dropout y la capa de salida.

4. **[src/gru/train.py](../src/gru/train.py).**
   - `run_epoch` (línea 64): una pasada por los datos; si hay optimizador,
     entrena; si no, solo evalúa.
   - `train` (línea 141): el bucle de épocas con early stopping (para cuando
     validación deja de mejorar 15 épocas seguidas) y la evaluación final en
     unidades físicas.
   - `save` (línea 264): escribe los tres archivos en `models/`.

5. **[src/gru/check.py](../src/gru/check.py).** Una serie inventada y
   perfecta; si la red no la aprende, hay un error en el código.

6. **[src/gru/predict.py](../src/gru/predict.py).** El que usa la API.
   - `load_model` (línea 64): carga pesos, normalizador y metadatos.
   - `last_contiguous_window` (línea 282): busca la última hora sin huecos.
   - `predict` (línea 318): directo si son 30 min o menos; si son más, repite:
     predice 30 min, los agrega a la ventana y vuelve a predecir.

7. **[src/sarimax/model.py](../src/sarimax/model.py)** y
   **[src/sarimax/train.py](../src/sarimax/train.py).**
   - `fourier_terms` (model.py, línea 48): los senos y cosenos del ciclo
     diario.
   - `test_windows` (train.py, línea 66): pide las ventanas de prueba a la
     misma clase del GRU; de ahí salen también las respuestas correctas.
   - `fit_all` (train.py, línea 84): para cada salón, ajusta un modelo por
     variable y pronostica sus ventanas.

8. **[src/compare_models.py](../src/compare_models.py).** Corre los dos y
   pone los resultados lado a lado.

### Preguntas para comprobar

1. ¿Por qué la media y la sigma se calculan solo con entrenamiento?
2. Si un salón se queda una hora sin lecturas, ¿qué pasa con las ventanas que
   cruzarían ese hueco, y en qué función se decide?
3. ¿Qué línea de `GRUCell.forward` es la diferencia con la GRU de PyTorch?
4. ¿Cómo se asegura que SARIMAX y el GRU se califican sobre las mismas
   ventanas?
5. Si pides un pronóstico a 2 horas, ¿cuántas veces corre la red?

Respuestas: 1) Si se usaran todos los datos, la red tendría información del
futuro (de prueba) desde antes de entrenar, y las métricas saldrían mejores de
lo que son. 2) Un hueco de más de 15 minutos sin lecturas corta el tramo
(`mark_blocks`), y ninguna ventana cruza un corte (`EnvironmentalWindows`). 3) La de la ecuación
3: `r * h_prev` se multiplica antes de aplicar `W_h`. 4) SARIMAX construye la
misma clase `EnvironmentalWindows`: de `origins` saca dónde pronosticar y de
`y` la respuesta correcta. 5) Cuatro: cada pasada
da 30 minutos.

---

## Ingesta (limpieza, horario y clima)

### Las ideas

**Exportación cruda.** Cada nodo ESP32 manda una lectura cada ~5 segundos, y
la plataforma las exporta en CSV: 370,341 filas del 1 al 11 de octubre. Las
columnas que importan son `deviceId`, `formattedTimestamp`, `temperature`,
`humidity`, `lightPercent`, `motion` y `eco2`.

**Contrato de datos.** Lo que esperan todos los modelos: una fila por salón y
por intervalo de 5 minutos, con `timestamp`, `space_id` y las 5 variables. Está
definido en `config.py`. La limpieza es la única parte que conoce el formato
crudo; si mañana los datos llegan de la base del Tec, solo cambia esta parte.

**De sensor a salón.** El modelo aprende salones, no sensores.
`config/sensors.csv` dice qué sensor estuvo en qué salón y desde cuándo. Si
mueves un sensor, se agrega una fila con fechas y las lecturas se reparten
solas.

**Invalidar, no borrar.** Una lectura imposible se vuelve `NaN` solo en la
variable afectada; las demás variables de esa lectura se quedan. Hay tres
reglas: el `-1` del sensor de gas, temperatura y humedad en 0 al mismo tiempo, y
valores fuera del rango del instrumento. Todo queda contado en el reporte.

**eCO₂ no es CO₂.** El sensor de gas es MOX: estima el CO₂ a partir de
compuestos volátiles. Sube con productos de limpieza y marcadores, y algunos
fines de semana se queda arriba de 1,000 ppm sin nadie en el salón.

**Horario sin día de la semana.** `schedule.csv` trae fechas, horas y salón,
pero no qué día de la semana se da la clase. Por eso `schedule.py` calcula una
ocupación *posible* (cualquier día entre semana dentro del rango de fechas) y
la compara contra lo que vieron los sensores.

**Clima.** Viene de Open-Meteo, gratis y sin clave. El archivo histórico se
publica con unos días de retraso; lo que falte se llena con el servicio de
pronóstico, y cada hora dice de cuál vino. Además de la humedad relativa se
calcula la absoluta (g/m³), porque la relativa depende de la temperatura.

### Cómo viaja todo

```
CSV de los sensores ─► cleaning.py ─► data/processed/aulas3.parquet  (el contrato: lo leen GRU, SARIMAX y la API)
                          │            data/processed/aulas3_report.json  (qué se descartó y por qué)
                  config/sensors.csv

schedule.csv ─► schedule.py ─► tabla en la terminal (¿el horario explica lo que ven los sensores?)
                   │
           config/spaces.json  (salón "3308" → space_id 308)

Open-Meteo ─► weather.py ─► data/processed/weather.parquet  (clima cada 5 minutos)
```

### Orden de lectura

1. **[config/sensors.csv](../config/sensors.csv).** Ocho filas. Qué buscar:
   `device` → `space_id`, y las notas: el 304 con el sensor de gas muerto, el
   402 con 2 horas de datos y el sensor 8 sin lugar.

2. **[src/ingestion/cleaning.py](../src/ingestion/cleaning.py).** Empieza por
   `clean` (línea 271): es la lista de pasos y da el panorama.
   - Las constantes (líneas 51 a 68): `COLUMNS` (qué columna cruda se vuelve
     cuál), `TIME_FORMAT`, `SENTINEL = -1` y `MIN_READINGS_PER_INTERVAL = 6`.
   - `read_raw` (línea 71): lee uno o varios CSV, convierte la hora y quita
     las lecturas repetidas entre exportaciones. Si una hora no tiene el
     formato esperado, se detiene en lugar de tirarla en silencio.
   - `assign_spaces` (línea 125): el bucle recorre `sensors.csv` y le pone a
     cada lectura su salón; las de sensores sin salón se descartan y se
     cuentan.
   - `invalidate` (línea 170): las tres reglas, en orden. `mark` (línea 191)
     cuenta por sensor antes de borrar.
   - `aggregate` (línea 215): agrupa por salón e intervalo de 5 minutos y
     promedia. Un intervalo con menos de 6 lecturas válidas queda vacío en esa
     variable. El movimiento promediado da la fracción del intervalo con
     detección.
   - `space_summary` (línea 245): el resumen por salón que imprime la
     terminal.

3. **[src/ingestion/schedule.py](../src/ingestion/schedule.py).**
   - `load_schedule` (línea 93): lee el CSV y junta las filas idénticas.
     Convierte las fechas y las horas con `_minutes` (línea 62: `1310` → 790
     minutos), y traduce el salón a `space_id` con `load_spaces` (línea 75).
   - `possible_occupancy` (línea 139): para cada intervalo, cuántos alumnos
     podría haber. Solo cuenta de lunes a viernes, dentro de las fechas y
     horas del bloque; si dos bloques se enciman, se queda el mayor.
   - `compare` (línea 175): por salón y día, el movimiento y el eCO₂ dentro y
     fuera de clase entre las 7 y las 22 h, y su correlación con los alumnos
     posibles.

4. **[src/ingestion/weather.py](../src/ingestion/weather.py).**
   - `absolute_humidity` (línea 63): la fórmula de Magnus de la WMO.
   - `_query` (línea 82) y `download` (línea 118): piden al archivo histórico
     y llenan con el pronóstico lo que falte.
   - `to_intervals` (línea 148): de horas a intervalos de 5 minutos, por
     interpolación lineal.
   - `attach` (línea 175) y `summary` (línea 190): pegan el clima a cada fila
     de los sensores y comparan interior contra exterior.

### Preguntas para comprobar

1. Si mañana mueves el sensor del 3308 al 3309, ¿qué archivo cambias y qué
   pasa con las lecturas anteriores?
2. Una lectura trae `eco2 = -1` pero la temperatura bien. ¿Se pierde la
   temperatura?
3. ¿Por qué un intervalo con solo 3 lecturas válidas de CO₂ queda vacío en
   CO₂?
4. ¿Por qué `schedule.py` habla de ocupación "posible" y no "programada"?
5. ¿Por qué `weather.py` calcula la humedad absoluta?

Respuestas: 1) `config/sensors.csv`: a la fila de `esp32_room_03` le pones
`until` con la fecha del cambio, y agregas otra fila para el mismo sensor con
`space_id` 309 y `since` en esa fecha. Las lecturas anteriores siguen siendo
del 308. 2) No: `invalidate` solo vacía la variable afectada. 3) Por
`MIN_READINGS_PER_INTERVAL = 6`: el promedio de tan pocas lecturas es ruido.
4) Porque el horario no trae el día de la semana: un bloque cuenta en todos los
días entre semana de su rango. 5) Porque la relativa depende de la
temperatura: el mismo aire marca ~95% afuera y ~46% adentro, ya calentado; el
balance de humedad se escribe en g/m³.

---

## Capa conversacional (el chat)

### Las ideas

**El modelo redacta, el código calcula.** Es el patrón RAG de la sección 4.3.
Llama 3.1 8B corre local en Ollama. La API busca los números (lecturas,
pronóstico del GRU, promedios del piso), los mete en el prompt y Llama solo
escribe la frase. Nunca se le pide sumar, comparar ni estimar: un modelo de
lenguaje que hace aritmética da respuestas fluidas y equivocadas.

**Intención.** Cada pregunta se clasifica en dos cosas: el **tipo** (lectura
actual, pronóstico, historial o anomalía) y el **lugar** (un salón, un piso o
todo el edificio). Se hace con reglas de palabras, no con el LLM, para que la
misma pregunta siempre siga el mismo camino.

**Respuestas directas.** Cuando no hay datos que redactar (un saludo, algo
fuera de tema, un salón que no existe o que no tiene sensor, una pregunta que
no dice dónde), el código contesta con frases fijas y Llama ni se entera.

**Dos prompts.** El de sistema son las reglas fijas (máximo 3 oraciones, solo
los datos recibidos, con unidades). El de usuario lleva la pregunta y los
datos de esa pregunta.

**Piso o edificio.** Llama escribe solo el resumen, en una o dos oraciones, a
partir del promedio y los salones extremos que calculó la API. La línea de
cada salón la escribe el código y se pega después.

**El estado.** Al arrancar, la API carga una vez el GRU y los datos en
`retrieval.STATE`. Es lo mismo que la variable global `model` de tus
servidores Flask.

### Cómo viaja una pregunta

```
"¿Cómo va a estar el piso 3 en una hora?"
   │
   ├─ classifier.classify          tipo forecast, piso 3, horizonte 60 min
   ├─ api._target_and_data         ¿salón? ¿número desconocido? ¿piso? ¿edificio? ¿charla?
   ├─ retrieval.floor_data         GRU salón por salón, más promedio, mínimo y máximo
   ├─ context.build_prompt         como es un piso, lo escribe floor_prompt.prompt
   ├─ llm_client.generate_answer   Llama escribe una o dos oraciones
   ├─ floor_prompt.detail          el código agrega una línea por salón
   └─ Answer                       el visor muestra el texto y deja visible el piso 3
```

### Orden de lectura

1. **[src/api/api.py](../src/api/api.py).** Es como tu servidor Flask: la
   aplicación y sus rutas.
   - `Question` y `Answer` (líneas 55 y 61): lo que manda y recibe el visor.
   - `_target_and_data` (línea 131): el ruteo. Lee los `if` en orden: salón,
     número desconocido, piso, edificio, charla, y al final "¿de qué salón?".
   - `chat` (línea 284): la cadena completa de una pregunta.

2. **[src/llm/classifier.py](../src/llm/classifier.py)** y
   **[src/llm/vocabulary.py](../src/llm/vocabulary.py).**
   - `classify` (línea 92): junta todo en un `Intent`.
   - `_detect_kind` (línea 175): la prioridad anomalía > pronóstico > historial
     > actual.
   - `_extract_space`, `_extract_floor` y `_extract_horizon` (líneas 241, 256
     y 310): el salón, el piso y los minutos.
   - `_detect_chat` (línea 140): saludo, agradecimiento o fuera de tema.
   - `vocabulary.py` son solo las listas de palabras; ahí se agregan
     sinónimos.

3. **[src/api/retrieval.py](../src/api/retrieval.py).** De dónde salen los
   números.
   - `STATE` (línea 46) y `space_rows` (línea 58): la única función que
     cambia cuando llegue la base de datos del Tec (`TODO E5`).
   - `current_data`, `history_data`, `anomaly_data` y `forecast_data`
     (líneas 96, 120, 154 y 208): un salón.
   - `floor_data` (línea 289): un piso o el edificio; `_floor_summary` (línea
     402) calcula el promedio y los extremos.

4. **[src/llm/context.py](../src/llm/context.py).** Los prompts de un salón.
   `SYSTEM_PROMPT` (línea 41) son las reglas; `build_prompt` (línea 57) elige
   qué prompt armar.

5. **[src/llm/floor_prompt.py](../src/llm/floor_prompt.py).** El prompt de un
   piso (`prompt`, línea 59) y el detalle por salón (`detail`, línea 227).

6. **[src/llm/formatting.py](../src/llm/formatting.py).** Cómo se escriben los
   números. `threshold_notes` (línea 129) pone las etiquetas "ALTO" o "por
   encima del rango de confort"; `trend` (línea 202) escribe "sube 62.8 ppm";
   `anomaly` escribe cada anomalía que encontró el detector.

7. **[src/llm/replies.py](../src/llm/replies.py).** Las frases fijas.
   `TEMPLATES` (línea 38) trae las plantillas; `reply` (línea 91) escoge una
   de las versiones con un hash de la pregunta, así que la misma pregunta
   recibe siempre la misma frase.

8. **[src/llm/llm_client.py](../src/llm/llm_client.py).** El POST a Ollama
   (`generate_answer`, línea 56) y la revisión de que esté encendido
   (`available`, línea 117).

9. **[src/common/spaces.py](../src/common/spaces.py).** El registro de
   salones: `labels` (línea 39) convierte 308 en "3308" y `space_4_2` en
   "oficina 4-2".

10. **[src/anomaly/detector.py](../src/anomaly/detector.py)** y
    **[src/anomaly/evaluate.py](../src/anomaly/evaluate.py).** El detector de
    anomalías de la sección 4.2.5.
    - `deviation` (línea 37): a cuántas sigmas está un valor de la media. Es
      la única fórmula; `check` (línea 53, una lectura, para el chat) y
      `flags` (línea 100, miles de lecturas, para la evaluación) la usan
      igual.
    - `inject_spikes`, `inject_damaged` e `inject_stuck` (líneas 88, 101 y
      110): las tres anomalías sintéticas.
    - `score` (línea 142): precisión y recall. Los resultados están en
      [src/anomaly/README.md](../src/anomaly/README.md).

### Preguntas para comprobar

1. Si Ollama está apagado y preguntas "¿cómo está el 3308?", ¿qué responde el
   chat y en qué función se decide?
2. ¿Por qué "¿qué método usas?" no se va a todo el edificio, aunque "método"
   contiene "todo"?
3. Quieres que "bochorno" cuente como temperatura. ¿Qué archivo cambias?
4. En la respuesta de un piso, ¿quién escribe "sube 62.8 ppm": Llama o el
   código?
5. Cuando llegue la base de datos del Tec, ¿qué función cambias?

Respuestas: 1) "No puedo responder ahora mismo." seguido de la razón; se
decide en `chat`, en el `except RuntimeError` (línea 302 de `api.py`). 2)
Porque las palabras de edificio se buscan como palabras completas, con
`_contains_word`. 3) `llm/vocabulary.py`: se agrega a
`VARIABLE_SYNONYMS["temperature"]`. 4) El código: `formatting.trend`. 5)
`space_rows`, en `api/retrieval.py`.
