# Mapa del código

Guía para leer el repositorio por partes. Cada sección explica primero las
ideas que hacen falta, luego el orden de lectura de los archivos y al final
unas preguntas para comprobar que se entendió. Los números de línea son del
25 de noviembre de 2026; si el código cambia, se pueden mover un poco.

| Parte | Carpetas | Estado de esta guía |
|---|---|---|
| Escaneos y visor (Potree) | `src/scans/`, `web/`, partes de `src/api/` | **completa** |
| Capa conversacional | `src/llm/`, `src/api/` | pendiente |
| Modelo predictivo y su preparación de datos | `src/gru/`, `src/sarimax/`, `src/common/` | **completa** |
| Ingesta (limpieza, horario, clima) | `src/ingestion/` | pendiente |

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

2. **[src/common/config.py](../src/common/config.py), líneas 39 a 63.**
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
   - Línea 149: `app.mount("/viewer", ...)` sirve la carpeta `web/` como
     página.
   - `root` (línea 697): abrir `localhost:8000` manda al visor.
   - `spaces` (línea 703): la ruta `/spaces`, que le da al visor la dirección
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
