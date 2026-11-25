# Mapa del código

Guía para leer el repositorio por partes. Cada sección explica primero las
ideas que hacen falta, luego el orden de lectura de los archivos y al final
unas preguntas para comprobar que se entendió. Los números de línea son del
25 de noviembre de 2026; si el código cambia, se pueden mover un poco.

| Parte | Carpetas | Estado de esta guía |
|---|---|---|
| Escaneos y visor (Potree) | `src/scans/`, `web/`, partes de `src/api/` | **completa** |
| Capa conversacional | `src/llm/`, `src/api/` | pendiente |
| Modelo predictivo | `src/gru/`, `src/sarimax/` | pendiente |
| Datos | `src/common/`, `src/ingestion/` | pendiente |

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
