# Capa conversacional

Implementa la sección 4.3 del documento de investigación: patrón RAG con
Llama 3.1 8B servido por Ollama, orquestado por FastAPI.

---

## Qué se le puede preguntar

Desde el chat del visor o `POST /chat`. Cada pregunta tiene un **tipo**, que
decide qué datos se buscan, y un **objetivo**: un salón o un piso entero.

| Tipo | Ejemplo | Qué responde |
|---|---|---|
| Actual | "¿Cómo está el CO₂ del 3308?" | Última lectura de las variables que nombras (todas si no nombras ninguna) |
| Pronóstico | "¿Cómo va a estar el 3308 en 45 minutos?" | GRU: directo hasta 30 min, iterativo hasta 2 h, en pasos de 5 min |
| Histórico | "¿Cuál fue el máximo de CO₂ en el 3206?" | Máximo, mínimo y promedio de **una** variable en las últimas 24 h |
| Anomalía | "¿Hay algo raro en el 3305?" | Que el detector todavía no existe; nunca un "todo bien" inventado |

| Objetivo | Cómo se reconoce | Qué cambia en la respuesta |
|---|---|---|
| Un salón | Su número o alias en el texto: "3308" | Datos de ese salón |
| Un piso | "piso 3", "piso tres", "tercer piso", "nivel 2", "planta baja" (piso 0) | Primero un resumen del piso (promedio, salón más alto y más bajo), después una oración por salón, y al final los espacios sin datos. El visor deja visible solo ese piso |
| Ninguno | — | Pregunta "¿Sobre qué salón o piso?", sin llamar a Llama. Nunca responde sobre un salón por omisión |

Si el texto nombra un salón y un piso ("el 3308 del piso 3"), gana el salón.
`POST /chat` acepta además un `space_id` para clientes que tengan un salón
seleccionado; el texto gana sobre él. El visor no lo manda.

**Lo que todavía no puede:** comparar dos salones, fechas u horas concretas
("el martes a las 10"), más de una variable en el histórico, detectar
anomalías, y datos en vivo: "ahora" es la última lectura del archivo limpio.
Solo hay datos de 3206, 3208, 3305 y 3308 (el 3304 tiene el sensor de gas
muerto).

**Fallas conocidas de redacción:** Llama omite a veces la advertencia de la
predicción iterativa (el visor la muestra igual, debajo de la respuesta) y
exagera adjetivos ("luz muy alta" con 15 %).

**Ya corregidas, quitándole la decisión al modelo:** la dirección de una
tendencia va escrita en el prompt ("sube 62.8 ppm"), porque con los dos números
solos dijo "disminuirá de 665 a 728 ppm"; el prompt nombra el salón por su
número y no por su id ("espacio 308"); y la lectura actual solo lleva las
variables que se preguntaron, porque con las cinco las reportaba todas.

---

## El reparto de trabajo

Una pregunta recorre cuatro pasos y cada uno tiene un dueño distinto:

```
classifier.py   decide de qué tipo es la pregunta       (reglas, sin LLM)
api/api.py        recupera los datos que ese tipo pide    (GRU, BD, detector)
context.py       arma el prompt con esos datos           (plantillas)
llm_client.py    pide la frase al modelo                 (HTTP a Ollama)
```

**Todos los números salen de los tres primeros pasos. El LLM solo redacta.**
Nunca se le pide calcular, comparar entre espacios ni estimar lo que falta,
porque un modelo de lenguaje al que se le pide aritmética produce una
respuesta fluida, plausible y equivocada, y nada río abajo podría notarlo. El
system prompt lo dice tres veces de formas distintas, y no por redundancia: es
la única defensa que hay en esa frontera.

---

## Por qué el clasificador no usa el LLM

Clasificar con el modelo agregaría un segundo punto de falla a cada petición.
Una decisión de ruteo equivocada mete los datos equivocados en el prompt, y un
prompt armado con los datos equivocados produce una respuesta segura de sí
misma y falsa. Las reglas por palabra clave son auditables, deterministas y
gratis: la misma pregunta siempre se rutea igual, y cuando se rutea mal se ve
en el código por qué.

El costo está medido y documentado, no escondido. La prioridad es

```
anomaly  >  forecast  >  history  >  current
```

y codifica urgencia, no precisión. La consecuencia visible:

| Pregunta | Se rutea como | Debería ser |
|---|---|---|
| `¿El CO2 va a estar muy alto en una hora?` | `anomaly` | `forecast` |

«muy alto» gana sobre «en una hora» y la pregunta pierde su horizonte.
Arreglarlo de verdad necesita un clasificador entrenado, no una lista de
palabras más larga. Mientras tanto se prefiere errar hacia la seguridad.

Para ver el ruteo completo:

```powershell
python -m llm.classifier
```

---

## Qué está conectado y qué no

| Pieza | Estado |
|---|---|
| GRU, predicción directa (30 min) | **real**, desde el checkpoint en `models/` |
| GRU, predicción iterativa (hasta 2 h) | **real** |
| Clasificación de intención | **real** |
| Construcción de prompts | **real** |
| Preguntas por piso, con resumen calculado en la API | **real** |
| Fuente de datos | archivo limpio `data/processed/aulas3.parquet`. `TODO E5` |
| Detector de anomalías 3σ | **no existe** |
| Llama 3.1 8B | **real**, vía Ollama (3 a 7 s por respuesta en la RTX 4090) |

Los tres métodos marcados `TODO E5` en `api/api.py` son los únicos que cambian
cuando haya acceso a la base de datos del Tec: `_space_rows()`,
`_current_data()` y `_history_data()`. El resto del orquestador, incluidas las
preguntas por piso, se apoya en ellos y no se toca.

### Sobre el detector que no existe

`_anomaly_data()` devuelve `state: "unavailable"`, **no una lista vacía de
anomalías**. La distinción no es cosmética: una lista vacía significa que el
detector corrió y no encontró nada, y afirmar eso mientras no hay detector le
pondría al usuario un «todo bien» falso enfrente. Esa es exactamente la
respuesta que esta capa no debe producir nunca. El prompt de anomalías tiene
una rama para ese estado que le ordena al modelo decir que la capacidad no
está disponible.

---

## Instalar Ollama

```powershell
# 1. Descargar e instalar desde https://ollama.com/download/windows
ollama --version

# 2. Descargar el modelo (~5 GB)
ollama pull llama3.1:8b

# 3. Comprobar que responde
python -m llm.llm_client
```

Ollama queda como servicio en segundo plano después de instalarse. Si no
responde, `ollama serve` lo arranca a mano.

`python -m llm.llm_client` sondea primero y solo pregunta si el modelo está;
termina en código 1 con el comando que hace falta si algo falta, en vez de
tronar con un error de red.

**El modelo corre local.** Ninguna lectura de Aulas 3 sale de la máquina, que
es la razón por la que el documento eligió un modelo local y no una API
hospedada.

---

## Probar sin Ollama

`/context` recorre toda la cadena excepto la llamada al modelo y devuelve el
prompt que se habría enviado. Sirve para dos cosas: demostrar la mitad de
recuperación en una máquina sin Ollama, y separar la culpa cuando una
respuesta sale mal —si el prompt ya traía el dato equivocado, el problema está
en la recuperación, no en el modelo.

```
uvicorn api.api:app --port 8000
```

Abre [localhost:8000/docs](http://localhost:8000/docs), elige `POST /context`
y manda, por ejemplo:

```json
{"question": "¿Cómo va a estar el CO2 del piso 3 en 45 minutos?"}
```

Desde la terminal los acentos se corrompen con facilidad, por eso `/docs`.

---

## Decisiones abiertas

**Los umbrales de confort no están fijados.** `context.py` anota las lecturas
con bandas de CO₂ (800 / 1000 / 1500 ppm) y rangos de temperatura y humedad
que son valores de trabajo, no tomados de una norma. La sección 4.3.4 no los
fija. Cuando se decidan, su lugar es `common/config.py`, junto a los límites
físicos, no aquí.

**Los nombres de salón y de piso salen de `config/spaces.json`.** El
clasificador lee de ahí los números de salón, los alias y a qué piso pertenece
cada espacio; agregar un alias ahí basta para que se reconozca.

**`motion` se reporta como número continuo.** El GRU lo predice bajo MSE
junto con las demás variables, así que su salida es un valor sin calibrar
entre 0 y 1, no una probabilidad. El prompt lo etiqueta `(1 = hay movimiento,
0 = no)`, lo que es honesto para una lectura real pero engañoso para una
predicción. Está relacionado con la decisión abierta de `src/gru/README.md`
sobre darle una cabeza aparte con `BCEWithLogitsLoss`.

**El horizonte iterativo se calcula en bloques de 30 min.** Pedir 45
minutos corre dos iteraciones (60 min) y recorta a 9 pasos. El modelo solo
sabe emitir bloques de 6 pasos, así que la alternativa sería entrenar cabezas
de otro tamaño.
