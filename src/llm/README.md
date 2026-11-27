# Capa conversacional

Implementa la sección 4.3 del documento de investigación: patrón RAG con
Llama 3.1 8B servido por Ollama, orquestado por FastAPI.

---

## Qué se le puede preguntar

Desde el chat del visor o `POST /chat`. Hay dos familias de respuesta:

- **Respuestas con datos.** Cuatro tipos (lectura actual, pronóstico,
  histórico, anomalía) sobre un objetivo (un salón, un piso o todo el
  edificio). Los números los pone el código; Llama redacta.
- **Respuestas de la herramienta.** Saludos, agradecimientos, preguntas
  fuera de tema, preguntas que no dicen dónde, salones que no existen,
  salones sin sensor y qué significan las etiquetas de una respuesta. No
  llevan datos que redactar, así que las contesta el código con frases fijas
  de `replies.py` y `glossary.py`, sin llamar a Llama.

Al abrir el visor, el chat empieza con una bienvenida (`GET /welcome`): qué
se le puede preguntar y qué quieren decir "medido", "simulado" y "estimado",
con los salones que tienen sensor y el error típico tomados de los datos.

```
pregunta ─► clasificador ─┬─ ¿qué significa una etiqueta? ───── sí ─► la definición
                          ├─ ¿nombra salón, piso o edificio? ── sí ─► respuesta con datos (Llama)
                          │                                            o "no tiene sensor"
                          └─ no ─┬─ pide algo de un chatbot general ─► fuera de tema
                                 ├─ habla del edificio (variable, ─────► "¿de qué salón o piso?"
                                 │   tiempo, salón, lecturas…)
                                 ├─ "gracias" ────────────────────────► agradecimiento
                                 ├─ saludo o "¿qué eres?" ────────────► presentación
                                 └─ nada reconocible ─────────────────► fuera de tema
```

### Respuestas con datos

Cada pregunta tiene un **tipo**, que decide qué datos se buscan, y un
**objetivo**.

| Tipo | Ejemplo | Qué responde |
|---|---|---|
| Actual | "¿Cómo está el CO₂ del 3308?" | Última lectura de las variables que nombras (todas si no nombras ninguna) |
| Pronóstico | "¿Cómo va a estar el 3308 en 45 minutos?" | GRU: directo hasta 30 min, iterativo hasta 2 h, en pasos de 5 min |
| Histórico | "¿Cuál fue el máximo de CO₂ en el 3206?" | Máximo, mínimo y promedio de **una** variable en las últimas 24 h |
| Anomalía | "¿Hay algo raro en el 3305?" | El detector 3σ sobre la última lectura de CO₂, temperatura, humedad y luz, más los rangos de confort aparte. Si no puede revisar, lo dice; nunca un "todo bien" inventado |

| Objetivo | Cómo se reconoce | Qué responde |
|---|---|---|
| Un salón | Su número o alias en el texto: "3308", "sala de cómputo" | Datos de ese salón |
| Un piso | "piso 3", "piso tres", "tercer piso", "nivel 2", "planta baja" (piso 0) | Un resumen de Llama (promedio, espacios extremos, lo que esté fuera de rango), después una línea por salón y los espacios sin sensor. El visor deja visible solo ese piso |
| Todo el edificio | "todos los salones", "todo el edificio", "el edificio", "todo", "todos", "resumen general" (como palabras completas: "método" no cuenta) | Igual que un piso, con todos los espacios con datos; los que no tienen sensor se cuentan en vez de listarse |

### Respuestas de la herramienta

Se contestan **sin llamar a Llama**, con frases de `replies.py`. Están
escritas en tono de conversación, porque son lo primero que ve alguien nuevo,
y las más frecuentes tienen dos versiones para que el chat no se repita. La
versión depende del texto de la pregunta, así que la misma pregunta siempre
recibe la misma respuesta y una demostración se puede repetir.

| Caso | Ejemplo | Respuesta |
|---|---|---|
| Saludo o pregunta sobre el asistente | "hola", "¿qué puedes hacer?", "¿qué es esto?" | Se presenta: qué sabe y ejemplos de preguntas |
| Agradecimiento | "gracias" | "¡Con gusto! Si quieres saber algo más del edificio, aquí sigo." |
| Fuera de tema (pregunta de chatbot general) | "¿cuál es la capital de Francia?", "escríbeme un poema", "explícame qué es el CO₂", "¿qué opinas de…?" | Que no es un asistente general y qué sí puede responder |
| Pregunta del edificio sin lugar | "¿cómo está el CO₂?", "¿cómo están los valores?" | "¿De qué salón o piso te gustaría saber?". Nunca responde sobre un salón por omisión |
| Salón que no existe | "3310", "2905" | Que no está registrado y qué salones hay en ese piso; o que los salones de Aulas 3 empiezan con 3 |
| Salón sin sensor, de un tipo medido | "3309", "¿cuál fue el máximo de CO₂ del 3301?", "¿cómo va a estar el 3201 en una hora?" | Una **estimación** del generador (lectura, historial o pronóstico), con una nota escrita por el código: que es estimado, de qué salones sale y su error típico. Anomalías no: solo se buscan en lecturas reales |
| Espacio sin sensor de un tipo nunca medido | "sala de juntas 2" | Que no tiene sensor y cuáles sí tienen datos |
| Alias | "3302" | Que corresponde al 3301 (el 3302 no existe físicamente), y su estimación |
| Qué significa una etiqueta | "¿qué significa que un salón esté estimado?", "¿qué diferencia hay entre simulado y estimado?", "¿qué es el error típico?", "¿qué significan las etiquetas?" | La definición de cada etiqueta que nombra, con los datos que la respaldan |
| Por qué un salón lleva una etiqueta | "¿por qué el 3201 sale estimado?", "¿por qué el 3308 dice simulado?" | Primero qué le toca a ese salón (sensor propio, sensor con pocos datos, sin sensor, tipo nunca medido) y de qué salones sale; luego la definición |

Una pregunta de significado necesita la etiqueta ("estimado", "simulado",
"medido", "medidos y estimados", "error típico", "generador") y una forma de
preguntar qué es ("qué significa", "qué es", "por qué", "qué diferencia",
"qué tan confiable"), y gana aunque nombre un salón. Con solo la etiqueta,
"¿cómo está el 3201 estimado?" sigue siendo una lectura.

Si el texto nombra un salón y un piso ("el 3308 del piso 3"), gana el salón.
El lugar sale siempre del texto: la petición solo lleva la pregunta.

Cómo distingue los casos sin lugar (`classifier._detect_chat`): las peticiones
de chatbot general ("escribe", "explícame", "poema", "capital de", "opinas"…)
ganan siempre, aunque mencionen una variable, porque "explícame qué es el CO₂"
pide una lección, no una lectura. Si no, cualquier palabra del edificio (una
variable, "salón", "lecturas", "mañana", "máximo"…) la marca como pregunta del
edificio a la que solo le falta el lugar. Lo que queda es saludo,
agradecimiento o fuera de tema. Las frases genéricas ("cuál es", "hoy") no
cuentan como palabras del edificio, porque también las tiene "¿cuál es la
capital de Francia?".

### Preguntas por una fecha

Una pregunta puede nombrar un momento del semestre (del 14 de septiembre al 6
de diciembre) de varias formas, que se pueden combinar:

| Forma | Ejemplos |
|---|---|
| Fecha | "el 3 de noviembre", "el 10 de nov" |
| Día de la semana | "el martes", "el martes pasado", "el próximo jueves", "el jueves que viene" |
| Día contado desde hoy | "hoy", "ayer", "anteayer", "mañana", "pasado mañana" |
| Hora | "a las 10", "a las 10:30", "a las 5 de la tarde" |
| Parte del día | "en la mañana" (10:00), "al mediodía", "en la tarde" (16:00), "en la noche" (20:00) |

**Qué martes es "el martes":** lo decide el tiempo del verbo. En pasado o en
presente ("¿cómo estuvo el martes?") es el martes más reciente, hoy incluido; en
futuro ("¿cómo estará el martes?"), el siguiente. "Pasado" y "próximo" mandan
sobre el verbo. Todo se cuenta desde el "ahora" (la última lectura medida), y
la respuesta muestra la fecha que usó, así que una confusión se ve.

Sin hora se toma el mediodía, salvo hoy, que es ahora; en una pregunta de
historial, el día completo. Una pregunta de pronóstico con fecha se responde
con el estado de ese momento, y el visor la marca como lectura. Sin fecha, todo
sigue como antes.

Cada respuesta tiene un solo origen, que el código decide (`api/sources.py`)
y escribe en el prompt, en una nota debajo de la respuesta y en el visor:

| Origen | Cuándo |
|---|---|
| medido | El sensor midió ese momento (o ese periodo) |
| simulado | El salón tiene sensor, pero no midió ese momento: generador con sus propios parámetros |
| estimado | El salón no tiene sensor, o tiene uno sin datos suficientes (3304): generador con los parámetros de su tipo |

En un piso o en el edificio, cada salón lleva su marca: "3201 (estimado)".
Fuera del semestre, o en un espacio que el generador no estima, el chat dice
de qué fechas sí hay datos. Todo sale de `data/processed/generated.parquet`,
que escribe `python -m generator.generate`.

**Lo que todavía no puede:** recordar la pregunta anterior ("¿y en una
hora?"), comparar dos salones, rangos de días ("toda la semana"), más de una
variable en el histórico, detectar sensores atascados, y datos en
vivo: "ahora" es la última lectura del archivo limpio. Hay datos de 3206,
3208, 3304 (sin CO₂ casi siempre), 3305, 3308 y la oficina de mentores del piso 4
(solo 2 lecturas, del 2 de octubre).

**Fallas conocidas de redacción:** Llama omite a veces la advertencia de la
predicción iterativa (el visor la muestra igual, debajo de la respuesta) y a
veces omite el nombre del salón extremo en el resumen.

**Ya corregidas, quitándole la decisión al modelo:**

| Falla observada | Corrección |
|---|---|
| "Disminuirá de 665 a 728 ppm" (sube) | La dirección va escrita en el prompt: "sube 62.8 ppm" |
| "Espacio 308", "espacio 5", "space_4_2" | El prompt nombra cada espacio por su número de salón o por su nombre ("oficina mentores") |
| Reportaba las cinco variables cuando se preguntó por una | La lectura actual solo lleva las variables preguntadas |
| Inventó los salones 3310 a 3334 como "espacios sin datos" | Llama solo escribe el resumen; el detalle por salón y la lista de espacios sin sensor los escribe el código |
| Respuestas de piso cortadas a la mitad y adjetivos exagerados ("632 ppm es alta") | Mismo arreglo: el resumen es corto y solo puede usar los adjetivos de la sección FUERA DE RANGO del prompt |

---

## El reparto de trabajo

Una pregunta recorre cuatro pasos y cada uno tiene un dueño distinto:

```
llm/classifier.py     decide de qué tipo es la pregunta y de dónde   (reglas, sin LLM)
llm/vocabulary.py       las palabras que busca el clasificador
api/retrieval.py      recupera los datos de un salón en un momento    (GRU, datos, detector)
api/floors.py           lo mismo para un piso o el edificio
api/sources.py          decide el origen: el sensor, o el generador
llm/context.py        arma el prompt de un salón                     (plantillas)
llm/floor_prompt.py   arma el prompt de un piso o del edificio,
                        y escribe el detalle por salón
llm/formatting.py       cómo se escriben números, rangos y tendencias
llm/llm_client.py     pide la frase al modelo                        (HTTP a Ollama)
llm/replies.py        frases fijas de la herramienta                 (sin LLM)
api/api.py            rutea entre todos los anteriores y sirve el visor
```

El registro de salones (`config/spaces.json`) se lee en un solo lugar,
`common/spaces.py`: los nombres para el clasificador, la etiqueta de cada
espacio ("3308", "oficina mentores") y a qué piso pertenece. Los espacios sin
número de salón, como las oficinas y salas del piso 4, llevan su nombre en el
campo `label`; ese nombre también se puede escribir en el chat.

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

Las palabras viven en `llm/vocabulary.py` y las reglas en
`llm/classifier.py`: para que se reconozca un sinónimo nuevo basta con
agregarlo a la lista correspondiente. No hay lista para "lectura actual": es lo
que recibe cualquier pregunta sin palabras de anomalía, pronóstico o historial.

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
| Detector de anomalías 3σ | **real**, sobre la última lectura; resultados en `src/anomaly/README.md` |
| Llama 3.1 8B | **real**, vía Ollama (3 a 7 s por respuesta en la RTX 4090) |

Cuando haya acceso a la base de datos del Tec, lo único que cambia es
`space_rows()` en `api/retrieval.py`, marcado `TODO E5`: todo lo demás,
incluidas las preguntas por piso, lee los datos a través de esa función.

### Sobre el detector de anomalías

`anomaly_data()` (en `api/retrieval.py`) revisa la última lectura con la regla
3σ de `src/anomaly/detector.py`. Distingue tres estados y nunca los mezcla: no
se pudo revisar (`unavailable`, con la razón), revisó y no encontró nada, y
revisó y encontró algo. Una lista vacía de anomalías solo aparece cuando el
detector sí corrió: decir "no hay nada" sin haber revisado le pondría al
usuario un «todo bien» falso enfrente, y esa es la respuesta que esta capa no
debe producir nunca.

Qué detecta y qué no, medido con anomalías inyectadas, está en
[src/anomaly/README.md](../anomaly/README.md).

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

**Los umbrales de confort no están fijados.** `llm/formatting.py` anota las lecturas
con bandas de CO₂ (800 / 1000 / 1500 ppm) y rangos de temperatura y humedad
que son valores de trabajo, no tomados de una norma. La sección 4.3.4 no los
fija. Cuando se decidan, su lugar es `common/config.py`, junto a los límites
físicos, no aquí.

**Los nombres de salón y de piso salen de `config/spaces.json`**, leído por
`common/spaces.py`. El clasificador toma de ahí los números de salón, los alias
y a qué piso pertenece cada espacio; agregar un alias ahí basta para que se
reconozca (después de reiniciar la API).

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
