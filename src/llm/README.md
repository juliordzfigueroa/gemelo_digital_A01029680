# Capa conversacional

Implementa la sección 4.3 del documento de investigación: patrón RAG con
Llama 3.1 8B servido por Ollama, orquestado por FastAPI.

---

## El reparto de trabajo

Una pregunta recorre cuatro pasos y cada uno tiene un dueño distinto:

```
clasificador.py   decide de qué tipo es la pregunta       (reglas, sin LLM)
api/api.py        recupera los datos que ese tipo pide    (GRU, BD, detector)
contexto.py       arma el prompt con esos datos           (plantillas)
cliente_llm.py    pide la frase al modelo                 (HTTP a Ollama)
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
anomalía  >  predictiva  >  histórica  >  actual
```

y codifica urgencia, no precisión. La consecuencia visible:

| Pregunta | Se rutea como | Debería ser |
|---|---|---|
| `¿El CO2 va a estar muy alto en una hora?` | `anomalia` | `predictiva` |

«muy alto» gana sobre «en una hora» y la pregunta pierde su horizonte.
Arreglarlo de verdad necesita un clasificador entrenado, no una lista de
palabras más larga. Mientras tanto se prefiere errar hacia la seguridad.

Para ver el ruteo completo:

```powershell
python -m llm.clasificador
```

---

## Qué está conectado y qué no

| Pieza | Estado |
|---|---|
| GRU, predicción directa (30 min) | **real**, desde el checkpoint en `models/` |
| GRU, predicción iterativa (hasta 2 h) | **real** |
| Clasificación de intención | **real** |
| Construcción de prompts | **real** |
| Fuente de datos | **sintética**, en memoria. `TODO E5` |
| Detector de anomalías 3σ | **no existe** |
| Llama 3.1 8B | requiere instalar Ollama |

Los tres métodos marcados `TODO E5` en `api/api.py` son los únicos que cambian
cuando haya acceso a la base de datos del Tec: `_filas_de_espacio()`,
`_datos_actuales()` y `_datos_historicos()`. El resto del orquestador no se
toca.

### Sobre el detector que no existe

`_datos_anomalia()` devuelve `estado: "no_disponible"`, **no una lista vacía de
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
python -m llm.cliente_llm
```

Ollama queda como servicio en segundo plano después de instalarse. Si no
responde, `ollama serve` lo arranca a mano.

`python -m llm.cliente_llm` sondea primero y solo pregunta si el modelo está;
termina en código 1 con el comando que hace falta si algo falta, en vez de
tronar con un error de red.

**El modelo corre local.** Ninguna lectura de Aulas 3 sale de la máquina, que
es la razón por la que el documento eligió un modelo local y no una API
hospedada.

---

## Probar sin Ollama

`/contexto` recorre toda la cadena excepto la llamada al modelo y devuelve el
prompt que se habría enviado. Sirve para dos cosas: demostrar la mitad de
recuperación en una máquina sin Ollama, y separar la culpa cuando una
respuesta sale mal —si el prompt ya traía el dato equivocado, el problema está
en la recuperación, no en el modelo.

```powershell
uvicorn api.api:app --port 8000
```

```powershell
curl -X POST http://localhost:8000/contexto `
  -H "Content-Type: application/json" `
  -d '{\"pregunta\": \"Como va a estar el CO2 del 3302 en 45 minutos?\", \"espacio_id\": 1}'
```

---

## Decisiones abiertas

**Los umbrales de confort no están fijados.** `contexto.py` anota las lecturas
con bandas de CO₂ (800 / 1000 / 1500 ppm) y rangos de temperatura y humedad
que son valores de trabajo, no tomados de una norma. La sección 4.3.4 no los
fija. Cuando se decidan, su lugar es `comun/config.py`, junto a los límites
físicos, no aquí.

**Los IDs de salón son inventados.** `ESPACIOS` en `clasificador.py` mapea
`3302`, `3305`, `3201`, `3105` y `pasillo` a los ids 1 a 5 del generador
sintético. Hay que rehacerlo con los números reales de Aulas 3 y su
correspondencia con `espacio_id` en la base de datos.

**`movimiento` se reporta como número continuo.** El GRU lo predice bajo MSE
junto con las demás variables, así que su salida es un valor sin calibrar
entre 0 y 1, no una probabilidad. El prompt lo etiqueta `(1 = hay movimiento,
0 = no)`, lo que es honesto para una lectura real pero engañoso para una
predicción. Está relacionado con la decisión abierta de `src/gru/README.md`
sobre darle una cabeza aparte con `BCEWithLogitsLoss`.

**El horizonte se redondea hacia arriba en bloques de 30 min.** Pedir 45
minutos corre dos iteraciones (60 min) y recorta a 9 pasos. El modelo solo
sabe emitir bloques de 6 pasos, así que la alternativa sería entrenar cabezas
de otro tamaño.
