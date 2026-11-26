# Generador de datos ambientales

Implementa la sección 4.4 del documento: un modelo de caja gris por variable,
con estructura física y parámetros calibrados con los salones que tienen
sensor. Sirve para estimar los espacios sin sensor y para simular el semestre
completo. **No sirve para evaluar modelos:** calificar el GRU con datos del
generador mediría qué tan bien imita las ecuaciones, no el edificio.

## Archivos

Un archivo por ecuación de la sección 4.4.2, más los que las juntan:

| Archivo | Qué hace |
|---|---|
| `occupancy.py` | N(t): los alumnos de cada espacio cada 5 minutos, a partir del horario |
| `co2.py` | dC/dt = λ(C_ext − C) + gN, el balance de masa |
| `humidity.py` | El mismo balance en humedad absoluta, con el exterior como frontera |
| `temperature.py` | T = T̄(s,h) + βN + ε, el modelo empírico |
| `light.py` | L = ℓ[s(t)] + c(t)·k·R(t) + ε, el modelo de estados |
| `motion.py` | Movimiento desde N(t), con las tasas de error del sensor |
| `noise.py` | El ε de todas: lo que la física no explica, como proceso AR(1) |
| `inputs.py` | La rejilla de 5 minutos de un salón, con lecturas, alumnos y clima |
| `room.py` | `RoomGenerator`: las cinco ecuaciones juntas, paso por paso, como un agente de Mesa |
| `calibrate.py` | Calibra cada espacio y guarda `models/generator_params.json` |
| `generate.py` | Genera cualquier rango de fechas del semestre |
| `validate.py` | Valida dejando un salón fuera |

## Cómo correrlo

Desde `src/`, con los datos limpios y el clima ya descargados
(`python -m ingestion.cleaning ...` y `python -m ingestion.weather`):

```bash
python -m generator.calibrate                                     # parámetros de todos los espacios
python -m generator.generate --since 2026-11-02 --until 2026-11-08  # una semana del periodo 3
python -m generator.validate                                      # dejando un salón fuera
```

Cada ecuación se puede revisar sola contra los datos: `python -m generator.co2`,
`python -m generator.light`, etc.

## Cómo se calibra cada variable

| Variable | Parámetros | De dónde salen |
|---|---|---|
| Ocupación | alumnos por bloque y día | Horario; un bloque se quita de un día de la semana si los sensores vieron el salón vacío (sin movimiento y con las lámparas apagadas) |
| CO₂ | λ, C_ext | Las caídas del CO₂ con el salón vacío, ajustadas a la exponencial (ASTM E741) |
| CO₂ | g | Mínimos cuadrados sobre los intervalos con clase |
| Humedad | g_h | Igual que g; λ es la del CO₂, porque es el mismo aire el que sale |
| Temperatura | T̄ por hora, β | Un ajuste de mínimos cuadrados por salón |
| Luz | ℓ apagadas / encendidas | Las noches: sin sol, el sensor ve solo las lámparas |
| Luz | k | Mediana de luz/radiación en fines de semana |
| Luz | c | Horas de clase con sol |
| Luz | cambios de estado | Conteo por hora, con y sin clase |
| Movimiento | P(dispara), nivel | Con y sin clase |
| Todas | ε | AR(1) de los residuos |

Con la primera semana de datos:

| Salón | λ (1/h) | C_ext (ppm) | g (ppm/h por alumno) | β (°C por alumno) | Lámparas encendidas | k | c |
|---|---|---|---|---|---|---|---|
| 3206 | 1.38 | 407 | 90 | 0.076 | 52% | 3 | 1.00 |
| 3208 | 1.50 | 413 | 63 | 0.094 | 27% | 13 | 0.97 |
| 3305 | 1.08 | 387 | 57 | 0.053 | 10%* | 120 | 0.37 |
| 3308 | 0.79 | 428 | 27 | 0.051 | 40% | 40 | 0.94 |

\*El 3305 casi nunca tiene las lámparas encendidas de noche; su nivel es el
mismo umbral de 10%, la estimación más cautelosa.

## Qué espacios se generan

`calibrate.py` reparte así:

- **Salón medido:** con sus propios parámetros. Sus valores son "simulado".
- **Salón sin sensor de un tipo medido:** con el promedio de los salones
  medidos de su tipo. Sus valores son "estimado".
- **Tipo nunca medido** (laboratorios, salas de juntas, oficinas): no se
  genera, porque no hay nada en qué basarlo. Hoy son 18 espacios, entre ellos
  todo el piso 4: la oficina de mentores tiene solo 2 lecturas.

El 3304 tiene sensor, pero su sensor de gas está muerto y sus demás
variables tienen menos de un día de lecturas, así que se genera completo con
los parámetros de su tipo.

## En el chat

Una pregunta por un salón sin sensor de un tipo medido ("¿cómo está el
3201?") se responde con una estimación (`src/api/estimates.py`). El generador
recorre las últimas 24 horas del salón con su horario y el clima de esas
horas, hasta el mismo "ahora" que los salones medidos. El último paso es la
lectura estimada y las 24 horas son el historial. La etiqueta va en tres
lugares: en el prompt, en una nota que escribe el código después de la
respuesta de Llama, y en la línea de abajo del visor ("estimado, sin
sensor"). El error esperado de la nota es el de la validación de abajo.

No hay pronóstico del GRU para esos salones, porque el GRU solo conoce los
medidos, ni revisión de anomalías, porque en una estimación no hay nada que
pueda fallar: el chat explica por qué. Los espacios de tipos nunca medidos
siguen respondiendo "no tiene sensor".

Para probarlo en el visor hay que haber corrido `python -m generator.calibrate`
y `python -m generator.validate`, y reiniciar la API.

## Validación

Con la primera semana de datos, dejando cada salón fuera (calibrar el tipo
con los otros tres y generar el cuarto), promedio de 10 realizaciones.
"Propios" es el mismo salón generado con sus propios parámetros; "fuera", con
los del tipo.

Correlación del perfil del día (medias por hora):

| Variable | Propios | Dejando fuera |
|---|---|---|
| CO₂ | 0.54 a 0.93 | 0.62 a 0.93 |
| Humedad | 0.01 a 0.57 | −0.00 a 0.69 |
| Temperatura | 0.99 a 1.00 | 0.84 a 0.90 |
| Luz | 0.94 a 0.99 | 0.69 a 0.96 |
| Movimiento | 0.76 a 0.94 | 0.78 a 0.93 |

Error medio del perfil del día, dejando fuera: CO₂ de 124 a 289 ppm,
temperatura de 0.8 a 2.0 °C, humedad de 2 a 12 %RH, luz de 4 a 11%.

Lo que dicen:

- **Con sus propios parámetros, el generador reproduce el salón**, salvo la
  humedad relativa: de hora en hora la manda el aire de afuera, no los
  alumnos, y el modelo solo sigue su promedio.
- **En un salón sin sensor el error crece**, sobre todo en temperatura (cada
  salón tiene su propio aire acondicionado), y por ella en humedad relativa.
  El 3305 es el caso extremo: es el más caliente (27.5 °C), y con la
  temperatura del promedio su humedad relativa sale 12 puntos arriba.
- **La luz necesita subtipos.** Al promediar un salón sin ventanas (3206) con
  uno de mucha ventana (3305), el tipo "salón" queda en medio de los dos. El
  diseño de la luz ya proponía heredar por tipo de iluminación (interior,
  ventana, mixto); falta anotar ese tipo en `spaces.json`.

## Decisiones que el documento no fija

1. **ε es un AR(1), no ruido blanco.** El sensor de CO₂ es MOX: los fines de
   semana marca de 1,000 a 2,750 ppm con movimiento 0 y la luz apagada. Es la
   deriva del sensor, no gente. Un residuo así es lento y tiene media propia,
   y un AR(1) calibrado con los residuos lo reproduce. Aun así, la mediana
   generada del CO₂ sale de 90 a 200 ppm más alta que la medida, porque ε
   tiene una sola media para todo el día.
2. **La humedad usa la λ del CO₂**, porque un salón tiene un solo caudal de
   aire.
3. **Sin volúmenes**, g y g_h son parámetros efectivos (G/V y su equivalente
   para el vapor) que se estiman directo de las lecturas. También absorben
   cuántos inscritos asisten de verdad.
4. **k sale de una mediana, no de mínimos cuadrados.** Con mínimos cuadrados,
   una lámpara encendida algún fin de semana le daba k = 7.6 al 3206, que no
   tiene ventanas; con la mediana queda en 3.
5. **Los días de clase salen de la luz y el movimiento, no del CO₂**, porque
   el CO₂ sube aunque no haya nadie.

## Para la versión 4 del documento

- La ecuación del CO₂ con parámetros efectivos λ y g, en lugar de Q/V y G/V,
  y quitar "cualquier espacio del que se conozca su volumen".
- ε como proceso AR(1), con la deriva del sensor MOX como su causa principal.
- La humedad comparte la λ del CO₂.
- Los resultados de la validación dejando un salón fuera, y que la luz
  necesita subtipos de iluminación.
- Qué tipos no se generan: todo lo que no es salón, por ahora.

## Pendiente

- Subtipos de iluminación en `spaces.json` (interior, ventana, mixto).
- Recalibrar con las exportaciones nuevas. Con más semanas se afinan los
  cambios de estado de la luz y se puede calibrar la oficina de mentores, que
  daría el tipo "oficina".
- Una media de ε por hora del día para el CO₂, si la mediana generada
  importa.
