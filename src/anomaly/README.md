# Detector de anomalías 3σ

Implementa la regla de la sección 4.2.5 del documento: cada lectura se compara
contra la media y la desviación estándar de su variable, calculadas sobre el
conjunto de entrenamiento, y lo que se aleje más de 3σ es una anomalía. No hay
ningún modelo entrenado de por medio.

## Archivos

| Archivo | Qué hace |
|---|---|
| `detector.py` | La regla. Usa la misma media y sigma que el normalizador del GRU (`models/gru_normalizer.json`), calculadas con el mismo entrenamiento, así que solo se calculan en un lugar. |
| `evaluate.py` | Precisión y recall, con anomalías sintéticas inyectadas en el conjunto de prueba, como pide la sección 4.2.5. |

## Qué revisa

CO₂, temperatura, humedad y luz. El movimiento no: es binario, y una banda de
3σ alrededor de una variable que solo vale 0 o 1 cae fuera de [0, 1].

Bandas con la primera semana de datos (media ± 3σ del entrenamiento):

| Variable | Banda normal |
|---|---|
| CO₂ | −520 a 2,251 ppm |
| Temperatura | 21.5 a 30.1 °C |
| Humedad | 28.2 a 59.3 %RH |
| Luz | −48 a 76 % |

## En el chat

"¿Hay algo raro en el 3208?" revisa la última lectura de ese salón; si la
pregunta es de un piso o del edificio, la de cada salón con datos. Hay tres
respuestas distintas y nunca se confunden: no se pudo revisar (por ejemplo, si
el GRU no está cargado o solo se preguntó por el movimiento), revisó y no
encontró nada, o revisó y encontró algo.

El prompt lleva también los rangos de confort, marcados como una referencia
aparte del detector: 1,162 ppm de CO₂ está dentro de la banda estadística y aun
así es "ALTO" para una persona. Sin esa línea, Llama leería "sin anomalías"
como "todo bien".

Si la lectura de un salón es más vieja que la del resto del piso (un sensor que
dejó de reportar), la respuesta dice de cuándo es.

## Cómo correrlo

Desde `src/`:

```bash
python -m anomaly.evaluate
```

Guarda los resultados en `models/anomaly_evaluation.json`.

## Resultados

Primera semana de datos; el conjunto de prueba es el viernes 9 de octubre, con
1,038 lecturas de 5 minutos. Se inyectan tres tipos de anomalía, cada uno sobre
una copia limpia de la prueba y una variable a la vez:

- **Pico:** un salto de un tamaño que pasa en un edificio: CO₂ de +1,000 a
  +3,000 ppm, temperatura de +3 a +8 °C, humedad de +10 a +30 %RH, luz de +40 a
  +90 %.
- **Sensor dañado:** lo que manda un sensor que falla, tal como aparece en las
  exportaciones reales: el −1 del sensor de gas, el 0 de una lectura vacía, o un
  valor por encima del límite del instrumento.
- **Sensor atascado:** el mismo valor repetido de 1 a 3 horas.

Recall (qué fracción de lo inyectado detecta):

| Tipo | CO₂ | Temperatura | Humedad | Luz |
|---|---|---|---|---|
| Pico | 0.79 | 0.65 | 0.79 | 0.39 |
| Sensor dañado | 0.41 | 1.00 | 1.00 | 0.46 |
| Sensor atascado | 0.06 | 0.00 | 0.00 | 0.00 |

Falsas alarmas en la prueba sin tocar: 10 de 1,038 en CO₂, 0 en temperatura y
humedad, 1 en luz. La precisión de los picos y los sensores dañados va de 0.80
a 1.00. Es una cota inferior: cuenta como falsa alarma cualquier marca sobre una
lectura que no se inyectó, aunque sea una rareza real del día.

## Qué dicen los resultados

- **Funciona** para picos grandes y para lecturas vacías de temperatura y
  humedad, con pocas falsas alarmas.
- **No detecta sensores atascados.** Un valor congelado dentro de la banda
  nunca la cruza. La regla obvia, "el mismo valor durante una hora", tampoco
  sirve aquí: la humedad se queda plana una hora o más en 3,765 intervalos de la
  semana, por la resolución del sensor.
- **No detecta el −1 del sensor de gas**: queda a 1.9σ de la media del CO₂. Lo
  detecta la limpieza (`ingestion/cleaning.py`, regla del centinela) antes de
  que el dato llegue a cualquier modelo.
- **La banda del CO₂ es muy ancha**, hasta 2,251 ppm, porque la sigma del
  entrenamiento mezcla clases, noches y fines de semana. Un pico de +1,000 ppm
  sobre una lectura baja no la cruza.
- **La luz da falsas alarmas con el sol.** En toda la semana marcó 15 lecturas,
  14 del 3305 y 12 en fin de semana: luz natural, no fallas. El día de prueba es
  viernes, por eso en la prueba solo aparece una.

## Para la versión 4 del documento

- Quitar "sensores atascados" de lo que detecta la regla 3σ, o describir una
  regla aparte y medirla.
- Aclarar que los valores imposibles bajos (el −1 del sensor de gas) no los
  detecta la regla, sino la limpieza.
- Luz: reportar las falsas alarmas por luz natural y decidir entre bandas por
  salón y hora del día, o sacarla del detector.
- Opcional: bandas por salón y hora del día también para el CO₂, que harían la
  banda mucho más angosta.
