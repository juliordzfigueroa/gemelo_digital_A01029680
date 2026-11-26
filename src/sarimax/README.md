# Baseline SARIMAX

Implementación de las secciones 4.2.4 y 4.2.5 del documento: modelo
estadístico contra el cual comparar el GRU.

## Archivos

| Archivo | Qué hace |
|---|---|
| `model.py` | Un modelo SARIMAX para una variable de un salón, con el ciclo diario como términos de Fourier. |
| `train.py` | Ajusta un modelo por variable y salón (hoy 20) y los evalúa sobre las mismas ventanas de prueba que el GRU. |

## Qué es y qué no es

SARIMAX es la **vara de medición**, no un componente del GRU ni su maestro. Los
dos modelos se entrenan por separado y con criterios distintos:

| | GRU | SARIMAX |
|---|---|---|
| Criterio | Minimiza MSE | Maximiza verosimilitud |
| Método | Backpropagation, paso a paso | Filtro de Kalman, de una sola vez |
| Alcance | Un modelo global multivariado | Un modelo por variable y espacio |
| Con la primera semana de datos | 1 modelo | **20 modelos** (5 variables × 4 salones) |

**No hay una función de pérdida que viaje de uno al otro.** SARIMAX no tiene
pesos que se actualicen por gradiente, así que no hay nada que pasar. Y aunque
lo hubiera, entrenar el GRU para imitar a SARIMAX le pondría como techo el
desempeño del baseline, que es exactamente lo contrario de para qué sirve un
baseline.

Lo que sí comparten, y es lo que hace comparables los números: los mismos
datos, las mismas particiones, las mismas ventanas de prueba y el mismo código
de métricas. Las ventanas no se reconstruyen aquí: `train.py` usa la misma
clase del GRU (`common.data.EnvironmentalWindows`), y de ella toma también la
respuesta correcta de cada ventana, así que son las mismas por construcción.

**Por qué 4 salones y no 6.** El GRU se entrena con los 6 salones que tienen
datos, pero SARIMAX solo se ajusta donde hay algo que calificar. El día de
prueba es el viernes 9 de octubre, y en él no hay lecturas del 304 (su sensor
dejó de reportar el día 7) ni del 402 (solo tiene 2 lecturas, del día 2).

## Estacionalidad: por qué no `s=288`

La sección 4.2.7.1 especifica un periodo estacional `s=288`, que es un día a
muestreo de 5 minutos. Es correcto en principio e inviable en la práctica: la
representación en espacio de estados crece con `s`, así que un término
estacional de 288 arrastra un estado de 289 dimensiones en cada paso del filtro
de Kalman.

Medido sobre una semana de datos, un solo modelo:

| Configuración | Tiempo de ajuste | Dimensión del estado |
|---|---|---|
| `s=288` estacional | **2,147 s** | 289 |
| ARIMAX + Fourier K=4 | **0.82 s** | 3 |

Medido sobre los datos del proyecto, no estimado: son **2,600x**. Con los 20
modelos de hoy serían unas 12 horas solo de ajuste; con Fourier, la corrida
completa (ajuste y pronóstico) tarda alrededor de un minuto. Después de esa
medición el modo `s=288` se quitó del código; estos números son su registro.

**La alternativa usa una parte de la misma ecuación que ya está ahí.** En vez de
pedirle al modelo que mire 288 pasos atrás, entran términos de seno y coseno con
periodo 288 por la vía de las variables exógenas, que la ecuación general ya
lleva como `Σ βᵢ Xᵢ,ₜ`. Captura la misma periodicidad diaria con 2K regresores en
lugar de 289 dimensiones de estado.

## Cómo correrlo

Desde `src/`:

```bash
python -m sarimax.train                 # un modelo por variable y salón, ~1 min
python compare_models.py                # GRU contra SARIMAX
```

## Detalles de implementación

**Los huecos entran como valores faltantes, no concatenando bloques.** SARIMAX
supone observaciones equiespaciadas, así que pegar dos tramos separados por dos
días falsearía el eje temporal. Cada espacio se coloca en una rejilla continua
de 5 minutos con `NaN` donde no hay dato: el filtro de Kalman maneja
observaciones ausentes de forma nativa, arrastrando el estado sin fingir que
llegaron lecturas.

**El pronóstico es dinámico desde cada origen.** Para cada ventana de prueba, el
modelo ve los valores reales hasta el paso 12 y produce los 6 siguientes con su
propia salida realimentada. Es exactamente la tarea sobre la que se evalúa el
GRU.

**Estacionariedad forzada.** Con `d=0`, el modelo supone que la serie es
estacionaria. Sin forzarlo, la temperatura y la humedad dieron raíces AR de
1.000 y 1.057, un proceso explosivo cuyos pronósticos a 30 minutos llegaron a
1e76.

**El orden `(p, d, q)` es una constante, `ORDER = (2, 0, 2)` en `model.py`.**
El documento pide determinarlo por análisis de ACF y PACF sobre el conjunto de
entrenamiento; eso sigue pendiente.

## Pendiente

- Selección de órdenes por ACF y PACF, como pide la sección 4.2.7.1.
- Decidir con la profesora si la tesis adopta la formulación con Fourier, lo que
  implica corregir la ecuación de la sección 4.2.7.1.
