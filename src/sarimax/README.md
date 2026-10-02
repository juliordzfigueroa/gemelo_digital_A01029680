# Baseline SARIMAX

Implementación de la sección 4.2.4 y 4.2.5 del documento: modelo estadístico
contra el cual comparar el GRU.

## Qué es y qué no es

SARIMAX es la **vara de medición**, no un componente del GRU ni su maestro. Los
dos modelos se entrenan por separado y con criterios distintos:

| | GRU | SARIMAX |
|---|---|---|
| Criterio | Minimiza MSE | Maximiza verosimilitud |
| Método | Backpropagation, paso a paso | Filtro de Kalman, de una sola vez |
| Alcance | Un modelo global multivariado | Un modelo por variable y espacio |
| Para 5 variables y 5 espacios | 1 modelo | **25 modelos** |

**No hay una función de pérdida que viaje de uno al otro.** SARIMAX no tiene
pesos que se actualicen por gradiente, así que no hay nada que pasar. Y aunque
lo hubiera, entrenar el GRU para imitar a SARIMAX le pondría como techo el
desempeño del baseline, que es exactamente lo contrario de para qué sirve un
baseline.

Lo que sí comparten, y es lo que hace comparables los números: los mismos datos,
las mismas particiones, las mismas ventanas de prueba y el mismo código de
métricas. `entrenar.py` verifica esa igualdad y aborta si no se cumple.

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

Medido sobre los datos del proyecto, no estimado: son **2,600x**. Y hacen
falta 25 modelos, lo que da unas 15 horas contra 26 segundos.

**La alternativa usa una parte de la misma ecuación que ya está ahí.** En vez de
pedirle al modelo que mire 288 pasos atrás, entran términos de seno y coseno con
periodo 288 por la vía de las variables exógenas, que la ecuación general ya
lleva como `Σ βᵢ Xᵢ,ₜ`. Captura la misma periodicidad diaria con 2K regresores en
lugar de 289 dimensiones de estado.

Las dos están disponibles con `--estacionalidad`, para que el costo se mida y no
se afirme.

## Cómo correrlo

Desde `src/`:

```bash
python -m sarimax.entrenar                              # Fourier, por omisión
python -m sarimax.entrenar --estacionalidad estacional  # s=288, muy lento
python -m sarimax.entrenar --orden 1,0,1                # otro orden (p,d,q)
python comparar_modelos.py                              # GRU contra SARIMAX
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
GRU, que también ve 12 pasos reales y tiene que producir los 6 siguientes solo.

**El orden `(p, d, q)` es un parámetro, no una decisión enterrada.** El valor por
omisión es `(2, 0, 2)`. El documento pide determinarlo por análisis de ACF y
PACF sobre el conjunto de entrenamiento; eso sigue pendiente.

## Pendiente

- Selección de órdenes por ACF y PACF, como pide la sección 4.2.7.1.
- Decidir con la profesora si la tesis adopta la formulación con Fourier, lo que
  implica corregir la ecuación de la sección 4.2.7.1.
