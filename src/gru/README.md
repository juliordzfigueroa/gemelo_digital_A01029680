# Modelo predictivo GRU

Implementación de la sección 4.2 del documento de investigación: GRU de dos
capas [128, 64] para predecir 5 variables ambientales 30 minutos hacia
adelante, a partir de 1 hora de contexto.

## Archivos

| Archivo | Qué hace |
|---|---|
| `config.py` | Mapeo de variables, cotas físicas de los sensores, hiperparámetros. **Sin medias ni sigmas**: todo parámetro derivado de datos se calcula y se guarda aparte. |
| `sintetico.py` | Genera datos de prueba que cumplen el contrato, con los mismos defectos esperados de los sensores reales. |
| `datos.py` | Bloques contiguos, rejilla regular, partición temporal, normalización y ventanas. |
| `modelo.py` | La arquitectura, con las dos implementaciones recurrentes. |
| `metricas.py` | MAE, RMSE, MAPE, sMAPE, R², por variable y por horizonte. |
| `entrenar.py` | Loop de entrenamiento con early stopping, evaluación y guardado. |
| `comparar_implementaciones.py` | Entrena las dos celdas recurrentes con la misma semilla y compara. |

## Cómo correrlo

```bash
cd src/gru
python entrenar.py --prueba-sintetica                        # verificación
python entrenar.py --dias 60                                 # entrenamiento
python entrenar.py --implementacion pytorch --dias 60        # versión rápida
python entrenar.py --datos ../../data/raw/aulas3.parquet     # datos reales
python comparar_implementaciones.py                          # cho vs pytorch
```

## El contrato de datos

El pipeline no sabe de dónde salen los datos. Cualquier fuente debe entregar un
DataFrame con estas columnas:

| Columna | Tipo | Nota |
|---|---|---|
| `timestamp` | datetime | hora local, intervalos de 5 min |
| `espacio_id` | int | identificador del espacio |
| `co2` | float | ppm |
| `temperatura` | float | °C |
| `humedad` | float | %RH |
| `luz` | float | lux |
| `movimiento` | float | 0 o 1 |

Los huecos se expresan como **filas ausentes**, no como NaN: es lo que pasa de
verdad cuando un nodo pierde la red. El día que existan los datos de Aulas 3,
lo único que cambia es `--datos`.

## Arquitectura

```
entrada   (lote, 12, 9)        5 variables + 4 rasgos de tiempo
+ embedding del espacio_id     8 dimensiones, concatenado en cada paso
  GRU 1   (lote, 12, 128)
  dropout 0.2
  GRU 2   (lote, 12, 64)
  último paso → lineal
salida    (lote, 6, 5)         6 pasos × 5 variables, sin activación
```

95,662 parámetros entrenables.

### Las dos implementaciones recurrentes

`config.IMPLEMENTACION_GRU` elige entre dos celdas con la misma arquitectura:

| Valor | Qué es |
|---|---|
| `"cho"` *(por omisión)* | `CeldaGRU`, las ecuaciones 1 a 4 de la sección 4.2.7.2 escritas literalmente. Las ecuaciones que publica la tesis son las que producen los resultados. |
| `"pytorch"` | `nn.GRU`, la variante fusionada de cuDNN. Unas 15× más rápida, pero no es la misma formulación. |

Las dos diferencias con `nn.GRU`, que son la razón de que `CeldaGRU` exista:

1. **Dónde se aplica el reset.** La ecuación 3 multiplica `r_t ⊙ h_{t-1}` y
   después aplica `W_h`. `nn.GRU` calcula `r_t ⊙ (W_hn·h_{t-1} + b_hn)`:
   la transformación lineal primero y el reset después. Son operaciones
   genuinamente distintas.
2. **El sentido de la compuerta update.** La ecuación 4 trata `z_t` como
   "cuánto tomar de lo nuevo"; `nn.GRU` lo trata como "cuánto conservar de lo
   viejo". Esta sí es solo convención de signo.

`CeldaGRU.reiniciar_parametros()` replica el esquema de inicialización de
`nn.GRU` — uniforme en ±1/√(unidades) — para que la comparación entre las dos
sea justa: una diferencia en resultados debe venir de la formulación, no de un
punto de partida distinto.

Otras dos decisiones que no son obvias desde el documento:

**Las capas van como dos módulos separados.** Con `--implementacion pytorch`
esto es obligado: PyTorch fuerza a que todas las capas de un mismo `nn.GRU`
compartan `hidden_size`, y la especificación pide [128, 64]. Por lo mismo el
dropout va como módulo explícito, porque el argumento `dropout` de `nn.GRU`
solo actúa entre capas internas.

**El `espacio_id` entra como embedding, no como one-hot.** El salón 3302 no es
"más" que el 3301, así que no puede entrar como número. Un one-hot crecería con
el número de salones del edificio; el embedding no, y aprende qué espacios se
parecen entre sí.

**Rasgos de tiempo cíclicos.** La hora entra como seno y coseno en vez de como
entero, porque las 23:55 están a un paso de las 00:00 pero como número están a
23 de distancia.

## Lo que está deliberadamente explícito

**La normalización se ajusta solo con entrenamiento.** Calcular la media y la
sigma sobre todo el dataset mete información del futuro en el preprocesamiento,
y las métricas salen mejores de lo que son. `Normalizador.ajustar()` recibe solo
la partición de entrenamiento.

**La partición es temporal, no aleatoria.** Una partición al azar dejaría el
paso *t* en entrenamiento y el *t+1* en prueba: el modelo se evaluaría contra
datos casi idénticos a los que vio.

**Las ventanas no cruzan bloques.** Unir dos tramos separados por días produce
ejemplos falsos. Una ventana es válida solo si sus 18 pasos caen en el mismo
bloque contiguo, sin faltantes y en la misma partición.

**El MAPE reporta su cobertura.** Se indefine donde el valor real es 0, y eso
pasa en el 39% de las lecturas de luz y en más de la mitad de las de
movimiento. Se calcula sobre las observaciones distintas de cero y se reporta
qué fracción cubrió, en vez de presentar como global un porcentaje sacado de
media muestra.

## Decisiones abiertas

**`movimiento` es binaria y se está prediciendo con MSE.** La tabla 1 de la
sección 5.3 del documento recomienda sigmoide para variables en [0, 1], pero la
capa de salida va sin activación por ser regresión continua. La consecuencia se
mide: en la prueba sintética, `movimiento` sale con R² de 0.56 y sMAPE de 133%,
contra R² de 0.99 en CO₂. La alternativa limpia es una cabeza aparte con
`BCEWithLogitsLoss` y reportarle precisión/recall en lugar de MAE. Sin resolver.

**Normalización global contra por espacio.** `config.py` sigue al documento
(global por variable), pero `--normalizar-por-espacio` existe porque las
escalas de iluminancia difieren en dos órdenes de magnitud entre espacios y lo
global colapsa los espacios oscuros cerca de cero. Falta comparar las dos.

## Pendiente

- Nota al pie en la tesis sobre la segunda diferencia con `nn.GRU` (el sentido
  de la compuerta update), por si alguien compara el código con la ecuación 4.
- Baseline SARIMAX, en `src/sarimax/`, sobre las mismas particiones y métricas.
  Antes hay que resolver lo de la estacionalidad: `s=288` tarda 104 s por modelo
  contra 0.3 s de los términos de Fourier.
- Predicción iterativa hasta 2 horas (sección 4.2.5 del documento).
- Detector de anomalías 3σ, en `src/anomalias/`.
- Extraer a `src/comun/` lo que compartan GRU y SARIMAX, para que compitan con
  las mismas particiones y el mismo código de métricas.
