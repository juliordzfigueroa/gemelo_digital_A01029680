# Modelo predictivo GRU

Implementación de la sección 4.2 del documento de investigación: GRU de dos
capas [128, 64] para predecir 5 variables ambientales 30 minutos hacia
adelante, a partir de 1 hora de contexto.

## Archivos

| Archivo | Qué hace |
|---|---|
| `model.py` | La arquitectura. La celda recurrente (`GRUCell`) son las ecuaciones 1 a 4 de la sección 4.2.7.2 escritas literalmente. |
| `train.py` | Entrenamiento con early stopping, evaluación sobre el día de prueba y guardado en `models/`. |
| `predict.py` | Pronóstico de un salón desde sus últimas lecturas: directo (30 min) o iterativo (hasta 2 h). Lo usa la API. |
| `check.py` | Prueba con una serie conocida: si la red no la aprende, el error está en el código. |

Lo que comparte con SARIMAX vive en `src/common/`: el contrato de datos
(`config.py`), la preparación y las ventanas (`data.py`) y las métricas
(`metrics.py`).

## Cómo correrlo

Desde `src/`:

```bash
python -m gru.check                          # ¿la red aprende?
python -m gru.train                          # entrena y evalúa (~1 min)
python -m gru.predict --space 308 --horizon 60
```

## Arquitectura

```
entrada   (lote, 12, 9)        5 variables + 4 rasgos de tiempo
+ embedding del space_id       8 dimensiones, concatenado en cada paso
  GRU 1   (lote, 12, 128)
  dropout 0.2
  GRU 2   (lote, 12, 64)
  último paso → lineal
salida    (lote, 6, 5)         6 pasos × 5 variables, sin activación
```

95,118 parámetros entrenables.

### Por qué la celda está escrita a mano

PyTorch tiene su propia celda GRU (`nn.GRU`), más rápida, pero no es la
formulación del documento:

1. **Dónde se aplica el reset.** La ecuación 3 multiplica `r_t ⊙ h_{t-1}` y
   después aplica `W_h`. `nn.GRU` calcula `r_t ⊙ (W_hn·h_{t-1} + b_hn)`: la
   transformación lineal primero y el reset después. Son operaciones
   genuinamente distintas.
2. **El sentido de la compuerta update.** La ecuación 4 trata `z_t` como
   "cuánto tomar de lo nuevo"; `nn.GRU` lo trata como "cuánto conservar de lo
   viejo". Esta sí es solo convención de signo.

Escribirla a mano garantiza que las ecuaciones publicadas en la tesis son las
que producen los resultados. El costo es velocidad: un bucle de Python por paso
de tiempo no puede usar el núcleo fusionado de cuDNN. Con los datos actuales
son unos 60 s de entrenamiento en lugar de 6.

Hasta noviembre de 2026 convivían las dos implementaciones. Con la misma
semilla y los mismos datos dieron resultados equivalentes (MAE global 26.07
con la celda a mano contra 26.47 con `nn.GRU`), y se quedó solo la celda del
documento.

### Otras decisiones

**Las capas van como dos módulos separados** porque la especificación pide
tamaños distintos, [128, 64], y el dropout va como módulo explícito entre
ellas.

**El `space_id` entra como embedding, no como one-hot.** El salón 3302 no es
"más" que el 3301, así que no puede entrar como número. Un one-hot crecería con
el número de salones del edificio; el embedding no, y aprende qué espacios se
parecen entre sí.

**Rasgos de tiempo cíclicos.** La hora entra como seno y coseno en vez de como
entero, porque las 23:55 están a un paso de las 00:00 pero como número están a
23 de distancia.

## Lo que está deliberadamente explícito

**La normalización se ajusta solo con entrenamiento.** Calcular la media y la
sigma sobre todo el dataset mete información del futuro en el preprocesamiento,
y las métricas salen mejores de lo que son. `Normalizer.fit()` recibe solo la
partición de entrenamiento.

**La partición es por días completos, no aleatoria.** Una partición al azar
dejaría el paso *t* en entrenamiento y el *t+1* en prueba. Validación y prueba
son los últimos días entre semana; los fines de semana se quedan en
entrenamiento, porque lo que importa predecir son los días con clase.

**Las ventanas no cruzan bloques.** Unir dos tramos separados por días produce
ejemplos falsos. Una ventana es válida solo si sus 18 pasos caen en el mismo
bloque contiguo, sin faltantes y en la misma partición. Esa regla vive solo en
`common/data.py`; SARIMAX usa las mismas ventanas.

**El MAPE reporta su cobertura.** Se indefine donde el valor real es 0, lo que
pasa con la luz de noche y con el movimiento casi siempre. Se calcula sobre las
observaciones distintas de cero y se reporta qué fracción cubrió.

## Decisiones abiertas

**`motion` es binaria y se está prediciendo con MSE.** La tabla 1 de la
sección 5.3 del documento recomienda sigmoide para variables en [0, 1], pero la
capa de salida va sin activación por ser regresión continua. La alternativa
limpia es una cabeza aparte con `BCEWithLogitsLoss` y reportarle precisión y
recall en lugar de MAE. Sin resolver.

## Pendiente

- Nota al pie en la tesis sobre la segunda diferencia con `nn.GRU` (el sentido
  de la compuerta update), por si alguien compara el código con la ecuación 4.
- Detector de anomalías 3σ.
- Reentrenar con más semanas de datos: con una semana, SARIMAX todavía gana.
