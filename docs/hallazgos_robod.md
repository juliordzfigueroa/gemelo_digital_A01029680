# Hallazgos sobre el dataset ROBOD

ROBOD se evaluó como dataset de desarrollo y **se descartó**: el esfuerzo de
corregir los defectos de un dataset ajeno no se justificaba frente a esperar los
datos reales de Aulas 3. Los datos y los scripts de limpieza se eliminaron del
repositorio; este documento conserva los hallazgos porque varios son reutilizables
(los mismos defectos van a aparecer con los sensores propios) y porque el
corrimiento de la iluminancia es un error del dataset publicado que vale reportar.

Lo evaluado fue la versión v7 de Figshare, descargada del espejo de GitHub
`ideas-lab-nus/robod` (commit `e9d1abe`) el 2026-10-01. Los CSV se pueden
recuperar desde ahí; las sumas SHA256 de abajo permiten verificar que sea la
misma versión.

## Identificación

| Campo | Valor |
|---|---|
| Nombre | ROBOD — Room-level Occupancy and Building Operation Dataset |
| Autores | Z. D. Tekler, E. Ono, Y. Peng, S. Zhang, B. Lasternas, A. Chong (IDEAS Lab, National University of Singapore) |
| Registro citable | Figshare, DOI `10.6084/m9.figshare.19234530.v7` (versión 7, publicada 2022-07-11) |
| Licencia | **CC BY 4.0** — https://creativecommons.org/licenses/by/4.0/ |
| Paper del dataset | *Building Simulation*, Tsinghua University Press, 2022 — DOI `10.1007/s12273-022-0925-9` |
| Edificio | School of Design and Environment 4 (SDE4), NUS, Singapur |
| Periodo | 2021-09-07 00:00 a 2021-12-23 23:55, UTC+08:00 |
| Intervalo de muestreo | 5 minutos |

## Cita obligatoria (CC BY exige atribución)

> Tekler, Zeynep Duygu, et al. "ROBOD, room-level occupancy and building
> operation dataset." *Building Simulation*. Tsinghua University Press, 2022.

## Descarga efectuada

Fuente usada: espejo en GitHub `ideas-lab-nus/robod`, rama `master`,
directorio `Data/`, vía `raw.githubusercontent.com`.

- Commit más reciente que toca `Data/`: `e9d1abecdc2836060067f2c9871135ee9eb1c94e`
- Fecha de descarga: 2026-10-01
- Tamaño total: 19.8 MB (5 archivos CSV sin comprimir)

Nota: el repositorio de GitHub **no incluye archivo LICENSE**. La licencia
aplicable es la declarada en el registro de Figshare (CC BY 4.0), que es
también la fuente a citar.

### SHA256

```
34a3d7a1946945c5321f06fbe38639d74e47e40fdd0df086c635d3ed73a63f3d  combined_Room1.csv
ac9a34cd7149814bc31be26c75d0481c6c205e3cc0e0775d42f963b86a90941f  combined_Room2.csv
be7dcfabe03ee398a54c680fc338ac1768d2a099dff14f9f3f2baab92bdddaab  combined_Room3.csv
efb330e418cc99ccd90ccedb5cc2680c9e698f3534e5960a3bc59aa89673ffe7  combined_Room4.csv
3105abcc33ac2a70389b4249f0df2c9345ea2e47c307fcf7bb89c045f49666ec  combined_Room5.csv
```

Los tamaños en bytes coinciden con los reportados por la API de GitHub.

## Estructura verificada

| Archivo | Tipo de espacio | Filas | Columnas | Días |
|---|---|---|---|---|
| combined_Room1.csv | Aula (lecture room) | 8,352 | 28 | 29 |
| combined_Room2.csv | Aula (lecture room) | 8,352 | 28 | 29 |
| combined_Room3.csv | Oficina administrativa | 8,352 | 36 | 29 |
| combined_Room4.csv | Oficina de investigadores | 13,536 | 36 | 47 |
| combined_Room5.csv | Biblioteca | 13,536 | 36 | 47 |
| **Total** | | **52,128** | 26 en común | 181 (suma) |

Los 181 días del paper son la **suma** entre salas, no 181 días por sala.

### Esquema no uniforme

Las salas 1 y 2 están servidas por fan coil (`fcu_fan_energy`,
`fcu_fan_speed`); las salas 3, 4 y 5 por unidad manejadora de aire (10 columnas
`ahu_*`, `damper_position`, `cooling_coil_*`, etc.). Hay **26 columnas comunes**
a las cinco salas, y las 5 variables de interés están todas dentro de esa
intersección. Para el modelo global hay que usar la intersección, no la unión.

### Mapeo a las 5 variables del proyecto

| Variable (Aulas 3) | Columna ROBOD | Nota |
|---|---|---|
| CO₂ | `indoor_co2` | ppm |
| Temperatura | `air_temperature` | °C |
| Humedad | `indoor_relative_humidity` | %RH |
| Luz | `illuminance` | lux |
| Movimiento | `occupant_presence` | **proxy**: binaria 0/1, ground truth anotado manualmente, no sensor PIR |

La sustitución de movimiento por `occupant_presence` es una decisión explícita
(2026-10-01) y debe declararse como limitación: ROBOD no instrumentó sensores
de movimiento.

## Limitaciones encontradas en la verificación

1. **Días no consecutivos.** Los datos vienen en bloques de días hábiles con
   huecos grandes (Room1: 8 bloques, con un salto de 2021-10-01 a 2021-12-09).
   No hay fines de semana. Las ventanas temporales deben construirse **dentro**
   de cada bloque contiguo.
2. **CO₂ censurado por abajo.** El mínimo es exactamente 400.00 ppm en las cinco
   salas: es el piso de calibración del sensor NDIR, no un valor real. La cola
   inferior está truncada.
3. **Escalas de iluminancia incomparables entre salas.** Room1 llega a 92 lux;
   Room5 a 10,699 lux (σ de 19 vs 504). Afecta la decisión de normalizar global
   vs por sala.
4. **Faltantes despreciables.** Entre 13 y 30 filas vacías por archivo (< 0.4%),
   siempre en el mismo grupo de columnas de sensor ambiental.
5. **Clima tropical.** Sin estacionalidad anual de temperatura. Los patrones
   térmicos no transfieren a Santa Fe; ROBOD valida el pipeline, no preentrena
   el modelo de producción.

## Hallazgo adicional: la columna `illuminance` viene corrida 8 horas

Detectado el 2026-10-01 al analizar la distribución horaria de la iluminancia.

**Síntoma.** Sin corregir, la iluminancia está anticorrelacionada con la luz
solar: en Room1 promedia 35 lux a las 04:00 y exactamente 0 lux a las 17:00,
cuando la radiación solar exterior aún marca 143 W/m². Es físicamente
imposible para un sensor de luz funcional.

**Evidencia.** Correlación de `illuminance` contra
`global_horizontal_solar_radiation` (medición exterior independiente) según el
corrimiento aplicado:

| Corrimiento | Room1 | Room2 | Room3 | Room4 | Room5 |
|---|---|---|---|---|---|
| 0 h (como viene) | −0.19 | −0.21 | −0.26 | +0.02 | −0.14 |
| **+8 h** | **+0.60** | **+0.60** | **+0.70** | **+0.66** | **+0.57** |

El máximo cae en +8.0 h exactas en Rooms 1, 2 y 4; en +7.5 h en Room 3 y +6.5 h
en Room 5. Ocho horas es exactamente el offset UTC+08:00 de Singapur.

**Descartado que sea la zona horaria del archivo completo.** La radiación solar
es 0 entre las 19 h y las 06 h y pica a las 13 h; la temperatura exterior pica a
las 13 h; y la ocupación de la biblioteca (Room5) es 0% de 00 a 07 h, 90% a las
09 h y cae a 6% a las 21 h. Las tres señales confirman que el archivo está en
hora local. El corrimiento afecta **solo** a la columna `illuminance`.

**Interpretación.** ROBOD integró sensores heterogéneos (lo dice su README). El
canal de iluminancia parece haber quedado en UTC mientras el resto se convirtió
a hora local. Es un error de integración del dataset publicado, no de la
descarga: los hashes coinciden con el origen.

**Corrección que se probó.** Adelantar la serie 8 horas deja la luz correlacionada
positivamente con el sol (r de +0.48 a +0.58 por tipo de espacio) y el día por
encima de la noche entre 4x y 16x. Costo: el arranque de cada uno de los 48
bloques queda sin origen para la iluminancia, lo que reduce las ventanas
entrenables de 51,312 a 46,390 (-9.6%).

**Pendiente.** Vale reportarlo a los autores del dataset, y declararlo en la
tesis como corrección propia sobre el dataset de referencia. Nota: Sabiri et
al. (2025) no usaron iluminancia (se limitaron a CO₂, temperatura y humedad),
así que sus resultados no están afectados por esto.

## Qué de esto se reutiliza con los sensores de Aulas 3

Lo que vale conservar de esta evaluación no son los datos, son los defectos: casi
todos reaparecen con sensores propios, y conviene tener resuelto cómo tratarlos
antes de que empiecen los dos meses de recolección.

| Defecto visto en ROBOD | ¿Reaparece en Aulas 3? |
|---|---|
| CO₂ con piso en 400 ppm | **Sí.** Es el piso de calibración de cualquier NDIR (MH-Z19, SCD30), no un rasgo de ROBOD. La cola inferior queda censurada y el umbral de −3σ no puede dispararse. |
| Luz en 0 exacto buena parte del tiempo | **Sí.** De noche el sensor lee 0, así que el MAPE queda indefinido en esa variable. |
| Escalas de luz incomparables entre espacios | **Sí, y más marcado.** Un aula con ventanal y un pasillo interior difieren en órdenes de magnitud. Obliga a normalizar y a fijar umbrales por espacio o por tipo de espacio, no de forma global. |
| Huecos en la serie temporal | **Sí, y peor.** La telemetría por WiFi va a caerse de forma irregular y por sensor. Las ventanas del modelo no deben cruzar los huecos. |
| Variable de movimiento binaria | **Sí, y peor.** El PIR es 0/1 y ruidoso. ±3σ cae fuera de [0,1], así que el detector estadístico no aplica: necesita criterio de racha. |
| Corrimiento horario entre canales | **Posible.** Si los Arduinos no sincronizan reloj por NTP, o si unos registran en UTC y otros en hora local, se reproduce el mismo error. Conviene fijar una sola fuente de tiempo desde el inicio. |

### Dos cosas a resolver antes de recolectar

1. **Compensación por altitud en el CO₂.** Santa Fe está a ~2,600 m. Los NDIR
   miden absorción infrarroja, que depende del número de moléculas en el camino
   óptico, así que a ~0.74 atm un sensor sin compensación de presión subestima
   el CO₂ de forma sistemática. El SCD30 expone registro de compensación por
   altitud; el MH-Z19 no. Hay que verificar el modelo de los circuitos antes de
   que empiecen los dos meses, porque un sesgo así invalida cualquier umbral de
   calidad del aire.
2. **Una sola fuente de tiempo.** Que todos los nodos sellen la hora con la
   misma referencia (de preferencia UTC en la base, y conversión a hora local
   solo en la presentación). El caso de la iluminancia de ROBOD es exactamente
   lo que pasa cuando dos canales usan referencias distintas.
