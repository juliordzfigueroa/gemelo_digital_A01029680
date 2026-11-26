"""The words the classifier looks for, kept apart from its rules.

Adding a synonym or a phrase is an edit to this file only; llm/classifier.py
decides what to do when one of them shows up. The words stay in Spanish
because they match what the user types.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-11-26 - Moved here from classifier.py.

Reference:
    Section 4.3.4 of the research document.
"""

# --------------------------------------------------------------------------
# What kind of answer the question asks for
# --------------------------------------------------------------------------
# No list for the present state: it is what any question without one of these
# words gets.
ANOMALY_WORDS = [
    "anomalía", "anomalia", "anomalías", "anomalias",
    "raro", "rara", "extraño", "extraña",
    "fuera de lo normal", "anormal", "inusual",
    "problema", "alerta", "alarma", "peligro", "peligroso",
    "excede", "supera", "demasiado", "muy alto", "muy alta",
    "algo mal", "algo está mal", "algo esta mal",
]

FORECAST_WORDS = [
    "estará", "estara", "estarán", "estaran",
    "va a estar", "van a estar",
    "predice", "predecir", "predicción", "prediccion", "predecirá",
    "pronostica", "pronóstico", "pronostico",
    "próximos", "proximos", "próximas", "proximas",
    "más tarde", "mas tarde", "al rato",
    "futuro", "tendencia", "subirá", "subira", "bajará", "bajara",
]

HISTORY_WORDS = [
    "ayer", "anteayer", "la semana pasada", "el mes pasado",
    "histórico", "historico", "histórica", "historica", "historial",
    "cuándo fue", "cuando fue", "cuándo estuvo", "cuando estuvo",
    "pico", "máximo", "maximo", "mínimo", "minimo",
    "promedio", "últimos días", "ultimos dias",
    "últimas horas", "ultimas horas", "últimas 24", "ultimas 24",
    "se registró", "se registro", "hubo",
]

# A time ahead said in words. "En 45 minutos" is caught by a pattern in the
# classifier instead, because the number varies.
FUTURE_PHRASES = [
    "en media hora", "en una hora", "en dos horas",
    "dentro de una hora", "dentro de media hora",
]

# How far ahead, when the question says it in words. Checked in this order, so
# "hora y media" is found before "una hora" would match part of it.
HORIZON_PHRASES = [
    ("dos horas", 120), ("2 horas", 120),
    ("hora y media", 90), ("1.5 horas", 90),
    ("una hora", 60), ("1 hora", 60),
    ("media hora", 30),
]

# --------------------------------------------------------------------------
# When: a date and a time of day
# --------------------------------------------------------------------------
MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6,
    "julio": 7, "agosto": 8, "septiembre": 9, "setiembre": 9, "octubre": 10,
    "noviembre": 11, "diciembre": 12,
    "ene": 1, "feb": 2, "abr": 4, "jun": 6, "jul": 7, "ago": 8, "sep": 9,
    "sept": 9, "oct": 10, "nov": 11, "dic": 12,
}

# Words after the hour that move it to the afternoon: "a las 5 de la tarde".
AFTERNOON_WORDS = ["pm", "p.m.", "de la tarde", "de la noche"]

# --------------------------------------------------------------------------
# Which variables
# --------------------------------------------------------------------------
VARIABLE_SYNONYMS = {
    "co2": ["co2", "co₂", "dióxido de carbono", "dioxido de carbono", "aire"],
    "temperature": ["temperatura", "calor", "frío", "frio", "fría", "fria", "grados"],
    "humidity": ["humedad", "húmedo", "humedo", "húmeda", "humeda"],
    "light": ["luz", "iluminación", "iluminacion", "iluminancia", "lux", "luminosidad"],
    "motion": ["movimiento", "ocupación", "ocupacion", "personas", "gente", "ocupado"],
}

# --------------------------------------------------------------------------
# Where: a floor or the whole building
# --------------------------------------------------------------------------
# Floor 0 is the ground floor ("planta baja"); "primer piso" is floor 1, as
# the room numbers say (3305 is on floor 3).
FLOOR_NOUN = r"(?:piso|nivel|planta)"
FLOOR_NUMBER_WORDS = {"cero": 0, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4}
FLOOR_ORDINALS = {
    "primer": 1, "primero": 1, "segundo": 2,
    "tercer": 3, "tercero": 3, "cuarto": 4,
}

# Phrases that ask about every space at once. Matched as whole words: "todo"
# is inside "método", and a substring match would send "¿qué método usas?"
# to the whole building.
BUILDING_WORDS = [
    "todos los salones", "todas las aulas", "todos los espacios",
    "todo el edificio", "el edificio", "edificio completo",
    "cada salón", "cada salon", "en general", "resumen general",
    "todo", "toda", "todos", "todas",
]

# --------------------------------------------------------------------------
# Messages that are not questions about the building
# --------------------------------------------------------------------------
# All matched as whole words, and only when the question names no space,
# floor or the building: "hola, ¿cómo está el 3308?" is a question about 3308.

# Words that tie a question to the building even without naming a room.
# Generic phrasing ("cuál es", "hoy") is not here on purpose: "¿cuál es la
# capital de Francia?" has it too.
DOMAIN_WORDS = [
    "salón", "salon", "salones", "aula", "aulas", "piso", "pisos",
    "edificio", "oficina", "oficinas", "sensor", "sensores",
    "lectura", "lecturas", "valores", "datos", "medición", "medicion",
    "mediciones", "estado", "ambiente", "ventilación", "ventilacion", "ventilar",
]

# About the assistant itself: greetings and "what is this".
HELP_WORDS = [
    "hola", "buenos días", "buenos dias", "buenas tardes", "buenas noches",
    "buenas", "qué eres", "que eres", "quién eres", "quien eres",
    "qué puedes", "que puedes", "qué sabes", "que sabes", "qué haces",
    "que haces", "para qué sirves", "para que sirves", "cómo funciona",
    "como funciona", "cómo te uso", "como te uso", "ayuda", "es esto",
    "instrucciones",
]

THANKS_WORDS = ["gracias", "te agradezco", "muy amable"]

# Requests a general-purpose chatbot would take and this one cannot.
OUT_OF_SCOPE_WORDS = [
    "escribe", "escríbeme", "escribeme", "redacta", "redáctame", "traduce",
    "explícame", "explicame", "explica", "poema", "chiste", "cuento",
    "canción", "cancion", "receta", "capital de", "quién fue", "quien fue",
    "quién es", "quien es", "programa", "código", "codigo", "tarea", "ensayo",
    "opinas", "opinión", "opinion", "chatgpt", "noticias", "película",
    "pelicula", "fútbol", "futbol", "hace en",
]
