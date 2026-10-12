"""HTTP client for Llama 3.1 8B served by Ollama.

A thin wrapper over one endpoint. It sends a system prompt and a user prompt
and returns the generated text, turning every transport failure into a
RuntimeError whose message says what to do about it.

Why it stays thin: nothing here decides what to ask, and nothing here reads
the answer. The prompts arrive already built by llm.contexto, and the caller
hands the text straight to the user. That keeps the one place where the
project depends on a language model small enough to swap.

The model runs locally. No reading from Aulas 3 leaves the machine, which is
the reason the research document chose a local model over a hosted API.

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-12 - Initial implementation.

Reference:
    Section 4.3.3 of the research document.
    Ollama API: https://github.com/ollama/ollama/blob/main/docs/api.md
"""

from __future__ import annotations

import httpx

# Ollama listens here once installed; it runs as a background service on
# Windows and does not need to be started by hand.
HOST = "http://localhost:11434"
RUTA_CHAT = f"{HOST}/api/chat"
RUTA_MODELOS = f"{HOST}/api/tags"

MODELO = "llama3.1:8b"

# Generous because the first call after boot includes loading ~6 GB of weights
# into VRAM; once warm the same call answers in a few seconds.
TIMEOUT_SEG = 120.0
TIMEOUT_SONDEO_SEG = 3.0

# Low temperature on purpose. This model reports sensor readings, so the same
# data should produce the same wording twice, and there is nothing to gain
# from sampling creatively.
TEMPERATURA = 0.2

# The system prompt already asks for at most three sentences; this is the hard
# stop in case it does not comply.
MAX_TOKENS = 300


def generar_respuesta(
    system_prompt: str,
    user_prompt: str,
    modelo: str = MODELO,
    temperatura: float = TEMPERATURA,
    max_tokens: int = MAX_TOKENS,
    timeout: float = TIMEOUT_SEG,
) -> str:
    """Send both prompts to Ollama and return the generated text.

    Args:
        system_prompt: Role and constraints, from llm.contexto.SISTEMA.
        user_prompt: The question plus the retrieved data.
        modelo: Ollama model tag.
        temperatura: Sampling temperature.
        max_tokens: Upper bound on generated tokens.
        timeout: Seconds to wait for the whole response.

    Returns:
        The generated text, stripped.

    Raises:
        RuntimeError: If Ollama is unreachable, the model is not installed,
            the request times out, or the response has an unexpected shape.
            Every message names the command that fixes it.
    """
    carga = {
        "model": modelo,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": temperatura, "num_predict": max_tokens},
    }

    try:
        respuesta = httpx.post(RUTA_CHAT, json=carga, timeout=timeout)
        respuesta.raise_for_status()
    except httpx.ConnectError as e:
        raise RuntimeError(
            f"No se puede conectar con Ollama en {HOST}. Verifica que esté "
            "corriendo con `ollama serve`, o instálalo desde "
            "https://ollama.com/download/windows"
        ) from e
    except httpx.TimeoutException as e:
        raise RuntimeError(
            f"Ollama no respondió en {timeout:.0f} s. La primera llamada "
            "después de encender la máquina carga el modelo en VRAM y puede "
            "tardar; vuelve a intentar."
        ) from e
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise RuntimeError(
                f"Ollama no tiene el modelo '{modelo}'. Descárgalo con "
                f"`ollama pull {modelo}`"
            ) from e
        raise RuntimeError(
            f"Ollama devolvió HTTP {e.response.status_code}: "
            f"{e.response.text[:200]}"
        ) from e

    datos = respuesta.json()
    try:
        return datos["message"]["content"].strip()
    except (KeyError, TypeError) as e:
        raise RuntimeError(f"Respuesta inesperada de Ollama: {datos}") from e


def disponible(modelo: str = MODELO) -> tuple[bool, str]:
    """Check whether Ollama is reachable and the model is installed.

    Used by the API health check, so it has to answer fast and never raise:
    a health endpoint that hangs or crashes is worse than one that reports a
    problem.

    Args:
        modelo: Model tag to look for.

    Returns:
        A tuple of (ready, human-readable reason). When ready is True the
        reason names the model that answered.
    """
    try:
        respuesta = httpx.get(RUTA_MODELOS, timeout=TIMEOUT_SONDEO_SEG)
        respuesta.raise_for_status()
    except httpx.HTTPError:
        return False, (
            f"Ollama no responde en {HOST}; instálalo o arráncalo con "
            "`ollama serve`"
        )

    try:
        instalados = [m["name"] for m in respuesta.json().get("models", [])]
    except (AttributeError, KeyError, TypeError):
        return False, "Ollama respondió algo que no se pudo interpretar"

    if modelo in instalados:
        return True, f"Ollama listo con {modelo}"

    # Ollama reports tags as "llama3.1:8b"; accept a bare family match so a
    # differently tagged build of the same model still counts as present.
    familia = modelo.split(":")[0]
    parecidos = [m for m in instalados if m.split(":")[0] == familia]
    if parecidos:
        return True, f"Ollama listo con {parecidos[0]} (se pidió {modelo})"

    return False, (
        f"Ollama corre pero no tiene '{modelo}'. Descárgalo con "
        f"`ollama pull {modelo}`. Instalados: {instalados or 'ninguno'}"
    )


def main() -> int:
    """Probe Ollama and ask it one question with known data.

    Returns:
        Process exit code: 0 when the model answered, 1 otherwise.
    """
    import sys

    # The answer comes back in Spanish and may contain characters outside the
    # console's cp1252 default, which would make print raise instead of
    # showing the result.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    listo, motivo = disponible()
    print(f"Sondeo: {motivo}")
    if not listo:
        return 1

    print(f"\nPreguntando a {MODELO}...")
    try:
        respuesta = generar_respuesta(
            system_prompt=(
                "Eres el asistente del Gemelo Digital del edificio Aulas 3. "
                "Responde en español en máximo 2 oraciones. Usa solo los "
                "datos que recibes."
            ),
            user_prompt=(
                "LECTURAS ACTUALES del salón 3302:\n"
                "  co2: 1180 ppm\n"
                "  temperatura: 27.4 °C\n"
                "REFERENCIA: CO2 por encima de 1000 ppm se considera alto.\n\n"
                "¿Cómo está el aire del salón?"
            ),
        )
    except RuntimeError as e:
        print(f"ERROR: {e}")
        return 1

    print(f"\nOK: Ollama respondió\n\n{respuesta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
