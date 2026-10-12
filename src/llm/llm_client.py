"""HTTP client for Llama 3.1 8B served by Ollama.

A thin wrapper over one endpoint. It sends a system prompt and a user prompt
and returns the generated text, turning every transport failure into a
RuntimeError whose message says what to do about it.

Why it stays thin: nothing here decides what to ask, and nothing here reads
the answer. The prompts arrive already built by llm.context, and the caller
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
CHAT_URL = f"{HOST}/api/chat"
MODELS_URL = f"{HOST}/api/tags"

MODEL = "llama3.1:8b"

# Generous because the first call after boot includes loading ~6 GB of weights
# into VRAM; once warm the same call answers in a few seconds.
TIMEOUT_S = 120.0
PROBE_TIMEOUT_S = 3.0

# Low temperature on purpose. This model reports sensor readings, so the same
# data should produce the same wording twice, and there is nothing to gain
# from sampling creatively.
TEMPERATURE = 0.2

# The system prompt already asks for at most three sentences; this is the hard
# stop in case it does not comply.
MAX_TOKENS = 300


def generate_answer(
    system_prompt: str,
    user_prompt: str,
    model: str = MODEL,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS,
    timeout: float = TIMEOUT_S,
) -> str:
    """Send both prompts to Ollama and return the generated text.

    Args:
        system_prompt: Role and constraints, from llm.context.SYSTEM_PROMPT.
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
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "options": {"temperature": temperature, "num_predict": max_tokens},
    }

    try:
        response = httpx.post(CHAT_URL, json=payload, timeout=timeout)
        response.raise_for_status()
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
                f"Ollama no tiene el modelo '{model}'. Descárgalo con "
                f"`ollama pull {model}`"
            ) from e
        raise RuntimeError(
            f"Ollama devolvió HTTP {e.response.status_code}: "
            f"{e.response.text[:200]}"
        ) from e

    data = response.json()
    try:
        return data["message"]["content"].strip()
    except (KeyError, TypeError) as e:
        raise RuntimeError(f"Respuesta inesperada de Ollama: {data}") from e


def available(model: str = MODEL) -> tuple[bool, str]:
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
        response = httpx.get(MODELS_URL, timeout=PROBE_TIMEOUT_S)
        response.raise_for_status()
    except httpx.HTTPError:
        return False, (
            f"Ollama no responde en {HOST}; instálalo o arráncalo con "
            "`ollama serve`"
        )

    try:
        installed = [m["name"] for m in response.json().get("models", [])]
    except (AttributeError, KeyError, TypeError):
        return False, "Ollama respondió algo que no se pudo interpretar"

    if model in installed:
        return True, f"Ollama listo con {model}"

    # Ollama reports tags as "llama3.1:8b"; accept a bare family match so a
    # differently tagged build of the same model still counts as present.
    family = model.split(":")[0]
    similar = [m for m in installed if m.split(":")[0] == family]
    if similar:
        return True, f"Ollama listo con {similar[0]} (se pidió {model})"

    return False, (
        f"Ollama corre pero no tiene '{model}'. Descárgalo con "
        f"`ollama pull {model}`. Instalados: {installed or 'ninguno'}"
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

    ready, reason = available()
    print(f"Sondeo: {reason}")
    if not ready:
        return 1

    print(f"\nPreguntando a {MODEL}...")
    try:
        response = generate_answer(
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

    print(f"\nOK: Ollama respondió\n\n{response}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
