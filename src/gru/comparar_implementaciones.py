"""Compare the two recurrent implementations on identical data and seeds.

Trains the same architecture twice, once with CeldaGRU (equations 1 to 4 of
section 4.2.7.2, written literally) and once with nn.GRU (the cuDNN variant),
and prints both results side by side.

The point is to support a claim the thesis needs: the equations published in
the document are the ones that produce the reported results, and the faster
library variant reaches equivalent accuracy. Without this comparison, choosing
the literal implementation would look like an unjustified cost.

Both runs share the seed, the splits and the normalizer, so any difference in
the metrics comes from the formulation and not from the data or the
initialization scale.

Usage, from the src/ directory:
    python -m gru.comparar_implementaciones
    python -m gru.comparar_implementaciones --dias 60

Author:
    Julio César Rodríguez Figueroa (A01029680)

Last modified:
    2026-10-01 - Initial implementation.

Reference:
    Section 4.2.7.2 of the research document.
"""

from __future__ import annotations

import argparse
import json

from comun import config as cfg
from gru import entrenar as ent
from comun import sintetico


def comparar(dias: int, espacios: int, epocas: int, paciencia: int) -> dict:
    """Train both implementations and collect their metrics.

    Args:
        dias: Days of synthetic data to generate.
        espacios: How many spaces to generate.
        epocas: Maximum epochs per run.
        paciencia: Epochs without improvement before early stopping.

    Returns:
        A dict keyed by implementation name, each holding its metrics, the
        epoch that won, the wall-clock seconds and the parameter count.
    """
    df = sintetico.genera(dias=dias, n_espacios=espacios)
    print(f"Datos: {len(df):,} filas, {espacios} espacios, {dias} dias\n")

    resumen = {}
    for implementacion in ("cho", "pytorch"):
        print(f"{'=' * 62}\n  Entrenando con implementacion: {implementacion}\n{'=' * 62}")
        salida = ent.entrenar(
            df.copy(),
            epocas=epocas,
            paciencia=paciencia,
            silencioso=False,
            implementacion=implementacion,
        )
        resumen[implementacion] = {
            "resultados": salida["resultados"],
            "mejor_epoca": salida["mejor_epoca"],
            "mejor_perdida_val": salida["mejor_perdida_val"],
            "segundos": salida["segundos"],
            "parametros": salida["red"].n_parametros(),
            "clase_recurrente": type(salida["red"].gru1).__name__,
        }
        print()

    # Guard against comparing a model with itself. If the implementation flag
    # ever stops reaching the constructor, both runs produce byte-identical
    # results and this script would cheerfully report them as "equivalent",
    # which is the most misleading failure it could have. The two cells differ
    # in parameter count because nn.GRU carries two bias vectors per gate while
    # CeldaGRU carries one, so that is a reliable fingerprint.
    clases = {k: v["clase_recurrente"] for k, v in resumen.items()}
    if clases["cho"] == clases["pytorch"]:
        raise RuntimeError(
            f"both runs used the same recurrent class ({clases['cho']}); "
            "the implementation argument is not reaching the model"
        )
    if resumen["cho"]["parametros"] == resumen["pytorch"]["parametros"]:
        raise RuntimeError(
            "both runs have the same parameter count, so they are the same "
            "model; the comparison would be meaningless"
        )
    return resumen


def imprimir_comparacion(resumen: dict) -> None:
    """Print the side-by-side table and the relative differences.

    Args:
        resumen: Output of comparar().
    """
    cho, torch_ = resumen["cho"], resumen["pytorch"]

    print(f"{'=' * 62}\n  COMPARACION\n{'=' * 62}\n")
    print(f"  {'':22} {'cho (ecs. 1-4)':>18} {'pytorch (cuDNN)':>18}")
    print("  " + "-" * 60)
    print(
        f"  {'parametros':22} {cho['parametros']:18,} {torch_['parametros']:18,}"
    )
    print(
        f"  {'mejor epoca':22} {cho['mejor_epoca']:18} {torch_['mejor_epoca']:18}"
    )
    print(
        f"  {'perdida validacion':22} {cho['mejor_perdida_val']:18.6f} "
        f"{torch_['mejor_perdida_val']:18.6f}"
    )
    print(
        f"  {'segundos':22} {cho['segundos']:18.1f} {torch_['segundos']:18.1f}"
    )
    print(
        f"  {'':22} {'':18} {cho['segundos'] / torch_['segundos']:17.1f}x"
    )

    print(f"\n  MAE por variable (unidades fisicas)")
    print(f"  {'variable':22} {'cho':>18} {'pytorch':>18}   diferencia")
    print("  " + "-" * 74)
    for variable in cfg.VARIABLES_ACTIVAS:
        a = cho["resultados"]["por_variable"][variable]["mae"]
        b = torch_["resultados"]["por_variable"][variable]["mae"]
        # Relative to the pytorch run, which is the reference the library gives.
        rel = 100 * (a - b) / b if b else float("nan")
        print(f"  {variable:22} {a:18.4f} {b:18.4f}   {rel:+7.1f}%")

    a_g = cho["resultados"]["global"]
    b_g = torch_["resultados"]["global"]
    print("  " + "-" * 74)
    print(f"  {'MAE global':22} {a_g['mae']:18.4f} {b_g['mae']:18.4f}")
    print(f"  {'R2 global':22} {a_g['r2']:18.4f} {b_g['r2']:18.4f}")

    # A gap of a few percent is ordinary run-to-run variation in a stochastic
    # optimizer. A large one would mean the formulations are not interchangeable
    # after all, and that the thesis cannot claim equivalence.
    brecha = abs(a_g["r2"] - b_g["r2"])
    print()
    if brecha < 0.05:
        print(
            f"  Las dos implementaciones son equivalentes: la diferencia en R2 "
            f"global es {brecha:.4f}."
        )
        print(
            "  Las ecuaciones del documento se pueden usar sin costo de desempeno."
        )
        if cho["mejor_epoca"] != torch_["mejor_epoca"]:
            print()
            print(
                f"  Cuidado al leer las diferencias de MAE: el early stopping "
                f"corto en\n  epocas distintas ({cho['mejor_epoca']} contra "
                f"{torch_['mejor_epoca']}), asi que una implementacion recibio "
                "mas\n  optimizacion que la otra. Esa brecha explica las "
                "diferencias por variable\n  mejor que la formulacion, y no "
                "sostiene afirmar que una sea superior."
            )
    else:
        print(
            f"  ATENCION: la diferencia en R2 global es {brecha:.4f}, demasiado "
            "grande para\n  llamarlas equivalentes. Vale revisar antes de "
            "afirmarlo en la tesis."
        )


def main() -> int:
    """Run the comparison from the command line and save the result.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Compare the Cho and PyTorch recurrent implementations."
    )
    parser.add_argument("--dias", type=int, default=20)
    parser.add_argument("--espacios", type=int, default=2)
    parser.add_argument("--epocas", type=int, default=30)
    parser.add_argument("--paciencia", type=int, default=8)
    args = parser.parse_args()

    resumen = comparar(args.dias, args.espacios, args.epocas, args.paciencia)
    imprimir_comparacion(resumen)

    cfg.DATOS_PROCESADOS.mkdir(parents=True, exist_ok=True)
    ruta = cfg.DATOS_PROCESADOS / "comparacion_implementaciones.json"
    ruta.write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nGuardado en {ruta.relative_to(cfg.RAIZ)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
