"""Ejecuta todos los brazos y la validación en secuencia."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def run(script: str, config: str) -> int:
    print(f"\n>>> Ejecutando {script} ...")
    result = subprocess.run(
        [sys.executable, script, "--config", config],
        check=False,
    )
    return result.returncode


def main(config: str, validate_only: bool = False) -> None:
    if validate_only:
        code = run("experiments/run_validation.py", config)
        if code != 0:
            sys.exit(code)
        return

    scripts = [
        "experiments/run_prepare_data.py",
        "experiments/run_arm_a.py",
        "experiments/run_arm_b.py",
        "experiments/run_arm_c.py",
    ]
    for script in scripts:
        code = run(script, config)
        if code != 0:
            print(f"ERROR en {script} (código {code})")
            sys.exit(code)
    print(
        "\n>>> Experimentos completados. Para validar concordancia, complete "
        "data/processed/test_with_preds.csv y ejecute run_all.py --validate-only."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Ejecuta solo la validación después de completar las anotaciones humanas.",
    )
    args = parser.parse_args()
    main(args.config, validate_only=args.validate_only)