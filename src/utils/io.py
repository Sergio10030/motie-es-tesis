"""Utilidades de entrada/salida: carga y guardado de datos y configuración."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Union

import pandas as pd
import yaml


def ensure_directory(path: Union[str, Path]) -> Path:
    """Crea un directorio y devuelve la ruta normalizada."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Carga un archivo YAML de configuración."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], path: Union[str, Path]) -> None:
    """Guarda un diccionario de configuración en YAML."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)


def load_corpus(path: Union[str, Path], fmt: str = "csv") -> pd.DataFrame:
    """Carga un corpus anotado en CSV o JSON.

    Args:
        path: Ruta al archivo.
        fmt: 'csv' o 'json'.

    Returns:
        DataFrame con el corpus.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No existe el archivo: {path}")

    if fmt == "csv":
        df = pd.read_csv(path, encoding="utf-8")
    elif fmt == "json":
        df = pd.read_json(path, encoding="utf-8")
    else:
        raise ValueError(f"Formato no soportado: {fmt}")

    return df


def save_corpus(df: pd.DataFrame, path: Union[str, Path], fmt: str = "csv") -> None:
    """Guarda un DataFrame en CSV o JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False, encoding="utf-8")
    elif fmt == "json":
        df.to_json(path, orient="records", force_ascii=False, indent=2)
    else:
        raise ValueError(f"Formato no soportado: {fmt}")


def load_json(path: Union[str, Path]) -> Any:
    """Carga un archivo JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: Union[str, Path]) -> None:
    """Guarda un objeto en JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_metrics(metrics: Dict[str, Any], name: str, out_dir: Union[str, Path]) -> None:
    """Guarda métricas en JSON y CSV."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(metrics, out_dir / f"{name}.json")

    flat = {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
    pd.DataFrame([flat]).to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8")


def load_prompt_template(path: Union[str, Path]) -> str:
    """Carga una plantilla de prompt desde un archivo de texto."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()