"""Prepara el corpus: carga, limpieza, partición estratificada."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.cleaner import RegisterAwarePreprocessor
from src.preprocessing.lexicon_loader import LexiconLoader
from src.preprocessing.register_metadata import RegisterMetadataCalculator
from src.utils.io import load_config, load_corpus
from src.utils.logger import setup_logger
from src.utils.seed import set_seed


def _detect_delimiter(path: Path) -> str:
    """Detecta delimitador de CSV para datasets reales, como el de solicitudes."""
    with path.open("r", encoding="utf-8") as handle:
        sample = handle.read(2048)
    return ";" if sample.count(";") > sample.count(",") else ","


def _normalize_real_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Ajusta el dataset real a la estructura esperada por el proyecto."""
    out = df.copy()

    text_candidates = ["texto", "Referencia", "descripcion", "detalle", "motivo", "mensaje", "texto_reporte"]
    text_col = next((c for c in text_candidates if c in out.columns), None)
    if text_col is None:
        raise ValueError("No se encontró una columna de texto en el dataset real.")
    out["texto"] = out[text_col].fillna("").astype(str)

    if "tipo_funcional" not in out.columns:
        if "Tipo" in out.columns:
            mapping = {
                "Queja": "queja_general",
                "Sugerencia": "sugerencia_general",
                "Incidencia": "incidencia_general",
            }
            out["tipo_funcional"] = out["Tipo"].map(mapping).fillna("incidencia_general")
        else:
            out["tipo_funcional"] = "incidencia_general"

    if "prioridad" not in out.columns:
        if "Estado" in out.columns:
            mapping = {
                "Contestado": "media",
                "Remitido Órgano": "media",
                "Anulada": "baja",
                "Inadmitida": "baja",
            }
            out["prioridad"] = out["Estado"].map(mapping).fillna("media")
        else:
            out["prioridad"] = "media"

    return out


def main(config_path: str) -> None:
    config = load_config(config_path)
    logger = setup_logger(
        "motie_es.prepare_data",
        level=config["logging"]["level"],
        log_file=config["logging"]["file"],
    )
    set_seed(config["project"]["seed"])

    logger.info("=== Preparando datos ===")

    raw_dir = Path(config["paths"]["data_raw"])
    candidates = list(raw_dir.glob("*.csv")) + list(raw_dir.glob("*.json"))
    if not candidates:
        logger.error("No hay archivos en %s", raw_dir)
        return

    path = candidates[0]
    if path.suffix.lower() == ".csv":
        sep = _detect_delimiter(path)
        df = pd.read_csv(path, sep=sep, encoding="utf-8")
    else:
        df = load_corpus(path, config["corpus"]["input_format"])

    if "texto" not in df.columns or "tipo_funcional" not in df.columns or "prioridad" not in df.columns:
        df = _normalize_real_dataset(df)

    logger.info("Corpus cargado: %d reportes", len(df))

    preprocessor = RegisterAwarePreprocessor(
        lowercase=config["preprocessing"]["lowercase"],
        remove_special_chars=config["preprocessing"]["remove_special_chars"],
        reduce_repeated_chars=config["preprocessing"]["reduce_repeated_chars"],
        max_repeated=config["preprocessing"]["max_repeated"],
        standardize_laughter=config["preprocessing"]["standardize_laughter"],
        normalize_emojis=config["preprocessing"]["normalize_emojis"],
        emoji_format=config["preprocessing"]["emoji_format"],
    )

    df["texto"] = df["texto"].fillna("").astype(str)
    df["text_clean"] = df["texto"].apply(preprocessor.clean)
    df["tokens"] = df["text_clean"].apply(lambda x: x.split())
    df["n_tokens"] = df["tokens"].apply(len)

    lexicon = LexiconLoader(
        config["lexicon"]["jerga_file"],
        config["lexicon"]["admin_file"],
        normalize=config["lexicon"]["normalize"],
        remove_duplicates=config["lexicon"]["remove_duplicates"],
    )
    calculator = RegisterMetadataCalculator(lexicon, preprocessor)
    df = calculator.compute_dataframe(df, "text_clean")

    out_dir = Path(config["paths"]["data_processed"])
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "corpus_processed.csv", index=False, encoding="utf-8")

    strat_cols = config["corpus"]["stratify_by"]
    test_size = config["corpus"]["test_size"]
    val_size = config["corpus"]["val_size"]

    strat_key = df[strat_cols].astype(str).agg("_".join, axis=1)
    use_stratify = True
    try:
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, stratify=strat_key, random_state=config["project"]["seed"]
        )
    except ValueError:
        logger.warning(
            "Stratified split no es viable para este dataset; usando partición aleatoria con mezcla."
        )
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=config["project"]["seed"]
        )
        use_stratify = False

    if use_stratify:
        strat_key_tv = train_val_df[strat_cols].astype(str).agg("_".join, axis=1)
        val_ratio = val_size / (1.0 - test_size)
        try:
            train_df, val_df = train_test_split(
                train_val_df, test_size=val_ratio, stratify=strat_key_tv,
                random_state=config["project"]["seed"]
            )
        except ValueError:
            logger.warning(
                "La partición de validación estratificada tampoco es viable; usando partición aleatoria."
            )
            train_df, val_df = train_test_split(
                train_val_df, test_size=val_ratio, random_state=config["project"]["seed"]
            )
    else:
        val_ratio = val_size / (1.0 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_ratio, random_state=config["project"]["seed"]
        )

    train_df.to_csv(out_dir / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(out_dir / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(out_dir / "test.csv", index=False, encoding="utf-8")

    logger.info(
        "Partición: train=%d, val=%d, test=%d",
        len(train_df), len(val_df), len(test_df),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    main(args.config)