"""Fijación de semillas aleatorias para reproducibilidad."""
from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Fija la semilla en todas las librerías relevantes.

    Args:
        seed: Semilla entera.
        deterministic: Si True, activa modo determinista en PyTorch.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:
                pass
    except ImportError:
        pass

    try:
        import transformers

        transformers.set_seed(seed)
    except ImportError:
        pass


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Devuelve un generador de números aleatorios con semilla."""
    return np.random.default_rng(seed)