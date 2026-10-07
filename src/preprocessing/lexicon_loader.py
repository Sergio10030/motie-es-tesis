"""C2: Carga y normalización del léxico escolar externo."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set, Union

from src.utils.io import load_json
from src.utils.logger import setup_logger

logger = setup_logger("motie_es.lexicon")


class LexiconLoader:
    """Carga y normaliza el léxico escolar en dos categorías.

    Attributes:
        jerga: Conjunto de términos de jerga escolar.
        admin: Conjunto de términos administrativo-disciplinarios.
    """

    def __init__(
        self,
        jerga_path: Union[str, Path],
        admin_path: Union[str, Path],
        normalize: bool = True,
        remove_duplicates: bool = True,
    ) -> None:
        self.jerga_path = Path(jerga_path)
        self.admin_path = Path(admin_path)
        self.normalize = normalize
        self.remove_duplicates = remove_duplicates
        self.jerga: Set[str] = set()
        self.admin: Set[str] = set()
        self._load()

    def _load(self) -> None:
        """Carga y normaliza los léxicos."""
        self.jerga = self._read_lexicon(self.jerga_path)
        self.admin = self._read_lexicon(self.admin_path)

        if self.remove_duplicates:
            self.admin -= self.jerga

        logger.info(
            "Léxico cargado: %d términos de jerga, %d términos administrativos",
            len(self.jerga),
            len(self.admin),
        )

    def _read_lexicon(self, path: Path) -> Set[str]:
        """Lee un archivo léxico y devuelve un conjunto de términos."""
        if not path.exists():
            logger.warning("No existe el archivo léxico: %s", path)
            return set()

        data = load_json(path)
        terms: List[str] = []

        if isinstance(data, list):
            terms = [str(t) for t in data]
        elif isinstance(data, dict):
            for key in ("terminos", "terms", "palabras", "words"):
                if key in data and isinstance(data[key], list):
                    terms = [str(t) for t in data[key]]
                    break
            else:
                # Aplanar valores de listas
                for v in data.values():
                    if isinstance(v, list):
                        terms.extend(str(t) for t in v)

        if self.normalize:
            terms = [t.strip().lower() for t in terms if t.strip()]

        return set(terms)

    def get_density(self, tokens: List[str], category: str) -> float:
        """Calcula la densidad de registro de una categoría.

        d_r = |T_r ∩ L_r| / |T_r|

        Args:
            tokens: Lista de tokens del reporte.
            category: 'jerga' o 'admin'.

        Returns:
            Densidad en [0, 1].
        """
        if not tokens:
            return 0.0
        lexicon = self.jerga if category == "jerga" else self.admin
        if not lexicon:
            return 0.0
        tokens_norm = [t.lower() for t in tokens]
        intersection = sum(1 for t in tokens_norm if t in lexicon)
        return intersection / len(tokens_norm)

    def get_all_terms(self) -> Dict[str, List[str]]:
        """Devuelve el léxico normalizado como diccionario."""
        return {
            "jerga": sorted(self.jerga),
            "administrativo": sorted(self.admin),
        }