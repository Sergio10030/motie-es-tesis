"""C3: Preprocesamiento sensible al registro."""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List

from src.utils.logger import setup_logger

logger = setup_logger("motie_es.cleaner")

# Patrones compilados
_REPEATED_CHARS = re.compile(r"(.)\1{3,}")
_LAUGHTER = re.compile(r"\b(?:ja){2,}\b|\b(?:ha){2,}\b|\b(?:je){2,}\b|\b(?:ji){2,}\b", re.IGNORECASE)
_SPECIAL_CHARS = re.compile(r"[^\w\sáéíóúüñÁÉÍÓÚÜÑ.,;:!?¿¡()\[\]{}\"'\-]", re.UNICODE)
_MULTIPLE_SPACES = re.compile(r"\s+")
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)


class RegisterAwarePreprocessor:
    """Preprocesador configurable que preserva el registro.

    A diferencia de los enfoques que eliminan la jerga, este módulo
    la preserva y la marca con metadatos de registro.
    """

    def __init__(
        self,
        lowercase: bool = True,
        remove_special_chars: bool = True,
        reduce_repeated_chars: bool = True,
        max_repeated: int = 3,
        standardize_laughter: bool = True,
        normalize_emojis: bool = True,
        emoji_format: str = "text",
    ) -> None:
        self.lowercase = lowercase
        self.remove_special_chars = remove_special_chars
        self.reduce_repeated_chars = reduce_repeated_chars
        self.max_repeated = max_repeated
        self.standardize_laughter = standardize_laughter
        self.normalize_emojis = normalize_emojis
        self.emoji_format = emoji_format

    def clean(self, text: str) -> str:
        """Aplica la secuencia de operaciones de limpieza."""
        if text is None:
            return ""

        if not isinstance(text, str):
            text = str(text)

        if text.strip().lower() in {"", "nan", "none", "null"}:
            return ""

        if self.lowercase:
            text = text.lower()

        if self.normalize_emojis:
            text = self._normalize_emojis(text)

        if self.remove_special_chars:
            text = _SPECIAL_CHARS.sub(" ", text)

        if self.reduce_repeated_chars:
            text = self._reduce_repeats(text)

        if self.standardize_laughter:
            text = _LAUGHTER.sub("jajaja", text)

        text = _MULTIPLE_SPACES.sub(" ", text).strip()
        return text

    def _reduce_repeats(self, text: str) -> str:
        """Limita repeticiones excesivas a max_repeated caracteres."""
        def _replace(match: re.Match) -> str:
            char = match.group(1)
            return char * self.max_repeated

        return _REPEATED_CHARS.sub(_replace, text)

    def _normalize_emojis(self, text: str) -> str:
        """Convierte emojis a texto o hexadecimal."""
        def _replace(match: re.Match) -> str:
            emoji = match.group(0)
            if self.emoji_format == "hex":
                return " ".join(f"U+{ord(c):04X}" for c in emoji)
            # Formato texto: nombre aproximado
            return self._emoji_to_text(emoji)

        return _EMOJI_PATTERN.sub(_replace, text)

    @staticmethod
    def _emoji_to_text(emoji: str) -> str:
        """Mapeo simplificado de emojis frecuentes a texto."""
        mapping = {
            "😀": "emoji cara sonriendo",
            "😂": "emoji cara llorando de risa",
            "😊": "emoji cara feliz",
            "😢": "emoji cara triste",
            "😡": "emoji cara enojada",
            "😱": "emoji cara de sorpresa",
            "👍": "emoji pulgar arriba",
            "👎": "emoji pulgar abajo",
            "❤️": "emoji corazón",
            "🔥": "emoji fuego",
            "⚠️": "emoji advertencia",
        }
        return mapping.get(emoji, f"emoji {hex(ord(emoji[0]))}")

    def tokenize(self, text: str) -> List[str]:
        """Tokenización simple por espacios tras limpieza."""
        cleaned = self.clean(text)
        return cleaned.split()

    def process(self, text: str) -> Dict[str, object]:
        """Devuelve texto limpio y tokens."""
        cleaned = self.clean(text)
        return {
            "text_clean": cleaned,
            "tokens": cleaned.split(),
            "n_tokens": len(cleaned.split()),
        }