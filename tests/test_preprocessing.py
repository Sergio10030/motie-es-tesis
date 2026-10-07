"""Pruebas unitarias para preprocesamiento y léxico."""
import pytest

from src.preprocessing.cleaner import RegisterAwarePreprocessor
from src.preprocessing.lexicon_loader import LexiconLoader


@pytest.fixture
def preprocessor():
    return RegisterAwarePreprocessor(
        lowercase=True,
        remove_special_chars=True,
        reduce_repeated_chars=True,
        max_repeated=3,
        standardize_laughter=True,
        normalize_emojis=True,
        emoji_format="text",
    )


def test_lowercase(preprocessor):
    assert preprocessor.clean("HOLA MUNDO") == "hola mundo"


def test_reduce_repeats(preprocessor):
    out = preprocessor.clean("paltaaaaaaaaaa")
    assert "aaaaaaaa" not in out
    assert "aaa" in out


def test_standardize_laughter(preprocessor):
    out = preprocessor.clean("jajajajajaja")
    assert out == "jajaja"


def test_remove_special_chars(preprocessor):
    out = preprocessor.clean("hola @#$ mundo")
    assert "@" not in out and "#" not in out and "$" not in out


def test_empty_string(preprocessor):
    assert preprocessor.clean("") == ""


def test_empty_like_values(preprocessor):
    assert preprocessor.clean(None) == ""
    assert preprocessor.clean("nan") == ""
    assert preprocessor.clean(float("nan")) == ""


def test_tokenize(preprocessor):
    tokens = preprocessor.tokenize("Hola   mundo")
    assert tokens == ["hola", "mundo"]


def test_lexicon_density(tmp_path):
    jerga = tmp_path / "jerga.json"
    admin = tmp_path / "admin.json"
    jerga.write_text('["palta", "chévere"]', encoding="utf-8")
    admin.write_text('["suspensión", "acta"]', encoding="utf-8")

    lex = LexiconLoader(str(jerga), str(admin))
    tokens = ["hola", "palta", "mundo"]
    density = lex.get_density(tokens, "jerga")
    assert density == pytest.approx(1/3)


def test_lexicon_missing_file(tmp_path):
    lex = LexiconLoader(str(tmp_path / "no.json"), str(tmp_path / "no2.json"))
    assert len(lex.jerga) == 0
    assert len(lex.admin) == 0