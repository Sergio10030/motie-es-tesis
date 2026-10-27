"""Predictor: Integra los 3 brazos experimentales para la interfaz web."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
import json

from src.utils.logger import setup_logger

logger = setup_logger("motie_es.predictor")

# Tipos y prioridades posibles
TIPOS = ["acoso", "conflicto_estudiantes", "daño_infraestructura", "incidente_salud", "falta_disciplinaria"]
PRIORIDADES = ["alta", "media", "baja"]


class BrazoAPredictor:
    """Brazo A: clasificación por vocabulario. Sin dependencias externas.

    Los términos se comparan con límite de palabra al inicio, de modo que un
    término actúa como raíz: "amenaz" cubre amenazó/amenaza/amenazar. Un
    término que termina en "$" exige palabra completa, para evitar que
    "arma" se dispare dentro de "alarma" o "armario".
    """

    RULES = {
        "acoso": ["acoso", "bullying", "violen", "golpe", "sangre", "ataque", "atac",
                  "amenaz", "agred", "hostig", "burla", "discrimin", "tocamiento",
                  "intimid", "humill"],
        "conflicto_estudiantes": ["pelea", "conflicto", "insult", "discusion", "discusión",
                                  "discuti", "agresion", "agresión", "riña", "empuj", "grit"],
        "daño_infraestructura": ["daño", "dano", "roto$", "rompi", "destru", "vidrio",
                                 "puerta", "ventana", "columna", "viga", "pared", "techo",
                                 "pupitre", "mobiliario", "infraestructura", "deterior",
                                 "mal cuidad", "grieta", "fisura", "vandalismo", "pintada",
                                 "baño", "instalacion", "instalación", "carpeta"],
        "incidente_salud": ["enferm", "salud", "fiebre", "desmay", "accidente", "herid",
                            "asma", "vomit", "convulsi", "intoxica", "fractura", "sangrado",
                            "alergia", "respirar", "lesion", "lesión"],
        "falta_disciplinaria": ["retraso", "tarde$", "falta$", "faltas$", "desobediencia",
                                "incumpli", "celular", "copia", "tarea", "uniforme",
                                "insubordina"],
    }

    # Prioridad por defecto de cada tipo. Refleja el coste asimétrico del
    # proyecto: salud y acoso escalan solos, el resto parte de media/baja.
    TYPE_PRIORITY = {
        "acoso": "alta",
        "incidente_salud": "alta",
        "conflicto_estudiantes": "media",
        "daño_infraestructura": "media",
        "falta_disciplinaria": "baja",
    }

    # Señales de gravedad: fuerzan prioridad alta incluso si el tipo no se
    # reconoce. Un falso negativo en clase crítica es el error más caro del
    # sistema (cost_alpha en config.yaml), así que se escala por defecto.
    ESCALATE = ["sangre", "arma$", "armas$", "grave", "urgente", "inmediato", "hospital",
                "desmay", "muerte", "fractura", "convulsi", "incendio", "derrumb",
                "colaps", "electrocut", "ambulancia", "asfixia", "suicid"]

    def __init__(self):
        self._rules_re = {
            tipo: [self._compile(kw) for kw in kws] for tipo, kws in self.RULES.items()
        }
        self._escalate_re = [self._compile(kw) for kw in self.ESCALATE]

    @staticmethod
    def _compile(keyword: str):
        """Raíz de palabra, o palabra completa si el término termina en '$'."""
        if keyword.endswith("$"):
            return re.compile(rf"\b{re.escape(keyword[:-1])}\b", re.IGNORECASE)
        return re.compile(rf"\b{re.escape(keyword)}", re.IGNORECASE)

    def predict(self, text: str) -> Tuple[str, str, float]:
        """Clasifica por coincidencia de vocabulario. Siempre devuelve algo."""
        scores = {
            tipo: sum(1 for rx in rxs if rx.search(text))
            for tipo, rxs in self._rules_re.items()
        }
        mejor = max(scores, key=lambda k: scores[k])
        n = scores[mejor]
        grave = any(rx.search(text) for rx in self._escalate_re)

        if n == 0:
            # Sin tipo reconocible. Si hay señal de gravedad se escala igual y
            # la confianza baja lo deriva a revisión humana.
            return "falta_disciplinaria", ("alta" if grave else "media"), 0.40

        prioridad = "alta" if grave else self.TYPE_PRIORITY[mejor]

        # Más coincidencias => más confianza, con techo deliberadamente bajo:
        # son reglas, no un modelo entrenado.
        return mejor, prioridad, round(min(0.55 + 0.10 * n, 0.80), 3)


class BrazoBPredictor:
    """Brazo B: Fine-tuning con transformers (usa checkpoints guardados)."""

    def __init__(self, checkpoint_dir: str = "models/checkpoints"):
        """Inicializa desde checkpoints guardados."""
        self.checkpoint_dir = Path(checkpoint_dir)
        self.tipo_model = None
        self.prioridad_model = None
        self.tokenizer = None
        self.loaded = False
        self.reason = None
        self.tipo_labels: Dict[int, str] = {}
        self.prioridad_labels: Dict[int, str] = {}
        self._load_models()

    def _load_models(self):
        """Carga los checkpoints solo si sus etiquetas son la taxonomía escolar.

        Un checkpoint entrenado sobre otro esquema (queja/sugerencia general, o
        prioridad sin 'alta') daría etiquetas incorrectas y nunca escalaría un
        caso crítico, así que se descarta en lugar de reinterpretarlo.
        """
        tipo_path = self.checkpoint_dir / "tipo" / "pysentimiento_robertuito-base-uncased_best"
        prioridad_path = self.checkpoint_dir / "prioridad" / "pysentimiento_robertuito-base-uncased_best"

        if not (tipo_path.exists() and prioridad_path.exists()):
            self.reason = f"no hay checkpoints en {self.checkpoint_dir}"
            logger.warning(f"Brazo B inactivo: {self.reason}")
            return

        self.tipo_labels = self._read_labels(tipo_path)
        self.prioridad_labels = self._read_labels(prioridad_path)

        faltan = (set(TIPOS) - set(self.tipo_labels.values())) | (
            set(PRIORIDADES) - set(self.prioridad_labels.values())
        )
        if faltan:
            self.reason = (
                "checkpoint entrenado con otro esquema de etiquetas "
                f"(tipo={sorted(self.tipo_labels.values())}, "
                f"prioridad={sorted(self.prioridad_labels.values())}); "
                "reentrena con la taxonomía escolar"
            )
            logger.warning(f"Brazo B inactivo: {self.reason}")
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            logger.info("Cargando modelos Fine-tuning del Brazo B...")
            self.tokenizer = AutoTokenizer.from_pretrained(str(tipo_path))
            self.tipo_model = AutoModelForSequenceClassification.from_pretrained(str(tipo_path))
            self.prioridad_model = AutoModelForSequenceClassification.from_pretrained(str(prioridad_path))
            self.loaded = True
            logger.info("Modelos del Brazo B cargados exitosamente")
        except Exception as e:
            self.reason = str(e)
            logger.warning(f"No se pudieron cargar modelos del Brazo B: {e}")

    @staticmethod
    def _read_labels(path: Path) -> Dict[int, str]:
        """Lee id2label del config.json del checkpoint."""
        try:
            cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
            return {int(k): v for k, v in (cfg.get("id2label") or {}).items()}
        except Exception:
            return {}

    def predict(self, text: str) -> Tuple[str, str, float]:
        """Predice usando modelos fine-tuned."""
        if not self.loaded:
            return "falta_disciplinaria", "baja", 0.0

        try:
            # Tokenizar
            inputs = self.tokenizer(text[:512], truncation=True, return_tensors="pt")

            # Predicciones
            import torch
            with torch.no_grad():
                tipo_logits = self.tipo_model(**inputs).logits
                prioridad_logits = self.prioridad_model(**inputs).logits

            tipo_pred = tipo_logits.argmax(dim=1).item()
            prioridad_pred = prioridad_logits.argmax(dim=1).item()

            # Confianza promedio
            tipo_conf = torch.softmax(tipo_logits, dim=1).max().item()
            prioridad_conf = torch.softmax(prioridad_logits, dim=1).max().item()
            confianza = (tipo_conf + prioridad_conf) / 2

            tipo = self.tipo_labels[tipo_pred]
            prioridad = self.prioridad_labels[prioridad_pred]

            return tipo, prioridad, confianza
        except Exception as e:
            logger.warning(f"Error en predicción Brazo B: {e}")
            return "falta_disciplinaria", "baja", 0.0


class BrazoCPredictor:
    """Brazo C: LLM Few-shot (GPT-4, Claude, etc con manejo de errores)."""

    # Paquete pip y variable de entorno de cada proveedor soportado.
    PROVIDERS = {
        "openai": ("openai", "OPENAI_API_KEY"),
        "anthropic": ("anthropic", "ANTHROPIC_API_KEY"),
        "google": ("google-generativeai", "GOOGLE_API_KEY"),
    }

    def __init__(self, provider: str = "openai", model: str = "gpt-4o"):
        """Inicializa cliente LLM."""
        self.provider = provider.lower()
        self.model = model
        self.client = None
        self.available = False
        self.reason = None
        self._init_client()

    def _init_client(self):
        """Prepara el cliente y, si no se puede, deja dicho exactamente por qué.

        Distingue falta de paquete de falta de credencial: decir solo "configura
        la API key" cuando lo que falta es el paquete manda a perseguir el
        problema equivocado.
        """
        if self.provider not in self.PROVIDERS:
            self.reason = f"proveedor '{self.provider}' no soportado"
            logger.warning(f"Brazo C inactivo: {self.reason}")
            return

        paquete, env_var = self.PROVIDERS[self.provider]
        api_key = os.getenv(env_var)
        faltan = []

        try:
            if self.provider == "openai":
                from openai import OpenAI
            elif self.provider == "anthropic":
                from anthropic import Anthropic
            else:
                import google.generativeai as genai
        except ImportError:
            faltan.append(f"instala el paquete con 'pip install {paquete}'")

        if not api_key:
            faltan.append(f"define la variable {env_var}")

        if faltan:
            self.reason = f"Brazo C ({self.provider}) no disponible: " + " y ".join(faltan)
            logger.warning(self.reason)
            return

        try:
            if self.provider == "openai":
                self.client = OpenAI(api_key=api_key)
            elif self.provider == "anthropic":
                self.client = Anthropic(api_key=api_key)
            else:
                genai.configure(api_key=api_key)
            self.available = True
            logger.info(f"Brazo C activo con {self.provider}/{self.model}")
        except Exception as e:
            self.reason = f"Brazo C ({self.provider}) no disponible: {e}"
            logger.warning(self.reason)

    def predict(self, text: str) -> Tuple[str, str, float, str]:
        """Predice usando LLM. Retorna (tipo, prioridad, confianza, error_msg)."""
        if not self.available:
            return "falta_disciplinaria", "media", 0.0, self.reason

        try:
            prompt = f"""Clasifica esta incidencia escolar en dos dimensiones:

TEXTO: "{text}"

RESPONDE EXACTAMENTE EN ESTE FORMATO (sin explicaciones):
TIPO: [acoso|conflicto_estudiantes|daño_infraestructura|incidente_salud|falta_disciplinaria]
PRIORIDAD: [alta|media|baja]
CONFIANZA: [0.0 a 1.0]

Solo la respuesta, sin explicaciones."""

            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=100,
                )
                respuesta = response.choices[0].message.content

            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=100,
                    messages=[{"role": "user", "content": prompt}],
                )
                respuesta = response.content[0].text

            elif self.provider == "google":
                import google.generativeai as genai
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(prompt)
                respuesta = response.text

            else:
                return "falta_disciplinaria", "media", 0.0, f"Proveedor desconocido: {self.provider}"

            # Parsear respuesta
            tipo = "falta_disciplinaria"
            prioridad = "media"
            confianza = 0.75

            lines = respuesta.strip().split('\n')
            for line in lines:
                if 'TIPO:' in line:
                    for t in TIPOS:
                        if t in line.lower():
                            tipo = t
                            break
                elif 'PRIORIDAD:' in line:
                    for p in PRIORIDADES:
                        if p in line.lower():
                            prioridad = p
                            break
                elif 'CONFIANZA:' in line:
                    try:
                        conf_str = line.split(':')[1].strip()
                        confianza = float(conf_str)
                    except:
                        pass

            return tipo, prioridad, confianza, None

        except Exception as e:
            error_msg = f"Error en predicción Brazo C ({self.provider}): {str(e)}"
            logger.warning(error_msg)
            return "falta_disciplinaria", "media", 0.0, error_msg


class PredictionEnsemble:
    """Combina predicciones de los 3 brazos."""

    def __init__(self, provider_c: str = "openai"):
        """Inicializa los 3 predictores."""
        self.brazo_a = BrazoAPredictor()
        self.brazo_b = BrazoBPredictor()
        self.brazo_c = BrazoCPredictor(provider_c)

    def predict(self, text: str) -> Dict[str, Any]:
        """Predice con los 3 brazos y retorna resultados."""
        # Brazo A
        tipo_a, prior_a, conf_a = self.brazo_a.predict(text)

        # Brazo B
        tipo_b, prior_b, conf_b = self.brazo_b.predict(text)

        # Brazo C
        tipo_c, prior_c, conf_c, error_c = self.brazo_c.predict(text)

        # Preferencia C > B > A. Tipo, prioridad y confianza vienen del mismo
        # brazo: mezclarlos daría una confianza que no corresponde a la etiqueta.
        if conf_c > 0:
            fuente, tipo, prioridad, confianza = "C (LLM few-shot)", tipo_c, prior_c, conf_c
        elif conf_b > 0:
            fuente, tipo, prioridad, confianza = "B (fine-tuning)", tipo_b, prior_b, conf_b
        else:
            fuente, tipo, prioridad, confianza = "A (palabras clave)", tipo_a, prior_a, conf_a

        motivos = [m for m in (error_c, self.brazo_b.reason) if m]

        return {
            "tipo": tipo,
            "prioridad": prioridad,
            "confianza": round(confianza, 3),
            "fuente": fuente,
            "error": "; ".join(motivos) or None,
            "brazos": {
                "a": {"tipo": tipo_a, "prioridad": prior_a, "confianza": round(conf_a, 3)},
                "b": {"tipo": tipo_b, "prioridad": prior_b, "confianza": round(conf_b, 3),
                      "motivo": self.brazo_b.reason},
                "c": {"tipo": tipo_c, "prioridad": prior_c, "confianza": round(conf_c, 3), "error": error_c},
            }
        }
