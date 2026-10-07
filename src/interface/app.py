"""C9: Interfaz web para los tres stakeholders (Flask)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request, session

from src.arms.arm_a_classic import ArmAClassic
from src.decision.abstention import AbstentionModule
from src.decision.asymmetric_priority import AsymmetricPrioritizer
from src.preprocessing.cleaner import RegisterAwarePreprocessor
from src.utils.io import load_config
from src.utils.logger import setup_logger

logger = setup_logger("motie_es.interface")

app = Flask(__name__)
app.secret_key = "motie-es-secret-key-change-in-production"

CONFIG_PATH = Path("config/config.yaml")
if CONFIG_PATH.exists():
    config = load_config(CONFIG_PATH)
else:
    config = {
        "preprocessing": {
            "lowercase": True, "remove_special_chars": True,
            "reduce_repeated_chars": True, "max_repeated": 3,
            "standardize_laughter": True, "normalize_emojis": True,
            "emoji_format": "text",
        },
        "decision": {
            "priority_levels": ["alta", "media", "baja"],
            "critical_level": "alta",
            "confidence_threshold": 0.70,
            "stage_discrepancy_threshold": 0.30,
        },
        "interface": {"max_text_length": 5000},
    }

PREPROCESSOR = RegisterAwarePreprocessor(
    **{k: v for k, v in config.get("preprocessing", {}).items()
       if k in ("lowercase", "remove_special_chars", "reduce_repeated_chars",
                "max_repeated", "standardize_laughter", "normalize_emojis", "emoji_format")}
)

# Almacén en memoria para demo (en producción usar base de datos)
INCIDENTS: Dict[str, Dict[str, Any]] = {}
ANNOTATIONS: Dict[str, Dict[str, Any]] = {}
INTERACTIONS: List[Dict[str, Any]] = []


def _log_interaction(stakeholder: str, action: str, details: Dict[str, Any]) -> None:
    """Registra una interacción para análisis posterior."""
    INTERACTIONS.append({
        "timestamp": datetime.utcnow().isoformat(),
        "stakeholder": stakeholder,
        "action": action,
        "details": details,
    })
    log_path = Path(config.get("paths", {}).get("results_logs", "results/logs"))
    log_path.mkdir(parents=True, exist_ok=True)
    with open(log_path / "interactions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(INTERACTIONS[-1], ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# Vistas
# ------------------------------------------------------------
@app.route("/")
def index() -> str:
    return render_template("base.html")


@app.route("/reportante", methods=["GET", "POST"])
def reportante() -> Any:
    """Vista del reportante: formulario simple de ingreso."""
    if request.method == "POST":
        data = request.get_json() or request.form.to_dict()
        texto = (data.get("texto") or "").strip()

        if not texto:
            return jsonify({"error": "El campo de texto es obligatorio."}), 400
        if len(texto) > config["interface"]["max_text_length"]:
            return jsonify({"error": f"El texto excede {config['interface']['max_text_length']} caracteres."}), 400

        incident_id = str(uuid.uuid4())[:8]
        cleaned = PREPROCESSOR.clean(texto)
        INCIDENTS[incident_id] = {
            "id": incident_id,
            "texto_original": texto,
            "texto_limpio": cleaned,
            "nivel_educativo": data.get("nivel_educativo", ""),
            "tipo_institucion": data.get("tipo_institucion", ""),
            "fecha": data.get("fecha", datetime.utcnow().date().isoformat()),
            "estado": "pendiente",
            "fecha_ingreso": datetime.utcnow().isoformat(),
            "tipo_predicho": None,
            "prioridad_predicha": None,
            "confianza": None,
            "explicacion": None,
        }
        _log_interaction("reportante", "crear_incidencia", {"id": incident_id})
        return jsonify({"ok": True, "id": incident_id, "mensaje": "Reporte registrado correctamente."})

    return render_template("reportante.html")


@app.route("/decisor")
def decisor() -> str:
    """Vista del decisor: panel de sugerencias."""
    return render_template("decisor.html")


@app.route("/evaluador")
def evaluador() -> str:
    """Vista del evaluador: anotación y discrepancias."""
    return render_template("evaluador.html")


# ------------------------------------------------------------
# API
# ------------------------------------------------------------
@app.route("/api/incidents", methods=["GET"])
def list_incidents() -> Any:
    return jsonify(list(INCIDENTS.values()))


@app.route("/api/incidents/<incident_id>", methods=["GET"])
def get_incident(incident_id: str) -> Any:
    inc = INCIDENTS.get(incident_id)
    if not inc:
        return jsonify({"error": "Incidencia no encontrada"}), 404
    return jsonify(inc)


@app.route("/api/incidents/<incident_id>/decision", methods=["POST"])
def decide_incident(incident_id: str) -> Any:
    """Registra la decisión del decisor (confirmar, ajustar, rechazar)."""
    inc = INCIDENTS.get(incident_id)
    if not inc:
        return jsonify({"error": "Incidencia no encontrada"}), 404

    data = request.get_json() or {}
    accion = data.get("accion")
    if accion not in ("confirmar", "ajustar", "rechazar"):
        return jsonify({"error": "Acción inválida"}), 400

    inc["decision"] = accion
    inc["decision_fecha"] = datetime.utcnow().isoformat()
    if accion == "ajustar":
        inc["tipo_ajustado"] = data.get("tipo")
        inc["prioridad_ajustada"] = data.get("prioridad")
    inc["estado"] = "decidido"

    _log_interaction("decisor", f"decision_{accion}", {"id": incident_id})
    return jsonify({"ok": True})


@app.route("/api/incidents/<incident_id>/predict", methods=["POST"])
def predict_incident(incident_id: str) -> Any:
    """Genera una sugerencia de triaje para una incidencia.

    En producción, aquí se invocaría el pipeline completo (Brazo A/B/C + C7).
    Para la demo, se genera una sugerencia simulada determinista.
    """
    inc = INCIDENTS.get(incident_id)
    if not inc:
        return jsonify({"error": "Incidencia no encontrada"}), 404

    texto = inc["texto_limpio"].lower()
    # Reglas heurísticas simples para demo
    if any(w in texto for w in ["golpe", "sangre", "violencia", "acoso", "arma"]):
        tipo, prioridad, confianza = "acoso", "alta", 0.85
    elif any(w in texto for w in ["pelea", "conflicto", "insulto"]):
        tipo, prioridad, confianza = "conflicto_estudiantes", "media", 0.75
    elif any(w in texto for w in ["vidrio", "daño", "rompi", "destru"]):
        tipo, prioridad, confianza = "daño_infraestructura", "media", 0.72
    elif any(w in texto for w in ["enferm", "salud", "fiebre", "desmay"]):
        tipo, prioridad, confianza = "incidente_salud", "alta", 0.80
    else:
        tipo, prioridad, confianza = "falta_disciplinaria", "baja", 0.65

    inc["tipo_predicho"] = tipo
    inc["prioridad_predicha"] = prioridad
    inc["confianza"] = confianza
    inc["explicacion"] = (
        f"Sugerencia generada con confianza {confianza:.2f}. "
        f"Tipo: {tipo}. Prioridad: {prioridad}."
    )
    inc["estado"] = "sugerido"

    _log_interaction("sistema", "prediccion", {"id": incident_id, "confianza": confianza})
    return jsonify(inc)


@app.route("/api/derived", methods=["GET"])
def list_derived() -> Any:
    """Lista casos derivados a revisión humana por baja confianza."""
    abstention = AbstentionModule(
        confidence_threshold=config["decision"]["confidence_threshold"],
        stage_discrepancy_threshold=config["decision"]["stage_discrepancy_threshold"],
    )
    derived = [
        inc for inc in INCIDENTS.values()
        if inc.get("confianza") is not None
        and abstention.should_abstain(inc["confianza"], inc.get("stage_discrepancy"))
    ]
    derived.sort(key=lambda x: x.get("prioridad_predicha") != "alta")
    return jsonify(derived)


@app.route("/api/annotations", methods=["GET", "POST"])
def annotations() -> Any:
    """Vista del evaluador: anotaciones de referencia."""
    if request.method == "POST":
        data = request.get_json() or {}
        ann_id = data.get("id")
        if not ann_id:
            return jsonify({"error": "Falta id"}), 400
        ANNOTATIONS[ann_id] = {
            "id": ann_id,
            "tipo_anotado": data.get("tipo"),
            "prioridad_anotada": data.get("prioridad"),
            "evaluador": data.get("evaluador", "anonimo"),
            "fecha": datetime.utcnow().isoformat(),
        }
        _log_interaction("evaluador", "anotar", {"id": ann_id})
        return jsonify({"ok": True})

    # Devolver incidencias con predicciones del Brazo C para comparar
    items = []
    for inc in INCIDENTS.values():
        if inc.get("tipo_predicho"):
            items.append({
                "id": inc["id"],
                "texto": inc["texto_original"],
                "tipo_predicho_arm_c": inc["tipo_predicho"],
                "prioridad_predicha_arm_c": inc["prioridad_predicha"],
                "confianza": inc["confianza"],
                "anotacion_referencia": inc.get("anotacion_referencia"),
                "anotacion_evaluador": ANNOTATIONS.get(inc["id"], {}),
            })
    return jsonify(items)


@app.route("/api/interactions", methods=["GET"])
def get_interactions() -> Any:
    return jsonify(INTERACTIONS)


@app.route("/health")
def health() -> Any:
    return jsonify({"status": "ok", "service": "MOTIE-ES"})


if __name__ == "__main__":
    app.run(
        host=config["interface"]["host"],
        port=config["interface"]["port"],
        debug=config["interface"]["debug"],
    )