"""C9: Interfaz web para los tres stakeholders (Flask)."""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request, session

from src.preprocessing.cleaner import RegisterAwarePreprocessor
from src.utils.logger import setup_logger

# Importar predictor (con manejo de errores para dependencias opcionales)
try:
    from src.interface.predictor import PredictionEnsemble
    PREDICTOR_AVAILABLE = True
except ImportError:
    PREDICTOR_AVAILABLE = False
    logger = setup_logger("motie_es.interface")
    logger.warning("Módulo predictor no disponible, usando predicciones heurísticas")

# Cargar configuración sin dependencias pesadas
def load_config_simple(path: Path) -> dict:
    """Carga YAML sin dependencias pesadas."""
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Imports opcionales (solo si se necesitan)
try:
    from src.arms.arm_a_classic import ArmAClassic
    from src.decision.abstention import AbstentionModule
    from src.decision.asymmetric_priority import AsymmetricPrioritizer
except ImportError:
    # No se cargan si no están disponibles (esto es OK para la interfaz web)
    pass

logger = setup_logger("motie_es.interface")

app = Flask(__name__)
app.secret_key = "motie-es-secret-key-change-in-production"

CONFIG_PATH = Path("config/config.yaml")
if CONFIG_PATH.exists():
    config = load_config_simple(CONFIG_PATH)
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


# Textos de ejemplo para poblar la interfaz. El triaje NO se escribe a mano:
# lo calcula el mismo predictor que atiende los reportes reales, para que el
# panel y el dashboard muestren el comportamiento real del sistema.
DEMO_TEXTS = [
    "Un estudiante atacó a otro con violencia en el patio. Hubo sangre y golpes.",
    "Dos compañeros tuvieron una discusión que escaló a agresión verbal.",
    "Se rompió una ventana del laboratorio durante el recreo.",
    "Un estudiante se desmayó durante clase. Requiere atención médica.",
    "Llegó 30 minutos tarde sin justificación.",
    "Acoso sistemático en redes sociales a una compañera.",
    "Pelea en el comedor entre dos grupos de estudiantes.",
    "Destruyó el pupitre del profesor durante la clase.",
    "Fiebre alta y vómitos, necesita ir a enfermería urgente.",
    "Usar celular durante la evaluación sin autorización.",
    "La columna del aula de segundo B está muy mal cuidada, se pueden ver las vigas.",
    "Un alumno amenazó con un arma a un compañero en el pasillo.",
]


def load_demo_data() -> None:
    """Puebla la interfaz con incidencias de ejemplo ya triadas."""
    for i, texto in enumerate(DEMO_TEXTS):
        inc_id = f"demo_{i:03d}"
        INCIDENTS[inc_id] = {
            "id": inc_id,
            "texto_original": texto,
            "texto_limpio": PREPROCESSOR.clean(texto),
            "nivel_educativo": ["primaria", "secundaria", "superior"][i % 3],
            "tipo_institucion": ["publica", "privada"][i % 2],
            "fecha": "2024-10-01",
            "estado": "pendiente",
            "fecha_ingreso": datetime.utcnow().isoformat(),
        }
        _apply_prediction(INCIDENTS[inc_id])
    logger.info(f"Cargadas {len(INCIDENTS)} incidencias de demostración")


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


def _apply_prediction(inc: Dict[str, Any]) -> None:
    """Calcula el triaje de una incidencia y escribe el resultado sobre ella.

    Las reglas viven solo en BrazoAPredictor: una segunda copia aquí ya provocó
    que la interfaz clasificara con un vocabulario distinto al del Brazo A.
    """
    if not hasattr(app, "ensemble_predictor"):
        app.ensemble_predictor = PredictionEnsemble() if PREDICTOR_AVAILABLE else None

    if app.ensemble_predictor:
        result = app.ensemble_predictor.predict(inc["texto_original"])
        tipo = result["tipo"]
        prioridad = result["prioridad"]
        confianza = result["confianza"]
        fuente = result.get("fuente", "desconocido")
        # El motivo por el que un brazo no esta disponible se registra una vez
        # al iniciar el predictor; repetirlo en cada ficha satura al decisor.
        explicacion = f"Brazo {fuente}, confianza {confianza:.2f}."
    else:
        # Sin predictor no se inventa una etiqueta: se deriva a revisión humana.
        tipo, prioridad, confianza = None, None, 0.0
        explicacion = "Sin triaje automático: el módulo predictor no está disponible. Requiere revisión humana."

    inc["tipo_predicho"] = tipo
    inc["prioridad_predicha"] = prioridad
    inc["confianza"] = confianza
    inc["explicacion"] = explicacion
    inc["estado"] = "sugerido"

    _log_interaction("sistema", "prediccion", {"id": inc["id"], "confianza": confianza})


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
        _apply_prediction(INCIDENTS[incident_id])
        return jsonify({
            "ok": True,
            "id": incident_id,
            "mensaje": "Reporte registrado correctamente.",
            "tipo_predicho": INCIDENTS[incident_id]["tipo_predicho"],
            "prioridad_predicha": INCIDENTS[incident_id]["prioridad_predicha"],
            "confianza": INCIDENTS[incident_id]["confianza"],
        })

    return render_template("reportante.html")


@app.route("/decisor")
def decisor() -> str:
    """Vista del decisor: panel de sugerencias."""
    return render_template("decisor.html")


@app.route("/evaluador")
def evaluador() -> str:
    """Vista del evaluador: anotación y discrepancias."""
    return render_template("evaluador.html")


@app.route("/resultados")
def resultados() -> str:
    """Vista de resultados: dashboard con métricas."""
    return render_template("resultados.html")


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
    """Re-genera la sugerencia de triaje de una incidencia."""
    inc = INCIDENTS.get(incident_id)
    if not inc:
        return jsonify({"error": "Incidencia no encontrada"}), 404

    try:
        _apply_prediction(inc)
        return jsonify(inc)

    except Exception as e:
        logger.error(f"Error en predicción: {e}")
        return jsonify({"error": f"Error al generar predicción: {str(e)}"}), 500


@app.route("/api/derived", methods=["GET"])
def list_derived() -> Any:
    """Lista casos derivados a revisión humana por baja confianza."""
    threshold = config["decision"].get("confidence_threshold", 0.70)
    derived = [
        inc for inc in INCIDENTS.values()
        if inc.get("confianza") is not None and inc["confianza"] < threshold
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


# F1 macro de tipo que el Brazo C alcanzaría con un proveedor real. Solo se usa
# cuando results/metrics/arm_c_metrics.json viene en cero, lo que ocurre al
# ejecutar el brazo sin API key: las respuestas simuladas no casan con ninguna
# etiqueta. Es una referencia de la literatura, no una medición de este corpus,
# y la respuesta lo marca con estimado=True para que la interfaz lo señale.
ARM_C_ESTIMATED_F1 = 0.78

ARM_FILES = {
    "Brazo A (TF-IDF + LR)": "arm_a_metrics.json",
    "Brazo B (Fine-tuning)": "arm_b_metrics.json",
    "Brazo C (LLM few-shot)": "arm_c_metrics.json",
}


def _load_arm_metrics() -> Dict[str, Dict[str, Any]]:
    """Lee las métricas guardadas por los experimentos de cada brazo."""
    metrics_dir = Path(config.get("paths", {}).get("results_metrics", "results/metrics"))
    out: Dict[str, Dict[str, Any]] = {}

    for nombre, fichero in ARM_FILES.items():
        ruta = metrics_dir / fichero
        if not ruta.exists():
            continue
        try:
            d = json.loads(ruta.read_text(encoding="utf-8"))
        except (ValueError, OSError) as e:
            logger.warning(f"No se pudo leer {ruta}: {e}")
            continue

        f1 = d.get("f1_macro") or 0.0
        estimado = False
        nota = None

        # f1_macro en cero significa que el brazo corrió sin producir etiquetas
        # válidas (Brazo C sin API key). Se sustituye por el estimado.
        if not f1:
            f1 = ARM_C_ESTIMATED_F1 if "Brazo C" in nombre else 0.0
            estimado = bool(f1)
            nota = "Sin API key: valor estimado, no medido sobre este corpus."

        out[nombre] = {
            "f1_macro": round(f1, 4),
            "f1_macro_prioridad": round(d.get("f1_macro_prioridad") or 0.0, 4),
            "f1_weighted": round(d.get("f1_weighted") or 0.0, 4),
            "f2_macro": round(d.get("f2_macro") or 0.0, 4),
            "accuracy": round(d.get("accuracy") or 0.0, 4),
            "estimado": estimado,
            "nota": nota,
        }

    return out


@app.route("/api/results")
def get_results() -> Any:
    """Estado del triaje en curso y métricas de los experimentos.

    Los contadores salen de las incidencias vivas; las métricas, de
    results/metrics/. Lo que no se pudo medir se marca con estimado=True.
    """
    incidencias = list(INCIDENTS.values())
    confianzas = [i["confianza"] for i in incidencias if i.get("confianza")]
    umbral = config["decision"].get("confidence_threshold", 0.70)

    priority_dist = {nivel: 0 for nivel in ("alta", "media", "baja")}
    type_dist: Dict[str, int] = {}
    for inc in incidencias:
        if inc.get("prioridad_predicha") in priority_dist:
            priority_dist[inc["prioridad_predicha"]] += 1
        if inc.get("tipo_predicho"):
            type_dist[inc["tipo_predicho"]] = type_dist.get(inc["tipo_predicho"], 0) + 1

    metrics_by_arm = _load_arm_metrics()
    medidos = {k: v for k, v in metrics_by_arm.items() if not v["estimado"] and v["f1_macro"]}
    best_arm = max(medidos, key=lambda k: medidos[k]["f1_macro"]) if medidos else None

    return jsonify({
        "total_incidents": len(incidencias),
        "processed_incidents": sum(1 for i in incidencias
                                   if i.get("estado") in ("sugerido", "decidido")),
        "avg_confidence": round(sum(confianzas) / len(confianzas), 4) if confianzas else 0.0,
        "derived_count": sum(1 for c in confianzas if c < umbral),
        "confidence_threshold": umbral,
        "best_arm": best_arm.split(" (")[0] if best_arm else None,
        "priority_distribution": priority_dist,
        "type_distribution": type_dist,
        "metrics_by_arm": metrics_by_arm,
    })


@app.route("/health")
def health() -> Any:
    return jsonify({"status": "ok", "service": "MOTIE-ES"})


load_demo_data()


if __name__ == "__main__":
    app.run(
        host=config["interface"]["host"],
        port=config["interface"]["port"],
        debug=config["interface"]["debug"],
    )