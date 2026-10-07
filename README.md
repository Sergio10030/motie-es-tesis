# MOTIE-ES

**Modelo de Triaje de Incidencias Escolares en Español Informal**

Artefacto software reproducible que implementa el modelo MOTIE-ES descrito en el artículo *"Triaje automático de incidencias escolares en español informal basado en modelos generativos en modo few-shot"*.

## Descripción

MOTIE-ES clasifica reportes narrativos de incidencias escolares redactados en español informal administrativo en dos dimensiones:

- **Tipo funcional:** conflicto entre estudiantes, daño a infraestructura, acoso, incidente de salud, falta disciplinaria.
- **Nivel de prioridad:** alta, media, baja.

El sistema implementa **tres brazos experimentales** comparables:

- **Brazo A:** línea base clásica (TF-IDF + regresión logística).
- **Brazo B:** fine-tuning de encoders preentrenados en español (BETO, RoBERTuito, BERTuit, RoBERTa-BNE).
- **Brazo C:** clasificador few-shot jerárquico en dos etapas con LLMs (GPT-4o, Claude, Gemini, LLaMA, DeepSeek).

Incluye módulos de priorización con coste asimétrico, abstención y derivación a revisión humana, validación de concordancia con criterio humano, y una interfaz web para tres stakeholders: reportante, decisor y evaluador.

## Estructura del proyecto

```text
config/       Configuración y plantillas de prompts
data/         Dataset original, léxicos y particiones procesadas
experiments/  Scripts reproducibles para preparación y brazos A/B/C
notebooks/    Análisis exploratorio, experimentos y comparación
results/      Métricas, registros y figuras generadas
src/          Preprocesamiento, modelos, evaluación e interfaz
tests/        Pruebas automatizadas
```

## Requisitos e instalación en Windows

Se recomienda Python 3.11 nativo de Windows (también compatible con las dependencias fijadas). Desde PowerShell, en la carpeta raíz del proyecto:

```powershell
py -3.11 -m venv .venv-windows
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv-windows\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-core.txt
```

Usa `.venv-windows` en PowerShell. No actives una `.venv` creada por MSYS/Git Bash: puede tener `bin/` en vez de `Scripts/` y no incluir los paquetes del proyecto. Comprueba el entorno con `python --version`, `python -c "import sys; print(sys.executable)"` y `python -m pytest -q`.

`requirements-core.txt` basta para preparar datos, ejecutar el Brazo A, correr las pruebas y abrir la interfaz. Para entrenar el Brazo B o usar proveedores del Brazo C, instala además el conjunto completo con `pip install -r requirements.txt`; B descargará modelos y C puede requerir una API key.

## Preparación y ejecución

Ejecuta desde la raíz del repositorio para que las rutas de `config/config.yaml` se resuelvan correctamente:

```powershell
python -m experiments.run_prepare_data
python -m experiments.run_arm_a
```

La preparación guarda el corpus limpio y las particiones en `data/processed/`. El Brazo A entrena TF-IDF + regresión logística y escribe métricas en `results/metrics/`.

El Brazo B descarga un encoder de Transformers y realiza fine-tuning; puede requerir bastante memoria, tiempo y conexión a internet:

```powershell
python -m experiments.run_arm_b
```

El Brazo C usa el proveedor configurado en `config/config.yaml`. Configura la variable de entorno de su API key antes de ejecutarlo si se desea hacer llamadas reales. Sin clave, el código usa respuestas simuladas, que sirven para comprobar el flujo pero no para medir la calidad de un LLM:

```powershell
$env:OPENAI_API_KEY = "<tu-clave>"
python -m experiments.run_arm_c
```

`python -m experiments.run_all` ejecuta preparación y los tres brazos (B puede descargar dos modelos y C puede llamar al proveedor configurado). Después de generar `data/processed/test_with_preds.csv`, complete las columnas `anotacion_humana` y `prioridad_anotacion_humana`; después ejecute `python -m experiments.run_all --validate-only`. La validación comprueba ambos objetivos y no vuelve a generar ni sobrescribe la plantilla.

Ejecuta las pruebas con `python -m pytest -q`. En VS Code, selecciona `.venv-windows\Scripts\python.exe` como intérprete y kernel de los notebooks. Los notebooks están en `notebooks/`; B y C requieren activar manualmente sus variables `RUN_FINETUNING` y `RUN_LLM_EXPERIMENT` para iniciar trabajos costosos.

## Dataset y alcance

El archivo disponible en `data/raw/` contiene sugerencias y quejas ciudadanas, no reportes escolares anotados para las categorías objetivo. La preparación lo adapta para probar técnicamente el flujo: `Tipo` se convierte en etiquetas generales (`queja_general`, `sugerencia_general`, etc.) y `Estado` se usa como aproximación de prioridad. Esas etiquetas no representan una anotación humana de gravedad ni la taxonomía escolar descrita arriba.

Por tanto, las métricas generadas con este CSV solo verifican que el pipeline funciona con datos reales y no deben interpretarse como evidencia de desempeño del modelo MOTIE-ES. Para una evaluación válida se necesita un corpus escolar con texto y etiquetas revisadas por personas. La validación de concordancia también requiere `data/processed/test_with_preds.csv` con predicciones y anotaciones humanas.

## Interfaz

Inicia la aplicación Flask desde la raíz:

```powershell
python -m src.interface.app
```

La dirección y el puerto se configuran en `config/config.yaml`.
