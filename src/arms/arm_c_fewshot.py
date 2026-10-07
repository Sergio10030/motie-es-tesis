"""C6: Brazo C - Clasificador few-shot jerárquico en dos etapas."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger
from src.utils.seed import set_seed

logger = setup_logger("motie_es.arm_c")


class LLMClient:
    """Cliente unificado para APIs de LLM (OpenAI, Anthropic, Google, etc.)."""

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-4o",
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 100,
        max_retries: int = 3,
    ) -> None:
        self.provider = provider
        self.model = model
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            logger.warning(
                "No se encontró la variable de entorno %s. "
                "El cliente operará en modo simulado.",
                api_key_env,
            )

    def query(self, prompt: str) -> str:
        """Consulta al LLM y devuelve la respuesta cruda.

        Si no hay API key, devuelve una respuesta simulada para pruebas.
        """
        if not self.api_key:
            return self._simulated_response(prompt)

        for attempt in range(self.max_retries):
            try:
                return self._dispatch(prompt)
            except Exception as exc:
                logger.warning(
                    "Error en intento %d/%d: %s", attempt + 1, self.max_retries, exc
                )
                time.sleep(1.0 * (attempt + 1))
        raise RuntimeError(f"Fallo tras {self.max_retries} reintentos")

    def _dispatch(self, prompt: str) -> str:
        """Despacha la consulta al proveedor configurado."""
        if self.provider == "openai":
            return self._query_openai(prompt)
        if self.provider == "anthropic":
            return self._query_anthropic(prompt)
        if self.provider == "google":
            return self._query_google(prompt)
        if self.provider in ("meta", "deepseek"):
            return self._query_openai_compatible(prompt)
        raise ValueError(f"Proveedor no soportado: {self.provider}")

    def _query_openai(self, prompt: str) -> str:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content or ""

    def _query_openai_compatible(self, prompt: str) -> str:
        return self._query_openai(prompt)

    def _query_anthropic(self, prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text if response.content else ""

    def _query_google(self, prompt: str) -> str:
        import google.generativeai as genai

        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": self.temperature, "max_output_tokens": self.max_tokens},
        )
        return response.text or ""

    @staticmethod
    def _simulated_response(prompt: str) -> str:
        """Respuesta simulada determinista para pruebas sin API."""
        prompt_lower = prompt.lower()
        if "categoría de tipo funcional" in prompt_lower:
            # Extraer categorías del prompt
            import re

            match = re.search(r"categorías de tipo funcional:\s*(.+?)\n", prompt_lower, re.DOTALL)
            if match:
                cats = [c.strip() for c in match.group(1).split(",") if c.strip()]
                if cats:
                    return cats[0]
            return "conflicto_estudiantes"
        if "nivel de prioridad" in prompt_lower:
            return "media"
        return "media"


class ArmCFewShot:
    """Brazo C: clasificador few-shot jerárquico en dos etapas.

    Etapa 1: clasificación de tipo funcional.
    Etapa 2: priorización condicionada por el tipo predicho.

    Selección de ejemplos por diversidad de registro.
    """

    def __init__(
        self,
        client: Optional[LLMClient] = None,
        k_shot: int = 5,
        temperature: float = 0.0,
        selection_strategy: str = "register_diversity",
        fallback_label: str = "revision_humana",
        prompts_dir: str = "config/prompts",
        seed: int = 42,
    ) -> None:
        self.client = client or LLMClient()
        self.k_shot = k_shot
        self.temperature = temperature
        self.selection_strategy = selection_strategy
        self.fallback_label = fallback_label
        self.prompts_dir = Path(prompts_dir)
        self.seed = seed
        set_seed(seed)

        self.prompt_etapa1 = self._load_prompt("etapa1_tipo.txt")
        self.prompt_etapa2 = self._load_prompt("etapa2_prioridad.txt")
        self.valid_types: List[str] = []
        self.valid_priorities: List[str] = ["alta", "media", "baja"]
        self.few_shot_examples: Dict[str, List[Dict[str, str]]] = {}
        self.query_log: List[Dict[str, Any]] = []

    def _load_prompt(self, filename: str) -> str:
        path = self.prompts_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el prompt: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def set_taxonomy(self, types: List[str], priorities: Optional[List[str]] = None) -> None:
        """Define la taxonomía de tipos y prioridades."""
        self.valid_types = types
        if priorities:
            self.valid_priorities = priorities

    def select_examples(
        self,
        df: pd.DataFrame,
        text_col: str = "texto",
        type_col: str = "tipo_funcional",
        priority_col: str = "prioridad",
        density_jerga_col: str = "densidad_jerga",
        density_admin_col: str = "densidad_admin",
    ) -> Dict[str, List[Dict[str, str]]]:
        """Selecciona K ejemplos por clase por diversidad de registro."""
        examples: Dict[str, List[Dict[str, str]]] = {}
        for tipo in self.valid_types:
            subset = df[df[type_col] == tipo]
            if subset.empty:
                examples[tipo] = []
                continue

            if self.selection_strategy == "register_diversity" and density_jerga_col in df.columns:
                # Ordenar por diversidad: combinar densidad de jerga y admin
                subset = subset.copy()
                subset["_div"] = (
                    subset[density_jerga_col].rank(pct=True)
                    + subset[density_admin_col].rank(pct=True)
                )
                subset = subset.sort_values("_div", ascending=False)

            k = min(self.k_shot, len(subset))
            selected = subset.head(k)
            examples[tipo] = [
                {
                    "texto": str(row[text_col]),
                    "tipo": str(row[type_col]),
                    "prioridad": str(row[priority_col]) if priority_col in row else "",
                }
                for _, row in selected.iterrows()
            ]
        self.few_shot_examples = examples
        return examples

    def _format_examples(self, examples: List[Dict[str, str]], include_priority: bool = False) -> str:
        """Formatea ejemplos para el prompt."""
        if not examples:
            return "(No hay ejemplos disponibles)"
        lines = []
        for i, ex in enumerate(examples, 1):
            label = ex.get("tipo", "")
            if include_priority and ex.get("prioridad"):
                label = f"{ex['tipo']} | prioridad: {ex['prioridad']}"
            lines.append(f"Ejemplo {i}: \"{ex['texto']}\" -> {label}")
        return "\n".join(lines)

    def _build_prompt_etapa1(self, report: str) -> str:
        taxonomy = "\n".join(f"- {t}" for t in self.valid_types)
        examples_lines = []
        for tipo, exs in self.few_shot_examples.items():
            for ex in exs:
                examples_lines.append(f"Reporte: \"{ex['texto']}\" -> Tipo: {tipo}")
        examples = "\n".join(examples_lines) if examples_lines else "(No hay ejemplos)"
        return self.prompt_etapa1.format(
            taxonomy=taxonomy,
            examples=examples,
            report=report,
        )

    def _build_prompt_etapa2(self, report: str, predicted_type: str) -> str:
        examples_lines = []
        for tipo, exs in self.few_shot_examples.items():
            for ex in exs:
                if ex.get("prioridad"):
                    examples_lines.append(
                        f"Reporte: \"{ex['texto']}\" -> Tipo: {tipo} | Prioridad: {ex['prioridad']}"
                    )
        examples = "\n".join(examples_lines) if examples_lines else "(No hay ejemplos)"
        return self.prompt_etapa2.format(
            predicted_type=predicted_type,
            examples=examples,
            report=report,
        )

    def _parse_response(self, response: str, valid_labels: List[str]) -> str:
        """Parsea la respuesta del LLM y valida la etiqueta."""
        if not response:
            return self.fallback_label
        text = response.strip().lower()
        # Buscar etiqueta válida en la respuesta
        for label in valid_labels:
            if label.lower() in text:
                return label
        return self.fallback_label

    def classify_one(self, report: str) -> Dict[str, Any]:
        """Clasifica un reporte en dos etapas."""
        # Etapa 1: tipo funcional
        prompt1 = self._build_prompt_etapa1(report)
        raw1 = self.client.query(prompt1)
        tipo = self._parse_response(raw1, self.valid_types)
        tipo_valido = tipo in self.valid_types

        # Etapa 2: prioridad condicionada
        prompt2 = self._build_prompt_etapa2(report, tipo)
        raw2 = self.client.query(prompt2)
        prioridad = self._parse_response(raw2, self.valid_priorities)
        prioridad_valida = prioridad in self.valid_priorities

        result = {
            "tipo_predicho": tipo if tipo_valido else self.fallback_label,
            "prioridad_predicha": prioridad if prioridad_valida else self.fallback_label,
            "tipo_valido": tipo_valido,
            "prioridad_valida": prioridad_valida,
            "prompt_etapa1": prompt1,
            "prompt_etapa2": prompt2,
            "respuesta_cruda_etapa1": raw1,
            "respuesta_cruda_etapa2": raw2,
            "confianza": 1.0 if (tipo_valido and prioridad_valida) else 0.5,
        }
        self.query_log.append(result)
        return result

    def classify(self, reports: List[str]) -> List[Dict[str, Any]]:
        """Clasifica una lista de reportes."""
        results = []
        for i, report in enumerate(reports, 1):
            logger.info("Clasificando reporte %d/%d", i, len(reports))
            results.append(self.classify_one(report))
        return results