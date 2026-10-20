"""Compatibilidad temporal.

La lógica principal fue movida a app/core, app/documents y app/services/pipeline.
Este módulo se conserva para que imports antiguos no rompan durante la migración.
"""
from app.services.pipeline import analyze_document, redact_analysis

__all__ = ["analyze_document", "redact_analysis"]
