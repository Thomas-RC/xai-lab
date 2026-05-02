"""Tlumaczenie etykiet ImageNet EN -> PL przez Gemini 2.5 Flash na Vertex AI."""

from src.translation.gemini import translate_labels

__all__ = ["translate_labels"]
