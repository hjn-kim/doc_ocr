# -*- coding: utf-8 -*-
"""엔진 레지스트리."""
from __future__ import annotations

from .base import OCREngine
from .easyocr_engine import EasyOCREngine
from .paddle_vl import PaddleVLEngine
from .tesseract_engine import TesseractEngine

ENGINES = {
    "paddleocr_vl": PaddleVLEngine,
    "tesseract": TesseractEngine,
    "easyocr": EasyOCREngine,
}
ALIASES = {"paddle": "paddleocr_vl", "paddleocr": "paddleocr_vl",
           "vl": "paddleocr_vl", "easy": "easyocr", "tess": "tesseract"}


def resolve(name: str) -> str:
    key = name.strip().lower()
    key = ALIASES.get(key, key)
    if key not in ENGINES:
        raise ValueError(f"모르는 엔진: {name} (가능: {', '.join(ENGINES)})")
    return key


def build(name: str, device: str, **options) -> OCREngine:
    key = resolve(name)
    return ENGINES[key](device=device, **options)


__all__ = ["OCREngine", "ENGINES", "build", "resolve"]
