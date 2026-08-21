# -*- coding: utf-8 -*-
"""Tesseract.

    pip install pytesseract
    apt-get install tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng ...

GPU 를 쓰지 않는다. 세 엔진 중 이것만 CPU 라 --device 를 무시한다.
언어는 문서 그룹에 맞춰 매 문서 바뀔 수 있다(languages.py).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import OCREngine


class TesseractEngine(OCREngine):
    name = "tesseract"
    lang_aware = True

    @property
    def uses_gpu(self) -> bool:
        return False

    def load(self):
        import pytesseract

        cmd = self.options.get("cmd")
        if cmd:
            pytesseract.pytesseract.tesseract_cmd = cmd
        return pytesseract

    def run(self, pdf: Path, pages: list[np.ndarray], lang=None) -> list[str]:
        pytesseract = self.ensure_loaded()
        lang = lang or self.options.get("lang") or "eng"
        config = self.options.get("config", "--oem 3 --psm 3")
        return [pytesseract.image_to_string(page, lang=lang, config=config).strip()
                for page in pages]
