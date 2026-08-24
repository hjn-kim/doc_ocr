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
        # pytesseract 는 껍데기라, 파이썬 패키지가 깔려 있어도 tesseract 실행
        # 파일이 없으면 문서마다 똑같은 예외를 뱉는다. 여기서 한 번에 걸러
        # 엔진째로 건너뛰게 만든다.
        try:
            pytesseract.get_tesseract_version()
        except Exception as exc:
            raise RuntimeError(
                "tesseract 실행 파일을 못 찾았다. pytesseract(파이썬 패키지)만으로는 "
                "안 되고 본체가 있어야 한다. sudo 가 없으면 "
                "conda install -y -c conda-forge tesseract 로 깔거나, "
                "--tesseract-cmd 로 실행 파일 경로를 알려줘라"
            ) from exc
        return pytesseract

    def run(self, pdf: Path, pages: list[np.ndarray], lang=None) -> list[str]:
        pytesseract = self.ensure_loaded()
        lang = lang or self.options.get("lang") or "eng"
        config = self.options.get("config", "--oem 3 --psm 3")
        return [pytesseract.image_to_string(page, lang=lang, config=config).strip()
                for page in pages]
