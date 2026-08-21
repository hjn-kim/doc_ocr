# -*- coding: utf-8 -*-
"""EasyOCR.

    pip install easyocr

Reader 하나에 아무 언어나 섞어 넣을 수는 없다(한국어와 중국어를 같이 못 넣는다).
그래서 문서 그룹이 바뀌면 Reader 를 새로 만들고 이전 것은 버린다. 문서는 경로
순으로 도니까 같은 그룹이 붙어 있어 교체는 그룹 수만큼만 일어난다.

detail=0 으로 글자만 받는다. paragraph=True 면 같은 문단의 줄을 붙여준다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import OCREngine


class EasyOCREngine(OCREngine):
    name = "easyocr"
    lang_aware = True

    def __init__(self, device: str = "cpu", **kwargs):
        super().__init__(device, **kwargs)
        self._langs = None

    def _make_reader(self, langs):
        import easyocr

        return easyocr.Reader(list(langs), gpu=self.uses_gpu,
                              model_storage_directory=self.options.get("model_dir"))

    def load(self):
        self._langs = tuple(self.options.get("langs") or ["en"])
        return self._make_reader(self._langs)

    def _reader_for(self, lang):
        want = tuple(lang) if lang else None
        if self._model is None:
            self.ensure_loaded()
        if want and want != self._langs:
            self._model = None                 # 이전 Reader 를 먼저 놓아준다
            self._langs = want
            self._model = self._make_reader(want)
        return self._model

    def run(self, pdf: Path, pages: list[np.ndarray], lang=None) -> list[str]:
        reader = self._reader_for(lang)
        paragraph = bool(self.options.get("paragraph", True))
        out = []
        for page in pages:
            lines = reader.readtext(page, detail=0, paragraph=paragraph)
            out.append("\n".join(str(x).strip() for x in lines if str(x).strip()))
        return out
