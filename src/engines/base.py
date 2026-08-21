# -*- coding: utf-8 -*-
"""엔진 공통 인터페이스.

무거운 import 는 전부 load() 안에 둔다. paddle 이 안 깔린 환경에서도
tesseract 만 돌릴 수 있어야 하기 때문이다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class OCREngine:
    name = "base"
    # 문서 그룹(en/ko/...)마다 언어를 바꿔 끼워야 하는 엔진인지
    lang_aware = False

    def __init__(self, device: str = "cpu", **kwargs):
        self.device = device
        self.options = kwargs
        self._model = None

    @property
    def uses_gpu(self) -> bool:
        return self.device not in ("cpu", "", None)

    def load(self):
        """모델을 메모리에 올린다. 첫 문서에서 한 번만 호출된다."""
        raise NotImplementedError

    def run(self, pdf: Path, pages: list[np.ndarray], lang=None) -> list[str]:
        """페이지별 인식 텍스트를 순서대로 돌려준다."""
        raise NotImplementedError

    def ensure_loaded(self):
        if self._model is None:
            self._model = self.load()
        return self._model

    def describe(self) -> str:
        return f"{self.name} (device={self.device})"
