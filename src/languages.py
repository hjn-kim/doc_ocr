# -*- coding: utf-8 -*-
"""data/ 하위 폴더 이름(en, ko, ch, ...)을 엔진별 언어 설정으로 바꾼다.

tesseract 와 easyocr 은 어떤 언어를 읽을지 미리 정해줘야 한다. 폴더가 언어별로
나뉘어 있으니 그 이름을 그대로 언어 힌트로 쓴다. PaddleOCR-VL 은 다국어 모델
하나가 전부 처리하므로 설정이 없다.

기본 표를 바꾸고 싶으면 JSON 을 만들어 --lang-map 으로 넘긴다.

    {"pil": {"tesseract": "tgl+eng", "easyocr": ["tl", "en"]}}
"""
from __future__ import annotations

import json
from pathlib import Path

# tesseract 는 traineddata 이름, easyocr 은 지원 언어 코드다.
# 영어를 같이 넣는 이유는 판결문/보도자료에 라틴 문자와 숫자가 섞여 나오기 때문이다.
DEFAULT_LANGS: dict[str, dict] = {
    "en":  {"tesseract": "eng",          "easyocr": ["en"]},
    "ko":  {"tesseract": "kor+eng",      "easyocr": ["ko", "en"]},
    "ch":  {"tesseract": "chi_sim+eng",  "easyocr": ["ch_sim", "en"]},
    "ru":  {"tesseract": "rus+eng",      "easyocr": ["ru", "en"]},
    "vn":  {"tesseract": "vie+eng",      "easyocr": ["vi", "en"]},
    "uz":  {"tesseract": "uzb+eng",      "easyocr": ["uz", "en"]},
    "pil": {"tesseract": "tgl+eng",      "easyocr": ["tl", "en"]},   # 필리핀 = 타갈로그
}
FALLBACK = {"tesseract": "eng", "easyocr": ["en"]}


def load_map(path: Path | None) -> dict[str, dict]:
    """기본 표 위에 JSON 을 덮어쓴다."""
    table = {k: dict(v) for k, v in DEFAULT_LANGS.items()}
    if path:
        override = json.loads(Path(path).read_text(encoding="utf-8"))
        for group, spec in override.items():
            table.setdefault(group, {}).update(spec)
    return table


def for_engine(table: dict[str, dict], group: str, engine: str):
    """해당 그룹에서 엔진이 쓸 언어 설정. 모르는 그룹이면 영어로 떨어진다."""
    return table.get(group, {}).get(engine, FALLBACK.get(engine))


def describe(spec) -> str:
    """CSV 에 적을 문자열."""
    if spec is None:
        return ""
    return "+".join(spec) if isinstance(spec, (list, tuple)) else str(spec)
