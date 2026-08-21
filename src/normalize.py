# -*- coding: utf-8 -*-
"""OCR 결과와 정답을 같은 기준으로 맞춘 뒤 비교하기 위한 정규화.

지표는 정규화 방식에 따라 크게 흔들린다. 어떤 방식으로 쟀는지 result.csv 에
같이 적어두려고 이름을 붙여 관리한다.
"""
from __future__ import annotations

import re
import unicodedata

# 유니코드 상 다양한 따옴표/붙임표를 ASCII 로 눕힌다. OCR 엔진마다 고르는 글자가
# 달라서, 이걸 안 맞추면 엔진 성능이 아니라 취향 차이를 재게 된다.
_PUNCT_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201b": "'", "\u2032": "'",
    "\u201c": '"', "\u201d": '"', "\u201f": '"', "\u2033": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-",
    "\u2014": "-", "\u2015": "-", "\u2212": "-",
    "\u00a0": " ", "\u200b": "", "\ufeff": "",
}
_PUNCT_RE = re.compile("|".join(map(re.escape, _PUNCT_MAP)))
_WS_RE = re.compile(r"\s+")
_ALLWS_RE = re.compile(r"\s")
# 문장부호 전부 (유니코드 카테고리 P/S) — nopunct 모드에서만 쓴다.
_MD_RE = re.compile(r"[*_`#>|~]+")

PROFILES = ("none", "basic", "lower", "nospace", "nopunct")


def _strip_markdown(text: str) -> str:
    """PaddleOCR-VL 의 markdown 출력에서 서식 기호만 걷어낸다."""
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)      # 이미지
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)   # 링크는 표시 텍스트만
    text = re.sub(r"^\s*[-=]{3,}\s*$", " ", text, flags=re.M)
    return _MD_RE.sub("", text)


def normalize(text: str, profile: str = "basic", strip_markdown: bool = False) -> str:
    """profile 에 따라 텍스트를 정규화한다.

    none    : 아무것도 하지 않는다 (앞뒤 공백만 제거)
    basic   : NFKC + 문장부호 통일 + 공백 1칸으로 축약        <- 기본값
    lower   : basic + 소문자화 (영문 대소문자 차이를 무시)
    nospace : basic + 공백 전부 제거 (띄어쓰기를 안 보는 한국어 평가용)
    nopunct : lower + 문장부호 제거
    """
    if profile not in PROFILES:
        raise ValueError(f"알 수 없는 정규화 프로파일: {profile} (가능: {PROFILES})")
    if text is None:
        return ""
    if strip_markdown:
        text = _strip_markdown(text)
    if profile == "none":
        return text.strip()

    text = unicodedata.normalize("NFKC", text)
    text = _PUNCT_RE.sub(lambda m: _PUNCT_MAP[m.group(0)], text)
    if profile in ("lower", "nopunct"):
        text = text.lower()
    if profile == "nopunct":
        text = "".join(
            " " if unicodedata.category(ch)[0] in ("P", "S") else ch for ch in text
        )
    if profile == "nospace":
        return _ALLWS_RE.sub("", text.strip())
    return _WS_RE.sub(" ", text).strip()
