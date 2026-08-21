# -*- coding: utf-8 -*-
"""Exact match / CER / WER.

외부 의존성 없이 레벤슈타인 거리로 직접 계산한다. jiwer 등을 쓰지 않는 이유는
GPU 서버에서 pip 하나라도 덜 깔리게 하려는 것뿐이고, 정의는 동일하다.
    CER = (S + D + I) / 정답 글자 수
    WER = (S + D + I) / 정답 단어 수
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def levenshtein(ref: Sequence, hyp: Sequence) -> int:
    """편집거리. 두 줄만 들고 있어서 긴 문서도 메모리를 안 먹는다."""
    if ref == hyp:
        return 0
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    # 짧은 쪽을 열로 두면 행 버퍼가 작아진다.
    if len(hyp) < len(ref):
        ref, hyp = hyp, ref
    prev = list(range(len(hyp) + 1))
    cur = [0] * (len(hyp) + 1)
    for i, r in enumerate(ref, 1):
        cur[0] = i
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1,            # 삭제
                         cur[j - 1] + 1,          # 삽입
                         prev[j - 1] + (r != h))  # 치환
        prev, cur = cur, prev
    return prev[len(hyp)]


@dataclass
class ErrorRate:
    """비율뿐 아니라 분자/분모도 들고 있는다. 코퍼스 전체 micro 평균을 내려면
    문서별 비율이 아니라 편집 횟수와 정답 길이의 합이 필요하다."""
    errors: int
    length: int

    @property
    def rate(self) -> float:
        if self.length == 0:
            # 정답이 비었는데 예측이 있으면 전부 오류(1.0), 둘 다 비었으면 0.0
            return 0.0 if self.errors == 0 else 1.0
        return self.errors / self.length


def char_errors(ref: str, hyp: str) -> ErrorRate:
    return ErrorRate(levenshtein(list(ref), list(hyp)), len(ref))


def word_errors(ref: str, hyp: str) -> ErrorRate:
    r, h = ref.split(), hyp.split()
    return ErrorRate(levenshtein(r, h), len(r))


def cer(ref: str, hyp: str) -> float:
    return char_errors(ref, hyp).rate


def wer(ref: str, hyp: str) -> float:
    return word_errors(ref, hyp).rate


def exact_match(ref: str, hyp: str) -> int:
    """문서 전체가 글자 하나까지 같으면 1. 정규화를 거친 문자열끼리 비교한다."""
    return int(ref == hyp)


def micro_average(rates: list[ErrorRate]) -> float:
    """편집 횟수 합 / 정답 길이 합. 긴 문서에 가중치가 실린다."""
    errors = sum(r.errors for r in rates)
    length = sum(r.length for r in rates)
    return ErrorRate(errors, length).rate


def macro_average(rates: list[ErrorRate]) -> float:
    """문서별 비율의 단순 평균. 문서 하나를 한 표로 센다."""
    if not rates:
        return 0.0
    return sum(r.rate for r in rates) / len(rates)
