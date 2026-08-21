# -*- coding: utf-8 -*-
"""data/ 안의 PDF와 정답 텍스트를 짝지어 준다.

data/en, data/ko 처럼 하위 폴더로 나뉜 구조를 그대로 읽는다. 첫 번째 하위
폴더 이름을 group 으로 잡아 두면 result.csv 에서 언어별로 갈라 볼 수 있다.

정답 파일 위치는 아직 안 정해졌으니 흔한 배치를 모두 훑는다.
data/ko/사건.pdf 의 정답이라면 아래를 순서대로 찾고 처음 걸리는 것을 쓴다.

    data/ko/사건.txt                     (PDF 옆)
    <--gt-dir>/ko/사건.txt, <--gt-dir>/사건.txt
    data/gt/ko/사건.txt, data/gt/사건.txt
    data/ground_truth|label|labels|txt|text/... (같은 규칙)

확장자는 .txt, .gt.txt, .md, .json 을 본다. .json 은 {"text": "..."} 또는
{"pages": [...]} 형태를 읽는다. 정답이 없으면 OCR 만 돌리고 지표 칸은 비워둔다.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

GT_SUBDIRS = ("gt", "ground_truth", "groundtruth", "label", "labels", "txt", "text")
GT_SUFFIXES = (".txt", ".gt.txt", ".md", ".json")


@dataclass
class Sample:
    pdf: Path
    rel: Path              # data_dir 기준 상대 경로
    group: str             # 첫 하위 폴더 이름 (없으면 "root")
    gt_path: Path | None
    gt_text: str | None

    @property
    def doc_id(self) -> str:
        """ko/사건.pdf -> "ko__사건". 폴더가 달라도 이름이 겹치지 않게 한다."""
        return "__".join(self.rel.with_suffix("").parts)


def _read_gt(path: Path) -> str:
    raw = path.read_text(encoding="utf-8-sig", errors="replace")
    if path.suffix.lower() != ".json":
        return raw
    obj = json.loads(raw)
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        if isinstance(obj.get("text"), str):
            return obj["text"]
        for key in ("pages", "texts", "lines"):
            if isinstance(obj.get(key), list):
                return "\n".join(str(x) for x in obj[key])
    if isinstance(obj, list):
        return "\n".join(str(x) for x in obj)
    raise ValueError(f"정답 JSON 형식을 모르겠다: {path}")


def find_gt(pdf: Path, rel: Path, data_dir: Path, gt_dir: Path | None = None) -> Path | None:
    sub = rel.parent                      # 예: Path("ko")
    roots: list[Path] = [pdf.parent]      # PDF 바로 옆
    for base in ([gt_dir] if gt_dir else []) + [data_dir / d for d in GT_SUBDIRS]:
        roots.append(base / sub)          # 하위 폴더 구조를 그대로 미러링한 경우
        roots.append(base)                # 한 폴더에 몰아넣은 경우
    roots.append(data_dir)

    seen = set()
    for root in roots:
        if root is None or root in seen:
            continue
        seen.add(root)
        if not root.is_dir():
            continue
        for suffix in GT_SUFFIXES:
            cand = root / (pdf.stem + suffix)
            if cand.is_file() and cand.resolve() != pdf.resolve():
                return cand
    return None


def load_samples(data_dir: Path, gt_dir: Path | None = None,
                 limit: int | None = None) -> list[Sample]:
    data_dir = data_dir.resolve()
    pdfs = sorted(p for p in data_dir.rglob("*.pdf") if p.is_file())
    if limit:
        pdfs = pdfs[:limit]
    samples = []
    for pdf in pdfs:
        rel = pdf.relative_to(data_dir)
        gt_path = find_gt(pdf, rel, data_dir, gt_dir)
        samples.append(Sample(
            pdf=pdf, rel=rel,
            group=rel.parts[0] if len(rel.parts) > 1 else "root",
            gt_path=gt_path,
            gt_text=_read_gt(gt_path) if gt_path else None,
        ))
    return samples
