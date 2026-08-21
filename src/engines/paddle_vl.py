# -*- coding: utf-8 -*-
"""PaddleOCR-VL 1.6.

    pip install -U "paddleocr[doc-parser]>=3.6.0"
    (paddlepaddle-gpu 3.2.1 이상이 먼저 깔려 있어야 한다)

PDF 경로를 그대로 넘기면 페이지별 결과가 나온다. 나머지 두 엔진과 같은 픽셀을
쓰려고 렌더링해둔 이미지를 넘기는 것도 가능하게 해뒀다(--paddle-input images).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .base import OCREngine


def _result_dict(res) -> dict:
    """PaddleOCR 버전에 따라 res.json 이 {'res': {...}} 로 한 겹 싸여 나온다."""
    data = getattr(res, "json", None)
    if isinstance(data, dict):
        return data.get("res", data)
    return {}


def _blocks_to_text(data: dict) -> str:
    blocks = data.get("parsing_res_list") or []
    out = []
    for b in blocks:
        if isinstance(b, dict):
            content = b.get("block_content") or b.get("block_text") or ""
        else:  # 일부 버전은 객체로 준다
            content = getattr(b, "block_content", "") or ""
        content = str(content).strip()
        if content:
            out.append(content)
    return "\n".join(out)


def _markdown_to_text(res) -> str:
    md = getattr(res, "markdown", None)
    if isinstance(md, dict):
        text = md.get("markdown_texts", "")
    else:
        text = md or ""
    if isinstance(text, list):
        text = "\n".join(str(t) for t in text)
    return str(text).strip()


class PaddleVLEngine(OCREngine):
    name = "paddleocr_vl"

    def load(self):
        from paddleocr import PaddleOCRVL

        kwargs = dict(
            pipeline_version=self.options.get("pipeline_version", "v1.6"),
            device=self.device,
        )
        engine = self.options.get("engine")  # None | "paddle" | "transformers"
        if engine:
            kwargs["engine"] = engine
        backend = self.options.get("backend")  # 예: "vllm-server"
        if backend:
            kwargs["vl_rec_backend"] = backend
            if self.options.get("server_url"):
                kwargs["vl_rec_server_url"] = self.options["server_url"]
        return PaddleOCRVL(**kwargs)

    def run(self, pdf: Path, pages: list[np.ndarray], lang=None) -> list[str]:
        # 다국어 모델 하나가 전부 처리하므로 lang 은 받기만 하고 안 쓴다.
        pipeline = self.ensure_loaded()
        mode = self.options.get("text_from", "blocks")  # blocks | markdown
        source = self.options.get("input", "pdf")       # pdf | images

        inputs = [str(pdf)] if source == "pdf" else pages
        collected: list[tuple[int, str]] = []
        for order, item in enumerate(inputs):
            for res in pipeline.predict(item):
                data = _result_dict(res)
                text = (_markdown_to_text(res) if mode == "markdown"
                        else _blocks_to_text(data))
                if mode == "blocks" and not text:
                    text = _markdown_to_text(res)  # 레이아웃 결과가 비면 markdown 으로 대체
                page_index = data.get("page_index")
                collected.append((order if page_index is None else page_index, text))
        collected.sort(key=lambda x: x[0])
        return [t for _, t in collected]
