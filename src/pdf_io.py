# -*- coding: utf-8 -*-
"""PDF 를 페이지 이미지로 편다.

tesseract 와 easyocr 은 이미지만 받는다. PaddleOCR-VL 은 PDF 를 직접 받지만,
세 엔진에 같은 픽셀을 먹여야 렌더링 차이가 지표에 섞이지 않으므로 여기서 뽑은
이미지를 공통으로 쓴다.

렌더러는 pypdfium2 를 기본으로 쓴다 (pip 만으로 설치되고 poppler 가 필요 없다).
없으면 PyMuPDF, pdf2image 순으로 넘어간다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def _render_pypdfium2(pdf: Path, dpi: int) -> list[np.ndarray]:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf))
    try:
        pages = []
        for i in range(len(doc)):
            page = doc[i]
            bitmap = page.render(scale=dpi / 72)
            pages.append(np.asarray(bitmap.to_pil().convert("RGB")))
        return pages
    finally:
        doc.close()


def _render_pymupdf(pdf: Path, dpi: int) -> list[np.ndarray]:
    import fitz  # PyMuPDF

    pages = []
    with fitz.open(str(pdf)) as doc:
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = np.frombuffer(pix.samples, dtype=np.uint8)
            img = img.reshape(pix.height, pix.width, pix.n)
            pages.append(np.ascontiguousarray(img[:, :, :3]))
    return pages


def _render_pdf2image(pdf: Path, dpi: int) -> list[np.ndarray]:
    from pdf2image import convert_from_path

    return [np.asarray(im.convert("RGB")) for im in convert_from_path(str(pdf), dpi=dpi)]


_RENDERERS = (_render_pypdfium2, _render_pymupdf, _render_pdf2image)


def render_pages(pdf: Path, dpi: int = 200) -> list[np.ndarray]:
    """PDF 각 페이지를 RGB ndarray 로 돌려준다."""
    errors = []
    for fn in _RENDERERS:
        try:
            return fn(pdf, dpi)
        except ImportError as exc:
            errors.append(f"{fn.__name__}: {exc}")
    raise RuntimeError(
        "PDF 렌더러가 없다. pip install pypdfium2 (권장) 하거나 PyMuPDF/pdf2image 를 "
        "설치해라.\n" + "\n".join(errors))
