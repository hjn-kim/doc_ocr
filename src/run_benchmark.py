# -*- coding: utf-8 -*-
"""data/ 의 PDF 를 세 엔진으로 읽고 exact match / CER / WER 을 낸다.

    python src/run_benchmark.py --device gpu:0
    python src/run_benchmark.py --engines tesseract --device cpu --limit 3
    python src/run_benchmark.py --metrics-only --normalize nospace   # 재채점만

엔진을 바깥 루프에 둔다. GPU 에 모델을 하나씩만 올리려는 것이다.
인식 결과는 result/pred/<engine>/<문서>.txt 에 남으므로, 채점 기준을 바꿔
다시 재려고 OCR 을 또 돌릴 필요가 없다(--metrics-only).
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

# 윈도우 콘솔(cp949)에서 한글/기호 출력이 죽지 않게
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

import engines as engine_registry          # noqa: E402
import metrics as M                        # noqa: E402
import languages                           # noqa: E402
from dataset import load_samples           # noqa: E402
from normalize import normalize            # noqa: E402
from pdf_io import render_pages            # noqa: E402

# 엔진이 안 깔려 있을 때 뭘 깔아야 하는지 같이 알려준다.
INSTALL_HINT = {
    'paddleocr_vl': ('pip install paddlepaddle-gpu==3.2.1 -i '
                     'https://www.paddlepaddle.org.cn/packages/stable/cu126/ 뒤에 '
                     'pip install -U paddleocr[doc-parser]>=3.6.0'),
    'tesseract': ('pip install pytesseract 와 '
                  'apt-get install -y tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng'),
    'easyocr': 'pip install easyocr',
}

PAGE_SEP = "\n"
ROW_FIELDS = [
    "doc_id", "group", "engine", "device", "lang", "n_pages", "seconds", "sec_per_page",
    "gt_found", "gt_chars", "pred_chars", "gt_words", "pred_words",
    "exact_match", "cer", "wer", "char_errors", "word_errors",
    "normalize", "error", "pdf", "gt", "pred_path",
]
SUMMARY_FIELDS = [
    "engine", "group", "device", "n_docs", "n_scored", "n_pages", "exact_match",
    "cer_micro", "cer_macro", "wer_micro", "wer_macro",
    "total_seconds", "sec_per_page", "n_failed", "normalize",
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PDF OCR 벤치마크 (exact match / CER / WER)")
    p.add_argument("--data", type=Path, default=ROOT / "data", help="PDF 가 있는 폴더")
    p.add_argument("--gt-dir", type=Path, default=None,
                   help="정답 텍스트 폴더 (자동 탐색에 실패할 때만 지정)")
    p.add_argument("--out", type=Path, default=ROOT / "result" / "result.csv")
    p.add_argument("--pred-dir", type=Path, default=ROOT / "result" / "pred")
    p.add_argument("--engines", nargs="+",
                   default=["paddleocr_vl", "tesseract", "easyocr"])
    p.add_argument("--device", default="gpu:0",
                   help="gpu:0 또는 cpu. tesseract 는 항상 CPU 로 돈다")
    p.add_argument("--dpi", type=int, default=200, help="PDF -> 이미지 렌더 해상도")
    p.add_argument("--limit", type=int, default=None, help="앞에서 N개 문서만")
    p.add_argument("--normalize", default="basic",
                   choices=["none", "basic", "lower", "nospace", "nopunct"],
                   help="채점 전 정규화 (기본 basic: NFKC + 공백 축약)")
    p.add_argument("--keep-markdown", action="store_true",
                   help="채점할 때 markdown 기호를 지우지 않는다")
    p.add_argument("--resume", action="store_true", help="이미 저장된 예측은 건너뛴다")
    p.add_argument("--metrics-only", action="store_true",
                   help="OCR 없이 저장된 예측으로 점수만 다시 낸다")
    # 엔진별 옵션
    p.add_argument("--paddle-version", default="v1.6", help="PaddleOCR-VL 파이프라인 버전")
    p.add_argument("--paddle-input", default="pdf", choices=["pdf", "images"],
                   help="PDF 를 그대로 넘길지, 렌더한 이미지를 넘길지")
    p.add_argument("--paddle-text", default="blocks", choices=["blocks", "markdown"],
                   help="레이아웃 블록 텍스트와 markdown 중 무엇을 인식 결과로 볼지")
    p.add_argument("--paddle-engine", default=None,
                   choices=["paddle", "transformers"])
    p.add_argument("--paddle-backend", default=None,
                   help="원격 추론 백엔드. 예를 들면 vllm-server")
    p.add_argument("--paddle-server-url", default=None)
    p.add_argument("--tesseract-lang", default="kor+eng")
    p.add_argument("--tesseract-config", default="--oem 3 --psm 3")
    p.add_argument("--tesseract-cmd", default=None, help="tesseract 실행 파일 경로")
    p.add_argument("--easyocr-langs", nargs="+", default=["ko", "en"])
    p.add_argument("--easyocr-no-paragraph", action="store_true")
    p.add_argument("--lang-map", type=Path, default=None,
                   help="폴더 이름 -> 엔진 언어 설정을 덮어쓸 JSON "
                        "(기본표는 src/languages.py)")
    return p.parse_args(argv)


def engine_options(name: str, args) -> dict:
    if name == "paddleocr_vl":
        return dict(pipeline_version=args.paddle_version, input=args.paddle_input,
                    text_from=args.paddle_text, engine=args.paddle_engine,
                    backend=args.paddle_backend, server_url=args.paddle_server_url)
    if name == "tesseract":
        return dict(lang=args.tesseract_lang, config=args.tesseract_config,
                    cmd=args.tesseract_cmd)
    if name == "easyocr":
        return dict(langs=args.easyocr_langs, paragraph=not args.easyocr_no_paragraph)
    return {}


def free_gpu():
    """다음 엔진을 올리기 전에 VRAM 을 돌려준다. 실패해도 그냥 넘어간다."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        import paddle
        paddle.device.cuda.empty_cache()
    except Exception:
        pass


def score(gt_text, pred_text: str, profile: str, strip_md: bool) -> dict:
    """정규화한 뒤 세 지표를 낸다. 정답이 없으면 지표 칸을 비워둔다."""
    pred_n = normalize(pred_text, profile, strip_markdown=strip_md)
    if gt_text is None:
        return {"gt_found": 0, "pred_chars": len(pred_n),
                "pred_words": len(pred_n.split()), "_char": None, "_word": None}
    gt_n = normalize(gt_text, profile, strip_markdown=strip_md)
    ce = M.char_errors(gt_n, pred_n)
    we = M.word_errors(gt_n, pred_n)
    return {
        "gt_found": 1,
        "gt_chars": len(gt_n), "pred_chars": len(pred_n),
        "gt_words": len(gt_n.split()), "pred_words": len(pred_n.split()),
        "exact_match": M.exact_match(gt_n, pred_n),
        "cer": round(ce.rate, 6), "wer": round(we.rate, 6),
        "char_errors": ce.errors, "word_errors": we.errors,
        "_char": ce, "_word": we,
    }


def summarize(name: str, device: str, group: str, records: list[dict],
              profile: str) -> dict:
    """문서 기록들을 한 줄로 접는다. group="ALL" 이면 전체 집계다."""
    chars = [r["char"] for r in records if r["char"] is not None]
    words = [r["word"] for r in records if r["word"] is not None]
    exacts = [r["exact"] for r in records if r["exact"] is not None]
    seconds = sum(r["seconds"] or 0.0 for r in records)
    pages = sum(r["pages"] or 0 for r in records)
    return {
        "engine": name, "group": group, "device": device,
        "n_docs": len(records), "n_scored": len(chars), "n_pages": pages,
        "exact_match": round(sum(exacts) / len(exacts), 6) if exacts else "",
        "cer_micro": round(M.micro_average(chars), 6) if chars else "",
        "cer_macro": round(M.macro_average(chars), 6) if chars else "",
        "wer_micro": round(M.micro_average(words), 6) if words else "",
        "wer_macro": round(M.macro_average(words), 6) if words else "",
        "total_seconds": round(seconds, 2),
        "sec_per_page": round(seconds / pages, 3) if pages else "",
        "n_failed": sum(1 for r in records if r["error"]),
        "normalize": profile,
    }


def run_engine(name: str, samples, args, writer, csv_file) -> list[dict]:
    """엔진 하나로 전체 문서를 돌리고 요약 행(전체 + 그룹별)을 돌려준다."""
    pred_root = args.pred_dir / name
    pred_root.mkdir(parents=True, exist_ok=True)
    # 객체만 만들어 둔다. 무거운 import 와 모델 로딩은 첫 문서에서 일어난다.
    engine = engine_registry.build(name, device=args.device, **engine_options(name, args))
    device = args.device if engine.uses_gpu else "cpu"
    strip_md = not args.keep_markdown
    lang_table = languages.load_map(args.lang_map)

    records = []
    print(f"\n=== {name} (device={device}) - 문서 {len(samples)}개 ===", flush=True)
    if not args.metrics_only:
        # 여기서 미리 올린다. 모델 로딩 시간이 첫 문서의 OCR 시간에 섞이지 않고,
        # 패키지가 아예 없으면 문서마다 같은 에러를 뱉는 대신 엔진째로 건너뛴다.
        t0 = time.perf_counter()
        engine.ensure_loaded()
        print(f"  모델 로딩 {time.perf_counter() - t0:.1f}s", flush=True)

    for i, sample in enumerate(samples, 1):
        pred_path = pred_root / f"{sample.doc_id}.txt"
        row = {f: "" for f in ROW_FIELDS}
        lang = (languages.for_engine(lang_table, sample.group, name)
                if engine.lang_aware else None)
        row.update(doc_id=sample.doc_id, group=sample.group, engine=name, device=device,
                   lang=languages.describe(lang), normalize=args.normalize,
                   pdf=str(sample.pdf), gt=str(sample.gt_path or ""),
                   pred_path=str(pred_path))
        pred_text, seconds, n_pages, error = "", None, None, ""

        cached = pred_path.is_file() and (args.metrics_only or args.resume)
        try:
            if cached:
                pred_text = pred_path.read_text(encoding="utf-8")
            elif args.metrics_only:
                raise FileNotFoundError(f"저장된 예측이 없다: {pred_path}")
            else:
                needs_images = not (name == "paddleocr_vl"
                                    and args.paddle_input == "pdf")
                pages_img = render_pages(sample.pdf, args.dpi) if needs_images else []
                start = time.perf_counter()
                page_texts = engine.run(sample.pdf, pages_img, lang)
                seconds = time.perf_counter() - start
                n_pages = len(page_texts) or len(pages_img)
                pred_text = PAGE_SEP.join(page_texts)
                pred_path.write_text(pred_text, encoding="utf-8")
        except Exception as exc:                   # 한 문서가 죽어도 나머지는 돈다
            error = f"{type(exc).__name__}: {exc}"
            print(f"  [{i}/{len(samples)}] {sample.doc_id} 실패 - {error}", flush=True)
            traceback.print_exc(limit=3)

        s = score(sample.gt_text, pred_text, args.normalize, strip_md)
        row.update({k: v for k, v in s.items() if not k.startswith("_")})
        if seconds is not None:
            row["seconds"] = round(seconds, 3)
            if n_pages:
                row["sec_per_page"] = round(seconds / n_pages, 3)
        if n_pages:
            row["n_pages"] = n_pages
        row["error"] = error
        writer.writerow(row)
        csv_file.flush()                           # 중간에 끊겨도 여기까진 남는다

        scored = s["_char"] is not None and not error
        records.append({
            "group": sample.group, "error": error, "seconds": seconds,
            "pages": n_pages,
            "char": s["_char"] if scored else None,
            "word": s["_word"] if scored else None,
            "exact": s["exact_match"] if scored else None,
        })
        mark = "cached" if cached else (f"{seconds:.1f}s" if seconds else "-")
        if row["lang"]:
            mark = f"{row['lang']} {mark}"
        detail = ("정답 없음" if s["gt_found"] == 0 else
                  f"EM={s['exact_match']} CER={s['cer']:.4f} WER={s['wer']:.4f}")
        print(f"  [{i}/{len(samples)}] {sample.doc_id}  {mark}  {detail}", flush=True)

    del engine
    free_gpu()

    rows = [summarize(name, device, "ALL", records, args.normalize)]
    groups = sorted({r["group"] for r in records})
    if len(groups) > 1:                            # en/ko 처럼 나뉘어 있을 때만
        rows += [summarize(name, device, g, [r for r in records if r["group"] == g],
                           args.normalize) for g in groups]
    return rows


def print_summary(rows: list[dict]):
    if not rows:
        print("\n요약할 결과가 없다.")
        return
    head = ["engine", "group", "n_scored", "exact_match",
            "cer_micro", "wer_micro", "sec_per_page"]
    widths = [max([len(h)] + [len(str(r.get(h, ""))) for r in rows]) for h in head]
    line = "  ".join(h.ljust(w) for h, w in zip(head, widths))
    print("\n" + line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r.get(h, "")).ljust(w) for h, w in zip(head, widths)))


def main(argv=None):
    args = parse_args(argv)
    if not args.data.is_dir():
        raise SystemExit(f"데이터 폴더가 없다: {args.data}")
    samples = load_samples(args.data, args.gt_dir, args.limit)
    if not samples:
        raise SystemExit(f"{args.data} 에서 PDF 를 못 찾았다")
    n_gt = sum(1 for s in samples if s.gt_text is not None)
    print(f"PDF {len(samples)}개, 정답 {n_gt}개, 정규화={args.normalize}, "
          f"엔진={', '.join(args.engines)}")
    if args.normalize == "nospace":
        print("! nospace 는 공백을 다 지우므로 문서 전체가 한 단어가 된다. "
              "WER 은 무의미하니 CER 만 봐라.")
    if n_gt == 0:
        print("! 정답 텍스트를 못 찾았다. OCR 결과만 저장하고 지표는 비워둔다. "
              "(PDF 옆에 같은 이름의 .txt 를 두거나 --gt-dir 로 폴더를 알려줘라)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    # utf-8-sig: 엑셀에서 한글이 안 깨지게
    with args.out.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        writer.writeheader()
        for name in args.engines:
            key = engine_registry.resolve(name)
            try:
                summaries += run_engine(key, samples, args, writer, f)
            except Exception as exc:      # 엔진 자체를 못 올린 경우(미설치 등)
                print(f"!! {key} 엔진을 건너뛴다 - {type(exc).__name__}: {exc}",
                      flush=True)
                if isinstance(exc, ImportError):
                    print(f"   설치: {INSTALL_HINT.get(key, '')}", flush=True)
                traceback.print_exc(limit=3)

    summary_path = args.out.with_name(args.out.stem + "_summary.csv")
    with summary_path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(summaries)

    print_summary(summaries)
    print(f"\n문서별: {args.out}\n요약  : {summary_path}\n예측  : {args.pred_dir}")


if __name__ == "__main__":
    main()
