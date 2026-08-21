# -*- coding: utf-8 -*-
"""PDF 텍스트 레이어를 뽑아 정답 초안을 만든다.

    python src/make_gt.py --data data/ko --out data/gt/ko
    python src/make_gt.py --report            # 저장 없이 상태만 본다

data/ 의 PDF 는 대부분 스캔본이 아니라 텍스트가 심긴 전자문서다. 그 텍스트를
뽑으면 사람이 처음부터 타이핑하지 않아도 되는 정답 초안이 나온다. 다만 뽑은
순서와 글자를 그대로 믿으면 안 된다. 아래 다섯 가지를 손봐서 내보낸다.

1) 읽는 순서 - PDF 내부 순서는 화면에 보이는 순서와 다르다. 판결문은 맨 아래
   쪽번호("- 1 -")가 제일 먼저 나온다. 좌표를 보고 위에서 아래로 다시 세운다.
2) N-up - 한 장에 논리 페이지 4쪽이 2x2 로 얹힌 PDF 가 있다(2015노4675).
   쪽번호 마커가 여러 개인 걸로 알아내서 사분면 단위로 순서를 잡는다.
3) 표 - 세로로 병합된 칸('A사(1명)' 이 두 줄을 덮는 식) 때문에 좌표만으로 줄을
   세우면 칸 순서가 엉킨다. 표는 격자를 읽어 행 -> 열 순으로 풀어 쓴다.
4) 빠진 띄어쓰기 - 보도자료(한글 문서)는 양쪽 정렬을 자간으로 맞춰서, 눈에는
   띄어쓰기가 보이는데 텍스트 레이어에는 공백이 없는 자리가 있다("개시 세 달"
   -> "개시세달"). 글자 사이가 글꼴 크기의 30%보다 벌어지면 공백을 넣는다.
5) 깨진 글꼴 - 법원 열람 사이트에서 뽑은 파일은 상단 고지 문구가 ToUnicode 가
   망가진 서체로 들어가 '⸬d䑄ᷤ㉐⏈' 처럼 나온다. 오프셋이 일정해서 되돌린다.

그래도 초안은 초안이다. 유니코드 대응이 없는 글자(네모 번호 같은 것)는 U+FFFD
로 남기고 어느 파일에 몇 개인지 알려주니, 원본을 보고 사람이 채워라.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent

PAGEMARK = re.compile(r"^-\s*\d+\s*-$")
ROW_TOL = 4.0          # 이 정도 y 차이는 같은 줄로 본다 (pt)
SPACE_RATIO = 0.30     # 글꼴 크기 대비 이만큼 벌어지면 띄어쓰기로 본다
UNKNOWN = "�"     # 유니코드 대응이 없는 글자 자리

# 깨진 서체의 오프셋. 코드 포인트 구간마다 값이 다르다.
# 확인 방법: 같은 문구가 멀쩡히 들어간 다른 판결문(2025고합394)과 글자를 맞춰봤다.
BROKEN_OFFSETS = ((0x20, 0x7E, -68), (0xA0, 0xFF, 19), (0x1000, 0xA000, 36556))

# 글리프는 제대로 그려지는데 유니코드 대응이 어긋나 엉뚱하게 나오는 글자들.
# 원본 이미지를 보고 하나씩 확인했다.
# 글머리표 자리에 들어간 딩벳 글꼴(Wingdings/Symbol)의 사설 영역 코드.
# 글자가 아니라 그림이라 정답에서는 뺀다 (OCR 도 못 읽거나 기호로만 읽는다).
DINGBATS = {"": "", "": "", "": "", "": "", "": ""}

GLYPH_FIXES = {
    "\udcf9": "中",   # 260120 표 머리 "입찰 담합기간 中 평균 낙찰율"
    "­": "-",        # soft hyphen 으로 들어간 「형사-행정-민사」의 붙임표
    "ᆞ": "ㆍ",   # 251223 제목의 가운뎃점. 같은 문서 본문은 U+318D 를 쓴다
}


def decode_broken(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        for lo, hi, delta in BROKEN_OFFSETS:
            if lo <= code <= hi and 0x20 <= code + delta <= 0x10FFFF:
                out.append(chr(code + delta))
                break
        else:
            out.append(ch)
    return "".join(out)


def looks_broken(text: str) -> bool:
    """깨진 줄인지 판단한다.

    '○', '社', 'ㄱ' 같은 멀쩡한 글자도 오프셋을 더하면 한글이 되기 때문에,
    "되돌리면 한글이 된다"만으로는 멀쩡한 줄을 망가뜨린다. 진짜 깨진 줄에는
    한글 음절이 하나도 없다는 점을 먼저 본다.
    """
    if any("가" <= c <= "힣" for c in text):
        return False
    # 되돌리면 한글이 되지만 원래 멀쩡한 글자들이 있다. 베트남어 성조 글자
    # (U+1E00~U+1EFF)와 따옴표/붙임표(U+2000~U+206F, 우즈베크어 o'g' 포함)가
    # 그렇다. 후보에서 빼지 않으면 베트남어·우즈베크어 줄을 통째로 망가뜨린다.
    body = [c for c in text if not c.isspace()]
    cand = [c for c in body if 0x1000 <= ord(c) <= 0xA000
            and not (0x1E00 <= ord(c) <= 0x1EFF)
            and not (0x2000 <= ord(c) <= 0x206F)]
    # 진짜 깨진 줄은 글자 대부분이 깨져 있다. 라틴 문자가 멀쩡히 섞인 줄은
    # 다른 언어이지 깨진 줄이 아니다.
    if len(cand) < 4 or len(cand) < 0.4 * len(body):
        return False
    fixed = sum(1 for c in cand if "가" <= chr(ord(c) + 36556) <= "힣")
    return fixed / len(cand) > 0.6


def clean(text: str) -> str:
    """글리프 교정 + 대응 없는 글자 표시."""
    if looks_broken(text):
        text = decode_broken(text)
    out = []
    for ch in text:
        if ch in DINGBATS:
            out.append(DINGBATS[ch])
        elif ch in GLYPH_FIXES:
            out.append(GLYPH_FIXES[ch])
        elif 0xD800 <= ord(ch) <= 0xDFFF or 0xE000 <= ord(ch) <= 0xF8FF:
            out.append(UNKNOWN)          # 서로게이트/사설 영역 = 대응 없는 글자
        else:
            out.append(ch)
    return "".join(out)


def page_lines(page) -> list[list]:
    """[x0, y0, x1, y1, 텍스트] 목록. 글자 간격을 보고 빠진 공백을 넣는다."""
    lines = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            chars = [c for span in line["spans"] for c in span["chars"]]
            if not chars:
                continue
            size = max((span["size"] for span in line["spans"]), default=10) or 10
            buf, prev = [], None
            for c in chars:
                if prev is not None:
                    gap = c["bbox"][0] - prev["bbox"][2]
                    if (gap > SPACE_RATIO * size and c["c"] != " "
                            and buf and buf[-1] != " "):
                        buf.append(" ")
                buf.append(c["c"])
                prev = c
            text = clean("".join(buf)).strip()
            if text:
                lines.append(list(line["bbox"]) + [text])
    return lines


def in_box(line, box) -> bool:
    cx, cy = (line[0] + line[2]) / 2, (line[1] + line[3]) / 2
    return box[0] - 1 <= cx <= box[2] + 1 and box[1] - 1 <= cy <= box[3] + 1


def table_chunks(page, lines):
    """표를 행 단위 텍스트로 풀고, 표가 삼킨 줄은 목록에서 뺀다.

    row.cells 는 병합된 칸을 처음 나오는 행에만 넣어 주므로, 행 -> 열 순으로
    돌면 사람이 읽는 순서 그대로 나온다.
    """
    try:
        tables = page.find_tables().tables
    except Exception:
        return [], lines
    chunks, used = [], set()
    for table in tables:
        if table.row_count < 2 or table.col_count < 2:
            continue                       # 글상자 하나를 표로 오인한 경우
        rows = []
        for row in table.rows:
            cells = []
            for box in row.cells:
                if box is None:
                    continue
                inner = [ln for ln in lines if id(ln) not in used and in_box(ln, box)]
                inner.sort(key=lambda ln: (round(ln[1] / ROW_TOL), ln[0]))
                for ln in inner:
                    used.add(id(ln))
                text = " ".join(ln[4] for ln in inner).strip()
                if text:
                    cells.append(text)
            if cells:
                rows.append(" ".join(cells))
        if rows:
            chunks.append(list(table.bbox) + [chr(10).join(rows)])
    return chunks, [ln for ln in lines if id(ln) not in used]


def split_regions(items):
    """N-up 이면 (x경계, y경계)를, 아니면 None 을 돌려준다."""
    marks = [it for it in items if PAGEMARK.match(it[4].strip())]
    if len(marks) < 2:
        return None

    def bounds(values):
        vals = []
        for v in sorted(values):
            if not vals or v - vals[-1] > 20:      # 같은 행/열의 마커는 묶는다
                vals.append(v)
        return [(a + b) / 2 for a, b in zip(vals, vals[1:])]

    return bounds(m[0] for m in marks), bounds(m[1] for m in marks)


def page_text(page) -> str:
    lines = page_lines(page)
    if not lines:
        return ""
    tables, lines = table_chunks(page, lines)
    items = lines + tables

    cuts = split_regions(items)
    if cuts is None:
        groups = {(0, 0): items}
    else:
        xcuts, ycuts = cuts
        groups = {}
        for it in items:
            key = (sum(1 for c in ycuts if it[1] > c),
                   sum(1 for c in xcuts if it[0] > c))
            groups.setdefault(key, []).append(it)

    chunks = []
    for key in sorted(groups):                     # 위 행부터, 각 행은 왼쪽부터
        group = groups[key]
        group.sort(key=lambda it: (round(it[1] / ROW_TOL), it[0]))
        marks = [it for it in group if PAGEMARK.match(it[4].strip())]
        body = [it for it in group if it not in marks]
        chunks.append(chr(10).join(it[4] for it in body + marks))
    return chr(10).join(chunks).strip()


def extract_text(pdf: Path) -> tuple[str, int]:
    """(텍스트, 쪽 수). PyMuPDF 를 먼저 쓰고 없으면 pypdfium2 로 넘어간다."""
    try:
        import fitz  # PyMuPDF

        with fitz.open(str(pdf)) as doc:
            pages = [page_text(page) for page in doc]
        return chr(10).join(p for p in pages if p).strip(), len(pages)
    except ImportError:
        pass

    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf))
    try:
        pages = [doc[i].get_textpage().get_text_range().strip() for i in range(len(doc))]
        return chr(10).join(pages).strip(), len(pages)
    finally:
        doc.close()


def main(argv=None):
    p = argparse.ArgumentParser(description="PDF 텍스트 레이어 -> 정답 초안")
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "gt",
                   help="초안을 저장할 폴더. --data 하위 구조를 그대로 미러링한다")
    p.add_argument("--overwrite", action="store_true", help="이미 있는 파일도 덮어쓴다")
    p.add_argument("--report", action="store_true", help="저장하지 않고 목록만 본다")
    args = p.parse_args(argv)

    data = args.data.resolve()
    pdfs = sorted(x for x in data.rglob("*.pdf") if x.is_file())
    if not pdfs:
        raise SystemExit(f"{data} 에서 PDF 를 못 찾았다")

    empty, unknown = [], []
    for pdf in pdfs:
        rel = pdf.relative_to(data)
        text, n_pages = extract_text(pdf)
        target = (args.out / rel).with_suffix(".txt")
        state = f"{n_pages:3}쪽 {len(text):7}자"
        if UNKNOWN in text:
            unknown.append((rel, text.count(UNKNOWN)))
            state += f"  <- 대응 없는 글자 {text.count(UNKNOWN)}개"
        if not text:
            empty.append(rel)
            state += "  <- 텍스트 레이어 없음(스캔본). 정답을 직접 만들어야 한다"
        elif not args.report:
            if target.exists() and not args.overwrite:
                state += "  <- 이미 있어서 건너뜀 (--overwrite)"
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
        print(f"{state}  {rel}")

    if not args.report:
        print(f"\n초안 저장 위치: {args.out}")
        print("도장·로고 속 글자와 아래 표시된 자리는 사람이 원본을 보고 채워야 한다.")
    if unknown:
        print(f"\n대응 없는 글자가 남은 파일 {len(unknown)}개 (U+FFFD 로 표시):")
        for rel, n in unknown:
            print(f"  - {rel}  {n}개")
    if empty:
        print(f"\n텍스트 레이어가 없는 파일 {len(empty)}개:")
        for rel in empty:
            print(f"  - {rel}")


if __name__ == "__main__":
    main()
