# -*- coding: utf-8 -*-
"""result/ 의 OCR 벤치마크 결과를 엔진별로 비교하는 대시보드.

    pip install streamlit altair
    streamlit run app.py

로컬에서 띄운다. GPU 서버에는 streamlit 을 깔지 않는다 — 거기서는 벤치마크만
돌리고 result/ 를 받아와서 여기서 본다.

result/result2.csv (문서 x 엔진 한 줄) 와 result/result2_summary.csv (엔진 x 언어
집계) 를 읽는다. 먼저 `python src/run_benchmark.py` 를 돌려 두 파일을 만들어야 한다.

CER 과 WER 은 **낮을수록 좋다**. 이 앱의 모든 정렬과 순위는 그 방향을 따른다.
"""

from __future__ import annotations

import difflib
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# 디자인(공통 CSS)은 여기서 정의하지 않고 shared_utils 에서 받아온다.
# 그 파일이 없거나 의존 패키지가 빠져도 앱은 떠야 하므로 실패는 무시한다.
try:
    from shared_utils import apply_common_styles
except Exception:                                   # noqa: BLE001
    def apply_common_styles():                      # 디자인만 빠지고 나머지는 그대로 동작
        pass

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))   # 채점에 쓴 정규화를 그대로 재사용한다

st.set_page_config(page_title="OCR 엔진 비교", page_icon="🔍", layout="wide")
apply_common_styles()          # 폰트·여백·탭 스타일 — shared_utils 가 갖고 있다


# ── 색 ────────────────────────────────────────────────────────
# dataviz 레퍼런스 팔레트의 계열 1~3 (파랑·주황·아쿠아). 엔진이 셋이라 딱 맞고,
# 이 세 칸은 light/dark 양쪽에서 all-pairs 검증을 통과한다
# (CVD ΔE 9.2/9.4, 일반시야 24.0/20.9). 네 번째 엔진이 생기면 그때 다시 검증해라.
# light 의 아쿠아는 표면 대비가 3:1 아래라 값 라벨과 표를 항상 같이 낸다.
def app_theme_is_dark() -> bool:
    """앱이 지금 어두운 테마로 그려지고 있는지. 브라우저 Settings 변경을 실시간 반영."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return st.get_option("theme.base") == "dark"


# 히트맵용 순차 램프. 파랑 한 가지를 밝기만 바꿔 쓴다(무지개 금지).
# "0 에 가까운 값이 표면 쪽으로 물러난다"는 규칙 때문에 어두운 테마는 순서를 뒤집는다.
SEQ_LIGHT = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
SEQ_DARK = ["#0d366b", "#184f95", "#2a78d6", "#5598e7", "#9ec5f4"]


def palette(dark: bool) -> dict:
    return {
        "series": (["#3987e5", "#d95926", "#199e70"] if dark
                   else ["#2a78d6", "#eb6834", "#1baf7a"]),
        "surface": "#1a1a19" if dark else "#fcfcfb",
        "grid": "#2c2c2a" if dark else "#e1e0d9",
        "muted": "#898781",                             # 축·라벨 (양쪽 공용)
        "seq": SEQ_DARK if dark else SEQ_LIGHT,
        # 진한 셀 위 / 옅은 셀 위 글자색. 어느 쪽 테마든 셀 대비를 확보한다.
        "on_strong": "#f3f2ef",
        "on_weak": "#2c2c2a",
    }


# 상태 색은 고정 — 계열 색으로 재사용하지 않는다. 항상 숫자/기호와 함께 쓴다.
TINT_GOOD = "rgba(12,163,12,0.20)"        # good     — 그 문서 1위
TINT_WARN = "rgba(250,178,25,0.20)"       # warning  — 오인식
TINT_SERIOUS = "rgba(236,131,90,0.20)"    # serious  — 없는 글자를 지어냄
TINT_CRIT = "rgba(208,59,59,0.20)"        # critical — 글자를 통째로 놓침

ENGINE_LABEL = {
    "paddleocr_vl": "PaddleOCR-VL 1.6",
    "tesseract": "Tesseract",
    "easyocr": "EasyOCR",
}

LANG_LABEL = {
    "ko": "한국어",
    "en": "영어",
    "ch": "중국어",
    "ru": "러시아어",
    "vn": "베트남어",
    "uz": "우즈베크어",
    "pil": "타갈로그어",
}

# 띄어쓰기가 없는 언어는 공백으로 자른 WER 이 뜻을 잃는다. 문장 전체가 한 단어로
# 잡혀서 한 글자만 틀려도 그 단어가 통째로 오답이 되고, WER 이 1 을 넘긴다.
NO_SPACE_LANGS = {"ch"}


def themed(chart, p: dict):
    """차트에 배경·여백·축 스타일을 입힌다.

    배경과 여백을 .configure() 로 주면 안 된다 — Altair 의 .configure() 는 config 를
    통째로 갈아끼워서 앞서 부른 .configure_axis() 설정이 조용히 사라진다.
    둘 다 Vega-Lite 최상위 속성이므로 .properties() 로 준다.
    """
    return (
        chart.properties(
            background=p["surface"],
            padding={"left": 12, "right": 12, "top": 12, "bottom": 12},
        )
        .configure_axis(
            labelColor=p["muted"], titleColor=p["muted"], tickColor=p["grid"],
            domainColor=p["grid"], gridColor=p["grid"], gridDash=[], grid=True,
        )
        .configure_legend(labelColor=p["muted"], titleColor=p["muted"])
        .configure_view(strokeWidth=0)
    )


# ── 데이터 로드 ───────────────────────────────────────────────
RESULT_FILES = ("result2.csv", "result2_summary.csv")


def results_stamp(d: Path) -> float:
    """결과 파일들의 최신 수정 시각. load_results 의 캐시 키로 쓴다.

    폴더 mtime 을 쓰면 안 된다 — 파일이 추가·삭제될 때만 바뀌고 같은 이름으로
    덮어쓰면 그대로라, 벤치마크를 다시 돌려도 옛 결과가 계속 보인다.
    """
    times = [(d / f).stat().st_mtime for f in RESULT_FILES if (d / f).exists()]
    return max(times) if times else 0.0


@st.cache_data(show_spinner=False)
def load_results(result_dir: str, stamp: float):
    """stamp 는 캐시 키 용도로만 받는다 (본문에서 쓰지 않는 게 정상)."""
    d = Path(result_dir)
    # run_benchmark 가 BOM 을 붙여 쓰므로 utf-8-sig 로 읽는다
    detail = pd.read_csv(d / "result2.csv", encoding="utf-8-sig")
    summary = pd.read_csv(d / "result2_summary.csv", encoding="utf-8-sig")

    detail["엔진"] = detail["engine"].map(lambda e: ENGINE_LABEL.get(e, e))
    detail["언어"] = detail["group"].map(lambda g: LANG_LABEL.get(g, g))
    detail["문서"] = detail["doc_id"].str.split("__", n=1).str[-1]
    summary["엔진"] = summary["engine"].map(lambda e: ENGINE_LABEL.get(e, e))
    return detail, summary


def rollup(df: pd.DataFrame, keys: list[str], mode: str) -> pd.DataFrame:
    """문서 단위 행을 묶어 CER/WER 을 낸다.

    micro = 편집 횟수 합 / 정답 길이 합 → 긴 문서에 가중치가 실린다. 엔진 비교용.
    macro = 문서별 비율의 평균       → 문서 하나를 한 표로 센다.
    두 값이 크게 갈리면 "긴 문서에서만 잘한다"는 뜻이라 둘 다 볼 수 있게 뒀다.
    """
    g = df.groupby(keys, dropna=False).agg(
        문서수=("doc_id", "nunique"),
        글자수=("gt_chars", "sum"),
        글자오류=("char_errors", "sum"),
        단어수=("gt_words", "sum"),
        단어오류=("word_errors", "sum"),
        CER_macro=("cer", "mean"),
        WER_macro=("wer", "mean"),
        일치율=("exact_match", "mean"),
    ).reset_index()
    g["CER_micro"] = g["글자오류"] / g["글자수"]
    g["WER_micro"] = g["단어오류"] / g["단어수"]
    g["CER"] = g[f"CER_{mode}"]
    g["WER"] = g[f"WER_{mode}"]
    return g


@st.cache_data(show_spinner=False)
def read_text(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        return ""
    return p.read_text(encoding="utf-8-sig", errors="replace")


def split_doc_id(doc_id: str) -> tuple[str, str]:
    """doc_id -> (언어 폴더, 파일 이름).

    dataset.py 가 "ko/사건.pdf" 를 "ko__사건" 으로 만든다. 하위 폴더 없이 data/ 바로
    아래 둔 PDF 는 "__" 가 없고 group 이 "root" 다.
    """
    group, sep, stem = doc_id.partition("__")
    return (group, stem) if sep else ("root", doc_id)


def local_paths(doc_id: str, engine: str) -> tuple[Path, Path]:
    """(정답, 인식결과) 로컬 경로.

    CSV 에 박힌 pdf/gt/pred_path 는 벤치마크를 돌린 GPU 서버의 절대 경로라 여기서는
    못 쓴다. doc_id 로 다시 만든다.
    """
    group, stem = split_doc_id(doc_id)
    sub = "" if group == "root" else group
    gt = ROOT / "data" / "gt" / sub / f"{stem}.txt"
    if not gt.is_file():
        gt = ROOT / "data" / sub / f"{stem}.txt"
    pred = ROOT / "result" / "pred" / engine / f"{doc_id}.txt"
    return gt, pred


DIFF_LIMIT = 24000   # 이 이상은 SequenceMatcher 가 눈에 띄게 느려진다


@st.cache_data(show_spinner=False, max_entries=64)
def diff_html(ref: str, hyp: str) -> str:
    """인식 결과를 정답과 맞춰 칠한다. 정답에 있는데 빠진 자리는 ‸ 로 표시.

    긴 문서는 한 번 그리는 데 몇 초 걸리므로 캐시에 얹는다. 문서를 왔다 갔다 해도
    다시 계산하지 않는다.
    """
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
            None, ref, hyp, autojunk=False).get_opcodes():
        piece = hyp[j1:j2].replace("&", "&amp;").replace("<", "&lt;")
        if tag == "equal":
            out.append(piece)
        elif tag == "replace":
            out.append(f'<mark style="background:{TINT_WARN}">{piece}</mark>')
        elif tag == "insert":
            out.append(f'<mark style="background:{TINT_SERIOUS}">{piece}</mark>')
        else:   # delete — 인식 결과에는 없는 자리라 표시만 남긴다
            out.append(f'<mark style="background:{TINT_CRIT}" '
                       f'title="누락 {i2 - i1}자">‸</mark>')
    return "".join(out)


def doc_label(doc_id: str) -> str:
    group, stem = split_doc_id(doc_id)
    return f"{LANG_LABEL.get(group, group)} · {stem}"


# ── 사이드바 ──────────────────────────────────────────────────
st.sidebar.header("설정")
result_dir = st.sidebar.text_input("result 폴더", value=str(ROOT / "result"))

rdir = Path(result_dir)
if not (rdir / "result2.csv").exists():
    st.title("OCR 엔진 비교")
    st.warning(f"`{rdir / 'result2.csv'}` 가 없습니다.")
    st.code("python src/run_benchmark.py --device gpu:0", language="bash")
    st.stop()

detail_all, summary_all = load_results(str(rdir), results_stamp(rdir))

# 채점된 행만 본다. 정답이 없거나 엔진이 죽은 문서는 지표 칸이 비어 있다.
scored = detail_all[(detail_all["gt_found"] == 1) & detail_all["cer"].notna()].copy()
failed = detail_all[detail_all["error"].notna()]

# 엔진 순서는 전체 CER(micro) 오름차순 — 잘하는 엔진이 위로 온다.
_rank = rollup(scored, ["엔진"], "micro").sort_values("CER_micro")
all_engines = _rank["엔진"].tolist()

picked = st.sidebar.multiselect("비교할 엔진", all_engines, default=all_engines)
if not picked:
    st.warning("엔진을 하나 이상 선택하세요.")
    st.stop()
order = [e for e in all_engines if e in picked]
scored = scored[scored["엔진"].isin(picked)].copy()

all_langs = [g for g in LANG_LABEL if g in set(scored["group"])]
lang_pick = st.sidebar.multiselect(
    "언어", all_langs, default=all_langs,
    format_func=lambda g: f"{LANG_LABEL[g]} ({g})",
)
if not lang_pick:
    st.warning("언어를 하나 이상 선택하세요.")
    st.stop()
scored = scored[scored["group"].isin(lang_pick)].copy()
lang_order = [g for g in all_langs if g in lang_pick]
label_order = [LANG_LABEL[g] for g in lang_order]

st.sidebar.divider()
mode = st.sidebar.radio(
    "평균 방식", ["micro", "macro"], horizontal=True,
    help="micro = 편집 횟수 합 / 정답 길이 합 (긴 문서에 가중치). 엔진 비교용은 이쪽. "
         "macro = 문서별 비율의 평균 (문서 하나가 한 표).",
)
theme_choice = st.sidebar.radio(
    "차트 테마", ["자동", "밝게", "어둡게"], horizontal=True,
    help="자동은 앱 테마(우상단 ⋮ → Settings)를 따라갑니다. "
         "밝게/어둡게는 앱과 무관하게 차트만 고정합니다 — 보고서 스크린샷용.",
)
dark = app_theme_is_dark() if theme_choice == "자동" else (theme_choice == "어둡게")
p = palette(dark)
series_scale = alt.Scale(domain=order, range=p["series"][: len(order)])

if st.sidebar.button("결과 다시 읽기"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
norm_used = ", ".join(sorted(scored["normalize"].dropna().unique())) or "?"
st.sidebar.caption(
    f"문서 {scored['doc_id'].nunique()}개 · 언어 {len(lang_order)}개 · "
    f"엔진 {len(order)}개  \n정규화 `{norm_used}`"
)
if len(failed):
    st.sidebar.warning(f"실패한 문서 {len(failed)}건 — 집계에서 빠졌습니다.")


# ── 헤더 ──────────────────────────────────────────────────────
st.title("OCR 엔진 비교")

overall = rollup(scored, ["엔진"], mode).set_index("엔진").reindex(order)
best = overall["CER"].idxmin()
has_timing = scored["seconds"].fillna(0).sum() > 0

c1, c2, c3 = st.columns(3)
c1.metric("1위 엔진", best, help="선택한 언어 전체에서 CER 이 가장 낮은 엔진")
c2.metric("CER", f"{overall.loc[best, 'CER']:.3f}",
          help="글자 단위 오류율. (치환+삭제+삽입) / 정답 글자 수. 낮을수록 좋다")
c3.metric("WER", f"{overall.loc[best, 'WER']:.3f}",
          help="공백으로 자른 단어 단위 오류율. 낮을수록 좋다")


tab_overview, tab_lang, tab_data, tab_text = st.tabs(
    ["개요", "언어별", "데이터", "인식 결과"]
)


# ── 1. 개요 ───────────────────────────────────────────────────
with tab_overview:
    st.subheader("엔진별 오류율")

    melted = (
        overall.reset_index()
        .melt(id_vars="엔진", value_vars=["CER", "WER"], var_name="지표", value_name="값")
    )
    bars = (
        alt.Chart(melted)
        .mark_bar(cornerRadiusEnd=4, size=32)   # 데이터 끝만 둥글게
        .encode(
            y=alt.Y("엔진:N", sort=order, title=None),
            x=alt.X("값:Q", title=None, axis=alt.Axis(format=".2f"),
                    scale=alt.Scale(nice=False, padding=6)),
            # 색은 y축과 같은 정보라 범례가 없어도 된다 (행 이름이 곧 엔진이다)
            color=alt.Color("엔진:N", legend=None, scale=series_scale),
            tooltip=[alt.Tooltip("엔진:N"), alt.Tooltip("지표:N"),
                     alt.Tooltip("값:Q", format=".3f")],
        )
    )
    # 막대 끝에 값 — 축을 읽지 않아도 되게. 텍스트는 계열 색이 아닌 muted 잉크.
    labels = bars.mark_text(align="left", dx=6, fontSize=13).encode(
        text=alt.Text("값:Q", format=".3f"), color=alt.value(p["muted"])
    )
    st.altair_chart(
        themed(
            (bars + labels)
            .properties(height=64 * len(order) + 40, width=350)
            .facet(column=alt.Column("지표:N", title=None,
                                     header=alt.Header(labelColor=p["muted"],
                                                       labelFontSize=13)))
            .resolve_scale(x="independent"),
            p,
        ),
        width="stretch",
    )


    st.subheader("지표")

    no_space = "·".join(LANG_LABEL.get(g, g) for g in lang_order if g in NO_SPACE_LANGS)
    # 문장 끝의 공백 두 칸 + 개행은 마크다운의 강제 줄바꿈이다. Streamlit 은 표 셀도
    # 마크다운으로 그리므로 CSS 와 무관하게 여기서 줄이 바뀐다.
    reason = (
        "OCR-D 평가 규격이 문자 단위 오류율(CER)을 OCR 품질의 1차 지표로 규정하고 있으며, "
        "문서 파싱 벤치마크 전반에서 텍스트 인식 정확도의 기본 척도로 사용됨. "
        "문자·숫자·기호가 혼용된 문서의 인식 정확도를 단일 수치로 비교할 수 있어 "
        "OCR 품질의 대표지표로 선정." + "\n"
        + "또한 WER 의 경우 띄어쓰기가 없는 중국어에서 평가 지표로 사용 불가."
    )

    st.table(
        pd.DataFrame(
            [
                ("CER", "(치환 + 삭제 + 삽입) / 정답 글자 수"),
                ("WER", "(치환 + 삭제 + 삽입) / 정답 단어 수"),
                ("OCR 대표 지표", "CER"),
                ("이유", reason),
            ],
            columns=["항목", "내용"],
        )
        .style.hide(axis="index")
        .set_table_styles([
            {"selector": "", "props": [("table-layout", "fixed"), ("width", "100%")]},
            {"selector": "th.col0, td.col0", "props": [("width", "20%")]},
            {"selector": "th.col1, td.col1", "props": [("width", "80%")]},
            # Streamlit 1.61 은 셀 내용을 width:fit-content / max-width:25rem(400px) 인
            # 마크다운 상자에 넣는다. 칸을 아무리 넓혀도 글이 400px 에서 접히는 이유가
            # 이것이다. 그 상자를 칸 폭까지 풀어준다.
            {"selector": 'td.col1 [data-testid="stMarkdownContainer"]',
             "props": [("max-width", "none"), ("width", "100%")]},
            # 상자 안의 문단은 white-space:normal 로 고정돼 있어 개행이 죽는다.
            # 마크다운 강제 줄바꿈이 막히는 경우를 대비한 보험이다.
            {"selector": 'td.col1 [data-testid="stMarkdownContainer"] p',
             "props": [("white-space", "pre-line")]},
        ])
    )

    st.subheader("엔진별 요약")

    per_doc_cer = scored.pivot_table(index="doc_id", columns="엔진",
                                     values="cer", aggfunc="mean")
    per_doc_cer = per_doc_cer[[e for e in order if e in per_doc_cer.columns]]
    win_counts = per_doc_cer.idxmin(axis=1).value_counts().reindex(order, fill_value=0)

    summary_view = overall[["문서수", "CER", "WER"]].copy()
    summary_view["CER 1위 문서"] = win_counts
    st.dataframe(
        summary_view.style.format({"CER": "{:.3f}", "WER": "{:.3f}"}),
        width="stretch",
    )

    


# ── 2. 언어별 ─────────────────────────────────────────────────
with tab_lang:
    st.subheader("언어별 오류율")
    lmetric = st.segmented_control(
        "지표", ["CER", "WER"], default="CER", key="lang_metric",
    ) or "CER"

    g = rollup(scored, ["엔진", "group"], mode)
    g["언어"] = g["group"].map(LANG_LABEL)
    vmax = float(g[lmetric].max())

    # 램프가 테마에 따라 뒤집히므로 셀 글자색 조건도 함께 뒤집는다
    hi_ink = p["on_weak"] if dark else p["on_strong"]
    lo_ink = p["on_strong"] if dark else p["on_weak"]

    cells = (
        alt.Chart(g)
        .mark_rect(stroke=p["surface"], strokeWidth=2)   # 셀 사이 2px 표면 간격
        .encode(
            x=alt.X("언어:N", sort=label_order, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("엔진:N", sort=order, title=None),
            color=alt.Color(
                f"{lmetric}:Q",
                scale=alt.Scale(range=p["seq"], domain=[0, vmax]),
                legend=alt.Legend(title=f"{lmetric} (낮을수록 좋음)", format=".2f",
                                  gradientLength=160),
            ),
            tooltip=[alt.Tooltip("엔진:N"), alt.Tooltip("언어:N"),
                     alt.Tooltip(f"{lmetric}:Q", format=".3f"),
                     alt.Tooltip("문서수:Q"), alt.Tooltip("글자수:Q", format=",")],
        )
    )
    cell_labels = cells.mark_text(fontSize=11).encode(
        text=alt.Text(f"{lmetric}:Q", format=".3f"),
        color=alt.condition(alt.datum[lmetric] >= vmax / 2,
                            alt.value(hi_ink), alt.value(lo_ink)),
    )
    # 폭은 픽셀이 아니라 화면 기준으로 잡는다. 4:1 로 나눈 왼쪽 칸(=화면의 80%)에
    # 그리고 width="stretch" 로 그 칸에 맞춘다 — 창을 줄이면 같이 줄고,
    # 칸 밖으로는 절대 못 나간다.
    heat_col, _ = st.columns([8, 1])
    heat_col.altair_chart(
        themed((cells + cell_labels).properties(height=84 * len(order) + 30), p),
        width="stretch",
    )
    
    st.subheader("그래프")
    per_lang = rollup(scored, ["엔진", "group"], mode)
    per_lang["언어"] = per_lang["group"].map(LANG_LABEL)
    lbars = (
        alt.Chart(per_lang)
        .mark_bar(cornerRadiusEnd=3, size=16)
        .encode(
            y=alt.Y("엔진:N", sort=order, title=None, axis=None),
            x=alt.X(f"{lmetric}:Q", title=None, axis=alt.Axis(format=".2f")),
            color=alt.Color("엔진:N", scale=series_scale,
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("언어:N"), alt.Tooltip("엔진:N"),
                        alt.Tooltip(f"{lmetric}:Q", format=".3f"),
                        alt.Tooltip("문서수:Q")],
        )
    )
    llabels = lbars.mark_text(align="left", dx=5, fontSize=11).encode(
        text=alt.Text(f"{lmetric}:Q", format=".3f"), color=alt.value(p["muted"])
    )
    st.altair_chart(
        themed(
            (lbars + llabels)
            # facet 차트는 Vega-Lite 가 컨테이너 맞춤을 지원하지 않아 폭이 픽셀 고정이다.
            # 한 줄에 3칸이면 1000px 이 넘어 화면을 벗어나므로 2칸으로 줄인다.
            .properties(height=34 * len(order) + 20, width=410)
            .facet(facet=alt.Facet("언어:N", title=None, sort=label_order,
                                    header=alt.Header(labelColor=p["muted"],
                                                        labelFontSize=12)),
                    columns=2),
            p,
        ),
        width="stretch",
    )


# ── 4. 데이터셋 ───────────────────────────────────────────────
with tab_data:
    st.subheader("데이터셋")

    docs = scored.drop_duplicates("doc_id")
    doc_order = docs["doc_id"].tolist()          # result2.csv 에 적힌 순서 그대로

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("문서", f"{docs['doc_id'].nunique()}건")
    d2.metric("언어", f"{docs['group'].nunique()}개")
    d3.metric("정답 글자", f"{int(docs['gt_chars'].sum()):,}자")
    d4.metric("정답 단어", f"{int(docs['gt_words'].sum()):,}개")

    ds = (docs.groupby("group")
          .agg(문서수=("doc_id", "nunique"), 글자수=("gt_chars", "sum"),
               단어수=("gt_words", "sum"))
          .reindex(lang_order))
    ds.insert(0, "언어", [LANG_LABEL.get(g, g) for g in ds.index])
    ds["문서당 글자수"] = ds["글자수"] / ds["문서수"]
    ds.index.name = "코드"
    st.dataframe(
        ds.style.format({"글자수": "{:,.0f}", "단어수": "{:,.0f}",
                         "문서당 글자수": "{:,.0f}"}),
        width="stretch",
    )
    
    st.subheader("문서 × 엔진")
    dmetric = st.segmented_control(
        "지표", ["CER", "WER"], default="CER", key="doc_metric",
    ) or "CER"

    per_doc = scored.pivot_table(index="doc_id", columns="엔진",
                                 values=dmetric.lower(), aggfunc="mean")
    engine_cols = [e for e in order if e in per_doc.columns]
    per_doc = per_doc.reindex(doc_order)[engine_cols]

    table = per_doc.copy()
    table.insert(0, "글자수", docs.set_index("doc_id")["gt_chars"].reindex(doc_order))
    table.index = pd.Index([doc_label(d) for d in doc_order], name="문서")

    def mark_best(row):
        """행마다 1등 칸만 초록. 색만으로 뜻을 싣지 않게 숫자를 같이 읽힌다."""
        out = pd.Series("", index=row.index)
        vals = pd.to_numeric(row[engine_cols], errors="coerce")
        if vals.notna().any():
            out[vals.idxmin()] = f"background-color: {TINT_GOOD}"
        return out

    fmt = {"글자수": "{:,.0f}"}
    fmt.update({e: "{:.3f}" for e in engine_cols})
    st.dataframe(
        table.style.apply(mark_best, axis=1).format(fmt),
        width="stretch",
        height=min(620, 36 * len(table) + 40),
    )



# ── 5. 인식 결과 보기 ─────────────────────────────────────────
with tab_text:
    st.subheader("정답과 인식 결과를 나란히")

    doc_ids = sorted(scored["doc_id"].unique())
    doc_id = st.selectbox("문서", doc_ids, format_func=doc_label, key="doc_pick")

    rows = (scored[scored["doc_id"] == doc_id]
            .set_index("엔진").reindex(order).dropna(subset=["engine"]))

    mcols = st.columns(max(len(rows), 1))
    for col, (eng, row) in zip(mcols, rows.iterrows()):
        col.metric(eng, f"CER {row['cer']:.3f}",
                   help=f"WER {row['wer']:.3f} · 정답 {int(row['gt_chars']):,}자 / "
                        f"인식 {int(row['pred_chars']):,}자")

    gt_path, _ = local_paths(doc_id, rows["engine"].iloc[0])
    gt_text = read_text(str(gt_path))
    if not gt_text:
        st.warning(f"정답 텍스트를 못 찾았습니다: `{gt_path}`")

    o1, o2 = st.columns([1, 2])
    use_norm = o1.toggle("채점과 같게 정규화", value=True,
                         help="공백·따옴표 통일 등 채점 직전 상태로 맞춰 봅니다. "
                              "끄면 파일에 저장된 원문 그대로입니다.")
    show_diff = o2.toggle("차이 강조", value=True,
                          help="정답과 다른 자리를 색으로 표시합니다. 긴 문서는 느릴 수 있습니다.")

    profile = str(rows["normalize"].iloc[0]) if len(rows) else "basic"

    def prep(text: str) -> str:
        if not use_norm or not text:
            return text
        try:
            from normalize import normalize as _norm
            return _norm(text, profile=profile)
        except Exception:
            return text

    st.markdown(
        f'<span style="background:{TINT_WARN}">&nbsp;오인식&nbsp;</span> &nbsp;'
        f'<span style="background:{TINT_SERIOUS}">&nbsp;없는 글자를 지어냄&nbsp;</span> &nbsp;'
        f'<span style="background:{TINT_CRIT}">&nbsp;‸ 누락&nbsp;</span>',
        unsafe_allow_html=True,
    )

    ref = prep(gt_text)
    panels = st.columns(len(rows) + 1)
    with panels[0]:
        st.markdown("**정답**")
        st.text_area("정답", ref, height=460, label_visibility="collapsed")

    for panel, (eng, row) in zip(panels[1:], rows.iterrows()):
        with panel:
            st.markdown(f"**{eng}** · CER {row['cer']:.3f}")
            _, pred_path = local_paths(doc_id, row["engine"])
            hyp = prep(read_text(str(pred_path)))
            if not hyp:
                st.warning(f"인식 결과가 없습니다: `{pred_path}`")
            elif show_diff and ref and max(len(ref), len(hyp)) <= DIFF_LIMIT:
                with st.spinner("차이 계산 중"):
                    body = diff_html(ref, hyp)
                st.markdown(
                    '<div style="height:460px;overflow:auto;white-space:pre-wrap;'
                    'word-break:break-all;font-size:13px;line-height:1.6;'
                    f'border:1px solid {p["grid"]};border-radius:6px;padding:10px">'
                    f"{body}</div>",
                    unsafe_allow_html=True,
                )
            else:
                if show_diff and ref:
                    st.caption(f"{DIFF_LIMIT:,}자가 넘어 차이 강조를 껐습니다.")
                st.text_area(eng, hyp, height=460, label_visibility="collapsed")
