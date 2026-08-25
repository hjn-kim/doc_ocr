# -*- coding: utf-8 -*-
"""result/ 의 OCR 벤치마크 결과를 엔진별로 비교하는 대시보드.

    pip install streamlit altair
    streamlit run app.py

로컬에서 띄운다. GPU 서버에는 streamlit 을 깔지 않는다 — 거기서는 벤치마크만
돌리고 result/ 를 받아와서 여기서 본다.

result/result.csv (문서 x 엔진 한 줄) 와 result/result_summary.csv (엔진 x 언어
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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))   # 채점에 쓴 정규화를 그대로 재사용한다

st.set_page_config(page_title="OCR 엔진 비교", page_icon="🔍", layout="wide")


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
    "paddleocr_vl": "PaddleOCR-VL",
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
RESULT_FILES = ("result.csv", "result_summary.csv")


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
    detail = pd.read_csv(d / "result.csv", encoding="utf-8-sig")
    summary = pd.read_csv(d / "result_summary.csv", encoding="utf-8-sig")

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
if not (rdir / "result.csv").exists():
    st.title("OCR 엔진 비교")
    st.warning(f"`{rdir / 'result.csv'}` 가 없습니다.")
    st.code("python src/run_benchmark.py --device gpu:0", language="bash")
    st.caption("먼저 벤치마크를 돌려 결과 파일을 만든 뒤 새로고침하세요.")
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

c1, c2, c3, c4 = st.columns(4)
c1.metric("1위 엔진", best, help="선택한 언어 전체에서 CER 이 가장 낮은 엔진")
c2.metric("CER", f"{overall.loc[best, 'CER']:.1%}",
          help="글자 단위 오류율. (치환+삭제+삽입) / 정답 글자 수. 낮을수록 좋다")
c3.metric("WER", f"{overall.loc[best, 'WER']:.1%}",
          help="공백으로 자른 단어 단위 오류율. 낮을수록 좋다")
c4.metric("정확히 일치", f"{int(scored['exact_match'].sum())}건",
          help="정규화 후 문서 전체가 글자 하나까지 같았던 경우. 문서 단위라 보통 0 이다")

st.caption(
    f"평균 방식 **{mode}** · 정규화 **{norm_used}** 기준. CER·WER 은 **낮을수록 좋습니다.** "
    "정규화를 바꾸면 숫자가 크게 흔들리니 결과를 공유할 때 같이 적으세요."
)

tab_overview, tab_ko, tab_lang, tab_doc, tab_text = st.tabs(
    ["개요", "한국어 vs 다국어", "언어별", "문서별", "인식 결과 보기"]
)


# ── 1. 개요 ───────────────────────────────────────────────────
with tab_overview:
    st.subheader("엔진별 오류율")
    st.caption("막대가 짧을수록 좋습니다. CER 과 WER 은 크기가 달라 축을 나눠 그렸습니다.")

    melted = (
        overall.reset_index()
        .melt(id_vars="엔진", value_vars=["CER", "WER"], var_name="지표", value_name="값")
    )
    bars = (
        alt.Chart(melted)
        .mark_bar(cornerRadiusEnd=4, size=20)   # 데이터 끝만 둥글게, 24px 이하
        .encode(
            y=alt.Y("엔진:N", sort=order, title=None),
            x=alt.X("값:Q", title=None, axis=alt.Axis(format=".0%"),
                    scale=alt.Scale(nice=False, padding=6)),
            # 색은 y축과 같은 정보라 범례가 없어도 된다 (행 이름이 곧 엔진이다)
            color=alt.Color("엔진:N", legend=None, scale=series_scale),
            tooltip=[alt.Tooltip("엔진:N"), alt.Tooltip("지표:N"),
                     alt.Tooltip("값:Q", format=".2%")],
        )
    )
    # 막대 끝에 값 — 축을 읽지 않아도 되게. 텍스트는 계열 색이 아닌 muted 잉크.
    labels = bars.mark_text(align="left", dx=6, fontSize=11).encode(
        text=alt.Text("값:Q", format=".1%"), color=alt.value(p["muted"])
    )
    st.altair_chart(
        themed(
            (bars + labels)
            .properties(height=42 * len(order) + 20, width=320)
            .facet(column=alt.Column("지표:N", title=None,
                                     header=alt.Header(labelColor=p["muted"],
                                                       labelFontSize=13)))
            .resolve_scale(x="independent"),
            p,
        ),
        width="stretch",
    )

    st.subheader("표로 보기")
    view = overall[["문서수", "CER", "WER", "CER_micro", "CER_macro",
                    "WER_micro", "WER_macro", "글자수", "글자오류"]]
    st.dataframe(
        view.style.format({
            "CER": "{:.3f}", "WER": "{:.3f}",
            "CER_micro": "{:.3f}", "CER_macro": "{:.3f}",
            "WER_micro": "{:.3f}", "WER_macro": "{:.3f}",
            "글자수": "{:,.0f}", "글자오류": "{:,.0f}",
        }),
        width="stretch",
    )
    st.caption(
        "**CER** — (치환+삭제+삽입) / 정답 **글자** 수.  \n"
        "**WER** — 같은 계산을 공백으로 자른 **단어** 기준으로.  \n"
        "**micro** 는 편집 횟수를 다 합쳐서 나눈 값이라 긴 문서가 결과를 끌고 갑니다. "
        "**macro** 는 문서별 비율의 평균이라 짧은 문서도 한 표를 갖습니다. "
        "두 값이 벌어지는 엔진은 문서 길이에 따라 성적이 달라진다는 뜻입니다."
    )

    st.subheader("벤치마크가 직접 쓴 집계 (result_summary.csv)")
    st.caption("위 표는 result.csv 를 이 앱에서 다시 묶은 것이고, 아래는 원본 그대로입니다.")
    st.dataframe(
        summary_all[summary_all["group"] == "ALL"]
        .set_index("엔진")
        .drop(columns=["engine", "group"], errors="ignore"),
        width="stretch",
    )

    if not has_timing:
        st.info(
            "이 결과에는 **소요 시간이 기록되지 않았습니다** (`seconds`, `n_pages` 가 비어 있음). "
            "정확도만 비교할 수 있고 속도 비교는 못 합니다. 속도까지 보려면 벤치마크를 다시 돌리세요."
        )
    else:
        st.subheader("속도 대비 정확도")
        st.caption("왼쪽 아래가 좋습니다 — 빠르고(쪽당 시간이 짧고) 정확한(CER 이 낮은) 엔진.")
        speed = (
            scored.groupby("엔진")
            .agg(초=("seconds", "sum"), 쪽=("n_pages", "sum"))
            .assign(쪽당초=lambda x: x["초"] / x["쪽"])
            .join(overall[["CER"]])
            .reset_index()
        )
        base = alt.Chart(speed).encode(
            x=alt.X("쪽당초:Q", title="쪽당 소요 시간 (초, 작을수록 빠름)",
                    scale=alt.Scale(zero=False, padding=24)),
            y=alt.Y("CER:Q", title="CER (낮을수록 정확)",
                    scale=alt.Scale(zero=False, padding=24),
                    axis=alt.Axis(format=".0%")),
            tooltip=[alt.Tooltip("엔진:N"), alt.Tooltip("CER:Q", format=".2%"),
                     alt.Tooltip("쪽당초:Q", format=".2f", title="초/쪽")],
        )
        dots = base.mark_point(
            size=200, filled=True, stroke=p["surface"], strokeWidth=2  # 겹칠 때 표면색 링
        ).encode(color=alt.Color("엔진:N", legend=None, scale=series_scale))
        names = base.mark_text(align="left", dx=14, fontSize=11).encode(
            text="엔진:N", color=alt.value(p["muted"])
        )
        st.altair_chart(themed((dots + names).properties(height=360), p), width="stretch")


# ── 2. 한국어 vs 다국어 ───────────────────────────────────────
with tab_ko:
    st.subheader("한국어와 다국어 중 어디서 갈리는가")
    st.caption(
        "전체 평균 하나로는 *한국어에서만 무너지는 엔진*이 보이지 않습니다. "
        "그래서 한국어와 나머지 언어를 갈라서 나란히 놓습니다."
    )

    others = [g for g in lang_order if g != "ko"]
    if "ko" not in lang_order:
        st.warning("사이드바에서 한국어(ko)를 선택해야 이 비교를 볼 수 있습니다.")
    elif not others:
        st.warning("한국어만 선택되어 있어 비교할 다국어가 없습니다. 사이드바에서 다른 언어도 고르세요.")
    else:
        cc1, cc2 = st.columns([2, 3])
        metric = cc1.radio("지표", ["CER", "WER"], horizontal=True, key="ko_metric")
        scope = cc2.radio(
            "다국어 범위",
            [f"한국어 외 {len(others)}개 언어", f"{len(lang_order)}개 언어 전체"],
            horizontal=True, key="ko_scope",
            help="'한국어 외' 는 두 집단이 겹치지 않아 격차가 또렷합니다. "
                 "'전체' 는 한국어를 포함한 평균이라 실제 서비스 평균에 가깝습니다.",
        )
        ko_only = scope.startswith("한국어 외")

        ko_df = scored[scored["group"] == "ko"]
        multi_df = scored[scored["group"] != "ko"] if ko_only else scored

        ko_agg = rollup(ko_df, ["엔진"], mode).set_index("엔진")[metric]
        multi_agg = rollup(multi_df, ["엔진"], mode).set_index("엔진")[metric]
        cmp = (pd.DataFrame({"한국어": ko_agg, "다국어": multi_agg})
               .reindex(order).reset_index())
        cmp["격차"] = cmp["한국어"] - cmp["다국어"]
        cmp["배수"] = cmp["한국어"] / cmp["다국어"]

        cols = st.columns(len(order))
        for col, eng in zip(cols, order):
            row = cmp[cmp["엔진"] == eng].iloc[0]
            col.metric(
                eng, f"한국어 {row['한국어']:.1%}",
                delta=f"{row['격차']:+.1%}p vs 다국어",
                delta_color="inverse",   # 오류율이라 올라가면 나쁘다
                help=f"다국어 {row['다국어']:.1%} · 한국어가 {row['배수']:.1f}배",
            )

        # 항목 하나가 두 상태 사이를 오가는 그림 → 덤벨. 막대 두 개보다 격차가 바로 읽힌다.
        long = cmp.melt(id_vars="엔진", value_vars=["다국어", "한국어"],
                        var_name="구분", value_name="값")
        rules = (
            alt.Chart(cmp)
            .mark_rule(strokeWidth=2, color=p["grid"])
            .encode(y=alt.Y("엔진:N", sort=order, title=None),
                    x=alt.X("다국어:Q", title=metric, axis=alt.Axis(format=".0%"),
                            scale=alt.Scale(nice=False, padding=44)),
                    x2="한국어:Q")
        )
        dots = (
            alt.Chart(long)
            .mark_point(size=190, filled=True, stroke=p["surface"], strokeWidth=2)
            .encode(
                y=alt.Y("엔진:N", sort=order, title=None),
                x=alt.X("값:Q", title=metric),
                # 색은 엔진(y축과 같은 정보), 모양이 한국어/다국어를 가른다 —
                # 색 하나에 두 가지 뜻을 얹지 않는다.
                color=alt.Color("엔진:N", legend=None, scale=series_scale),
                shape=alt.Shape("구분:N", title=None,
                                scale=alt.Scale(domain=["다국어", "한국어"],
                                                range=["circle", "diamond"]),
                                legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("엔진:N"), alt.Tooltip("구분:N"),
                         alt.Tooltip("값:Q", format=".2%", title=metric)],
            )
        )
        # 격차는 줄 오른쪽 끝에 한 번만 — 점마다 숫자를 붙이면 겹친다
        gaps = (
            alt.Chart(cmp.assign(끝=cmp[["한국어", "다국어"]].max(axis=1)))
            .mark_text(align="left", dx=14, fontSize=11)
            .encode(y=alt.Y("엔진:N", sort=order, title=None), x=alt.X("끝:Q"),
                    text=alt.Text("격차:Q", format="+.1%"), color=alt.value(p["muted"]))
        )
        st.altair_chart(
            themed((rules + dots + gaps).properties(height=54 * len(order) + 40), p),
            width="stretch",
        )
        st.caption(
            "◇ 한국어 · ○ 다국어. 선이 길수록 두 축의 성적이 다릅니다. "
            "◇ 가 ○ 오른쪽에 있으면 그 엔진은 한국어에서 더 많이 틀립니다. "
            "줄 끝 숫자는 한국어 − 다국어 격차(%p)입니다."
        )

        st.subheader("표로 보기")
        st.dataframe(
            cmp.set_index("엔진").style.format(
                {"한국어": "{:.3f}", "다국어": "{:.3f}",
                 "격차": "{:+.3f}", "배수": "{:.2f}배"}
            ),
            width="stretch",
        )

        # 해설은 데이터에서 뽑는다 — 결과를 다시 돌려도 문장이 따라 바뀐다.
        ko_best = cmp.loc[cmp["한국어"].idxmin()]
        multi_best = cmp.loc[cmp["다국어"].idxmin()]
        worst_gap = cmp.loc[cmp["격차"].idxmax()]
        lines = [
            f"- **한국어 1위 — {ko_best['엔진']}** ({metric} {ko_best['한국어']:.1%})",
            f"- **다국어 1위 — {multi_best['엔진']}** ({metric} {multi_best['다국어']:.1%})",
            f"- 한국어에서 가장 손해 보는 엔진은 **{worst_gap['엔진']}** — "
            f"다국어 {worst_gap['다국어']:.1%} → 한국어 {worst_gap['한국어']:.1%} "
            f"({worst_gap['배수']:.1f}배).",
        ]
        if ko_best["엔진"] != multi_best["엔진"]:
            lines.append(
                f"- 두 축의 1위가 다릅니다. 한국어 문서가 주력이면 **{ko_best['엔진']}**, "
                f"여러 언어를 고루 받아야 하면 **{multi_best['엔진']}** 쪽입니다."
            )
        else:
            lines.append(f"- 두 축 모두 **{ko_best['엔진']}** 이 1위라 고민할 게 없습니다.")
        st.markdown("\n".join(lines))

        st.subheader("다국어 쪽을 언어별로 펼치면")
        per_lang = rollup(scored, ["엔진", "group"], mode)
        per_lang["언어"] = per_lang["group"].map(LANG_LABEL)
        lbars = (
            alt.Chart(per_lang)
            .mark_bar(cornerRadiusEnd=3, size=13)
            .encode(
                y=alt.Y("엔진:N", sort=order, title=None, axis=None),
                x=alt.X(f"{metric}:Q", title=None, axis=alt.Axis(format=".0%")),
                color=alt.Color("엔진:N", scale=series_scale,
                                legend=alt.Legend(orient="top", title=None)),
                tooltip=[alt.Tooltip("언어:N"), alt.Tooltip("엔진:N"),
                         alt.Tooltip(f"{metric}:Q", format=".2%"),
                         alt.Tooltip("문서수:Q")],
            )
        )
        llabels = lbars.mark_text(align="left", dx=5, fontSize=10).encode(
            text=alt.Text(f"{metric}:Q", format=".1%"), color=alt.value(p["muted"])
        )
        st.altair_chart(
            themed(
                (lbars + llabels)
                .properties(height=18 * len(order) + 14, width=260)
                .facet(facet=alt.Facet("언어:N", title=None, sort=label_order,
                                       header=alt.Header(labelColor=p["muted"],
                                                         labelFontSize=12)),
                       columns=3),
                p,
            ),
            width="stretch",
        )
        if metric == "WER" and NO_SPACE_LANGS & set(lang_pick):
            st.warning(
                "중국어는 띄어쓰기가 없어 **WER 이 의미가 없습니다** — 문장이 통째로 한 단어로 "
                "잡혀 1 을 넘기기도 합니다. 중국어는 CER 로 보세요."
            )


# ── 3. 언어별 ─────────────────────────────────────────────────
with tab_lang:
    st.subheader("언어별 오류율")
    lmetric = st.radio("지표", ["CER", "WER"], horizontal=True, key="lang_metric")

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
                legend=alt.Legend(title=f"{lmetric} (낮을수록 좋음)", format=".0%",
                                  gradientLength=160),
            ),
            tooltip=[alt.Tooltip("엔진:N"), alt.Tooltip("언어:N"),
                     alt.Tooltip(f"{lmetric}:Q", format=".2%"),
                     alt.Tooltip("문서수:Q"), alt.Tooltip("글자수:Q", format=",")],
        )
    )
    cell_labels = cells.mark_text(fontSize=11).encode(
        text=alt.Text(f"{lmetric}:Q", format=".1%"),
        color=alt.condition(alt.datum[lmetric] >= vmax / 2,
                            alt.value(hi_ink), alt.value(lo_ink)),
    )
    st.altair_chart(
        themed((cells + cell_labels).properties(height=48 * len(order) + 20), p),
        width="stretch",
    )
    st.caption(
        "**진한 칸이 많이 틀린 언어입니다.** 세로로 훑으면 *그 언어를 어느 엔진이 잘 읽는지*, "
        "가로로 훑으면 *그 엔진이 어떤 언어에서 무너지는지* 보입니다."
    )

    st.subheader("표로 보기")
    pivot = g.pivot(index="엔진", columns="언어", values=lmetric)
    pivot = pivot.reindex(index=[e for e in order if e in pivot.index],
                          columns=[c for c in label_order if c in pivot.columns])
    st.dataframe(
        pivot.style.format("{:.3f}")
        .background_gradient(cmap="Blues", vmin=0, vmax=vmax),
        width="stretch",
    )
    counts = g.drop_duplicates("group").set_index("언어")["문서수"]
    st.caption(
        "언어별 문서 수 — "
        + " · ".join(f"{c} {int(counts[c])}개" for c in pivot.columns if c in counts.index)
    )

    st.subheader("언어별 1위")
    ranked = g.sort_values(lmetric)
    winners = (
        # drop_duplicates 로 첫 행을 통째로 가져온다. groupby().first() 는 열마다
        # 따로 첫 값을 골라서, 어느 열에 NaN 이 있으면 서로 다른 행이 섞인다.
        ranked.drop_duplicates("group")
        .assign(순서=lambda x: x["group"].map({l: i for i, l in enumerate(lang_order)}))
        .sort_values("순서")
    )
    # 2위와의 차이가 작으면 "1위"라는 말이 과하다. 그 간격도 같이 보여준다.
    gap = ranked.groupby("group")[lmetric].apply(
        lambda s: (s.iloc[1] - s.iloc[0]) if len(s) > 1 else float("nan")
    )
    winners["2위와 차이"] = winners["group"].map(gap)
    st.dataframe(
        winners[["언어", "엔진", lmetric, "2위와 차이", "문서수"]]
        .rename(columns={"엔진": "1위 엔진"}).set_index("언어")
        .style.format({lmetric: "{:.3f}", "2위와 차이": "{:+.3f}"}),
        width="stretch",
    )
    st.caption("**2위와 차이**가 0 에 가까우면 사실상 동률입니다. 그 언어는 다른 기준으로 고르세요.")
    if lmetric == "WER" and NO_SPACE_LANGS & set(lang_pick):
        st.warning("중국어 WER 은 띄어쓰기가 없어 뜻이 없습니다. CER 로 판단하세요.")


# ── 4. 문서별 ─────────────────────────────────────────────────
with tab_doc:
    dmetric = st.radio("지표", ["CER", "WER"], horizontal=True, key="doc_metric")

    per_doc = scored.pivot_table(index=["group", "문서"], columns="엔진",
                                 values=dmetric.lower(), aggfunc="mean")
    per_doc = per_doc[[e for e in order if e in per_doc.columns]]

    st.subheader("어느 엔진이 몇 개 문서에서 1등인가")
    wins = per_doc.idxmin(axis=1).value_counts().reindex(order, fill_value=0)
    wins_df = wins.rename_axis("엔진").rename("문서 수").reset_index()

    wbars = (
        alt.Chart(wins_df)
        .mark_bar(cornerRadiusEnd=4, size=20)
        .encode(
            y=alt.Y("엔진:N", sort=order, title=None),
            x=alt.X("문서 수:Q", title=None,
                    scale=alt.Scale(domain=[0, int(len(per_doc))], nice=False)),
            color=alt.Color("엔진:N", legend=None, scale=series_scale),
            tooltip=[alt.Tooltip("엔진:N"), alt.Tooltip("문서 수:Q")],
        )
    )
    wlabels = wbars.mark_text(align="left", dx=6, fontSize=11).encode(
        text=alt.Text("문서 수:Q"), color=alt.value(p["muted"])
    )
    st.altair_chart(
        themed((wbars + wlabels).properties(height=42 * len(order) + 20), p),
        width="stretch",
    )
    st.caption(
        f"전체 {len(per_doc)}개 문서에서 {dmetric} 이 가장 낮았던 횟수입니다. "
        "평균이 좋은 엔진과 *많은 문서에서 1등인* 엔진은 다를 수 있습니다 — "
        "한 문서에서 크게 망하면 평균은 무너져도 승수는 그대로거든요."
    )

    st.subheader("문서 × 엔진")
    st.caption("초록 칸이 그 문서의 1등입니다. 어려운 문서(엔진 평균이 나쁜 순)부터 정렬했습니다.")
    table = per_doc.copy()
    table.insert(0, "평균", table.mean(axis=1))
    table = table.sort_values("평균", ascending=False)
    table.index = pd.Index([f"{LANG_LABEL.get(grp, grp)} · {doc}"
                            for grp, doc in table.index], name="문서")
    engine_cols = [c for c in table.columns if c != "평균"]

    def mark_best(row):
        """행마다 1등 칸만 초록. 색만으로 뜻을 싣지 않게 숫자를 같이 읽힌다."""
        out = pd.Series("", index=row.index)
        vals = row[engine_cols].astype(float)
        if vals.notna().any():
            out[vals.idxmin()] = f"background-color: {TINT_GOOD}"
        return out

    st.dataframe(
        table.style.apply(mark_best, axis=1).format("{:.3f}"),
        width="stretch",
        height=min(620, 36 * len(table) + 40),
    )

    st.subheader("모든 엔진이 헤맨 문서")
    hard = table[table[engine_cols].min(axis=1) > 0.3]
    if hard.empty:
        st.success(f"모든 문서를 최소 한 엔진이 {dmetric} 0.3 아래로 읽었습니다.")
    else:
        st.warning(
            f"{len(hard)}개 문서는 **어느 엔진도** {dmetric} 0.3 아래로 못 읽었습니다. "
            "엔진 성능 문제가 아니라 **정답 텍스트가 틀렸을 가능성**을 먼저 보세요 — "
            "표 순서가 꼬였거나, 머리말·쪽번호가 섞여 들어갔거나, 원문이 스캔본일 수 있습니다."
        )
        st.dataframe(hard.style.format("{:.3f}"), width="stretch")


# ── 5. 인식 결과 보기 ─────────────────────────────────────────
with tab_text:
    st.subheader("정답과 인식 결과를 나란히")
    st.caption(
        "숫자가 왜 그렇게 나왔는지는 결국 원문을 봐야 압니다. "
        "`result/pred/<엔진>/<문서>.txt` 와 `data/gt/` 의 정답을 직접 읽습니다."
    )

    doc_ids = sorted(scored["doc_id"].unique())
    doc_id = st.selectbox("문서", doc_ids, format_func=doc_label, key="doc_pick")

    rows = (scored[scored["doc_id"] == doc_id]
            .set_index("엔진").reindex(order).dropna(subset=["engine"]))

    mcols = st.columns(len(rows))
    for col, (eng, row) in zip(mcols, rows.iterrows()):
        col.metric(eng, f"CER {row['cer']:.1%}",
                   help=f"WER {row['wer']:.1%} · 정답 {int(row['gt_chars']):,}자 / "
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
            st.markdown(f"**{eng}** · CER {row['cer']:.1%}")
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
