"""results/ 의 벤치마크 결과를 모델별로 비교하는 대시보드.

    pip install streamlit
    streamlit run app.py

로컬에서 띄운다. GPU 파드에는 streamlit 을 설치하지 않는다 (requirements.txt 주석 참고).

results/summary.csv, details.json, chunks.json 을 읽는다.
먼저 `python -m src.run_benchmark` 를 돌려서 그 파일들을 만들어야 한다.
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="임베딩 모델 비교", page_icon="📊", layout="wide")


# ── 색 ────────────────────────────────────────────────────────
# dataviz 레퍼런스 팔레트. 두 계열 조합은 light/dark 모두 검증기를 통과했다
# (CVD ΔE 24.7 / 26.8, 일반시야 33.6 / 31.8, 대비 3:1 이상).
# 어두운 쪽은 밝은 쪽을 그냥 뒤집은 게 아니라, 어두운 표면에 맞춰 따로 고른 단계다.
def app_theme_is_dark() -> bool:
    """앱이 지금 어두운 테마로 그려지고 있는지. 브라우저 Settings 변경을 실시간 반영."""
    try:
        return st.context.theme.type == "dark"
    except Exception:
        return st.get_option("theme.base") == "dark"


# 히트맵용 순차 램프. 계열색과 같은 파랑 한 가지를 밝기만 바꿔 쓴다(무지개 금지).
# "0 에 가까운 값이 표면 쪽으로 물러난다"는 규칙 때문에 어두운 테마는 순서를 뒤집는다.
SEQ_LIGHT = ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95"]
SEQ_DARK = ["#0d366b", "#184f95", "#2a78d6", "#5598e7", "#9ec5f4"]


def palette(dark: bool) -> dict[str, str]:
    return {
        "series1": "#3987e5" if dark else "#2a78d6",   # 파랑
        "series2": "#d95926" if dark else "#eb6834",   # 주황
        "surface": "#1a1a19" if dark else "#fcfcfb",
        "grid": "#2c2c2a" if dark else "#e1e0d9",
        "muted": "#898781",                             # 축·라벨 (양쪽 공용)
        "seq": SEQ_DARK if dark else SEQ_LIGHT,
        # 진한 셀 위 / 옅은 셀 위 글자색. 어느 쪽 테마든 셀 대비를 확보한다.
        "on_strong": "#f3f2ef",
        "on_weak": "#2c2c2a",
    }


# 질문 id 접두사가 곧 문서의 언어다 (ch-01-1 → ch). 질문 자체는 전부 한국어이므로
# 이 축은 "한국어로 물었을 때 그 언어 문서를 얼마나 잘 찾아오는가"를 뜻한다.
LANG_LABEL = {
    "ch": "중국어",
    "en": "영어",
    "ko": "한국어",
    "pil": "타갈로그어",
    "ru": "러시아어",
    "uz": "우즈베크어",
    "vn": "베트남어",
}


# 상태 색은 고정 — 계열 색으로 재사용하지 않는다. 항상 숫자/기호와 함께 쓴다.
RANK_TINT = {
    1: "rgba(12,163,12,0.20)",     # good     — 1등으로 맞힘
    2: "rgba(250,178,25,0.20)",    # warning  — 2~3등
    3: "rgba(250,178,25,0.20)",
    4: "rgba(236,131,90,0.20)",    # serious  — 4~5등
    5: "rgba(236,131,90,0.20)",
}
MISS_TINT = "rgba(208,59,59,0.20)"  # critical — 상위 5개 밖


def themed(chart, p: dict[str, str]):
    """차트에 배경·여백·축 스타일을 입힌다.

    배경과 여백을 .configure() 로 주면 안 된다 — Altair 의 .configure() 는 config 를
    통째로 갈아끼워서 앞서 부른 .configure_axis() 설정이 조용히 사라진다.
    둘 다 Vega-Lite 최상위 속성이므로 .properties() 로 준다.
    격자는 실선 hairline 으로 눌러 데이터만 도드라지게 한다.
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
        .configure_view(strokeWidth=0)
    )


# ── 데이터 로드 ───────────────────────────────────────────────
RESULT_FILES = ("summary.csv", "details.json", "chunks.json")


def results_stamp(d: Path) -> float:
    """결과 파일들의 최신 수정 시각. load_results 의 캐시 키로 쓴다.

    폴더 mtime 을 쓰면 안 된다 — 파일이 추가·삭제될 때만 바뀌고 같은 이름으로
    덮어쓰면 그대로라, 벤치마크를 다시 돌려도 옛 결과가 계속 보인다.
    """
    times = [(d / f).stat().st_mtime for f in RESULT_FILES if (d / f).exists()]
    return max(times) if times else 0.0


@st.cache_data(show_spinner=False)
def load_results(results_dir: str, stamp: float):
    """stamp 는 캐시 키 용도로만 받는다 (본문에서 쓰지 않는 게 정상)."""
    d = Path(results_dir)
    summary = pd.read_csv(d / "summary.csv")
    details = json.loads((d / "details.json").read_text(encoding="utf-8"))
    chunks_path = d / "chunks.json"
    chunks = (
        json.loads(chunks_path.read_text(encoding="utf-8")) if chunks_path.exists() else []
    )
    return summary, details, chunks


def build_question_frame(details: list[dict]) -> pd.DataFrame:
    """질문 × 모델 단위 표. gold_rank = 정답 청크가 몇 등으로 올라왔는지(없으면 NaN)."""
    rows = []
    for d in details:
        rank = next((i for i, t in enumerate(d["top5"], start=1) if t["is_gold"]), None)
        rows.append(
            {
                "model": d["model"],
                "qid": d["qid"],
                "lang": d["qid"].split("-")[0],
                "question": d["question"],
                "n_gold": d["n_gold"],
                "gold_rank": rank,
                **{k: v for k, v in d.items() if k[:4] in ("Hit@", "MRR@", "nDCG")},
            }
        )
    return pd.DataFrame(rows)


# ── 사이드바 ──────────────────────────────────────────────────
st.sidebar.header("설정")
results_dir = st.sidebar.text_input("results 폴더", value=str(ROOT / "results"))

rdir = Path(results_dir)
if not (rdir / "summary.csv").exists():
    st.title("임베딩 모델 비교")
    st.warning(f"`{rdir / 'summary.csv'}` 가 없습니다.")
    st.code("python -m src.run_benchmark", language="bash")
    st.caption("먼저 벤치마크를 돌려 결과 파일을 만든 뒤 새로고침하세요.")
    st.stop()

summary, details, chunks = load_results(str(rdir), results_stamp(rdir))
qdf = build_question_frame(details)

hit_cols = sorted([c for c in summary.columns if c.startswith("Hit@")],
                  key=lambda c: int(c.split("@")[1]))
ks = [int(c.split("@")[1]) for c in hit_cols]
lo, hi = min(ks), max(ks)
sort_col = f"nDCG@{hi}" if f"nDCG@{hi}" in summary.columns else hit_cols[0]
order = summary.sort_values(sort_col, ascending=False)["model"].tolist()

picked = st.sidebar.multiselect("비교할 모델", order, default=order)
if not picked:
    st.warning("모델을 하나 이상 선택하세요.")
    st.stop()

summary = summary[summary["model"].isin(picked)].copy()
qdf = qdf[qdf["model"].isin(picked)].copy()
order = [m for m in order if m in picked]

if st.sidebar.button("결과 다시 읽기"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
theme_choice = st.sidebar.radio(
    "차트 테마",
    ["자동", "밝게", "어둡게"],
    horizontal=True,
    help="차트의 배경과 계열 색을 함께 바꿉니다. "
         "자동은 앱 테마(우상단 ⋮ → Settings)를 따라가고, "
         "밝게/어둡게는 앱 테마와 무관하게 차트만 고정합니다 — 보고서 스크린샷용.",
)
dark = (
    app_theme_is_dark() if theme_choice == "자동" else (theme_choice == "어둡게")
)
p = palette(dark)

n_questions = qdf["qid"].nunique()
n_chunks = int(summary["청크수"].iloc[0]) if "청크수" in summary.columns else len(chunks)

st.sidebar.divider()
st.sidebar.caption(f"질문 {n_questions}개 · 청크 {n_chunks}개 · 모델 {len(order)}개")


# ── 헤더 ──────────────────────────────────────────────────────
st.title("임베딩 모델 검색 성능 비교")

best = summary.sort_values(sort_col, ascending=False).iloc[0]
c1, c2, c3, c4 = st.columns(4)
c1.metric("1위 모델", best["model"])
c2.metric(f"Hit@{lo}", f"{best[f'Hit@{lo}']:.3f}",
          help=f"상위 {lo}개 안에 정답이 있던 질문 비율")
c3.metric(f"정답@{lo}", f"{int(best[f'정답@{lo}'])} / {n_questions}")
c4.metric("인코딩 속도", f"{best['chunks_per_s']:,.0f} chunks/s")

st.caption(
    f"표 정렬 기준 **{sort_col}**. "
)

tab_overview, tab_lang, tab_questions, tab_misses, tab_chunks = st.tabs(
    ["개요", "언어별", "질문별 비교", "오답 분석", "청크 탐색"]
)


# ── 1. 개요 ───────────────────────────────────────────────────
with tab_overview:
    st.subheader("요약")
    st.dataframe(
        summary.set_index("model").drop(columns=["hf_id"], errors="ignore"),
        width="stretch",
    )
    st.caption(
        f"**MRR@{hi}** — 정답 청크가 몇 등으로 올라왔는지의 역수를 평균한 값 "
        f"(1등 1.0 · 2등 0.5 · 3등 0.33 · {hi}등 밖 0)으로, 맞혔는지와 얼마나 위에 "
        f"올렸는지를 한 숫자로 묶은 지표.  \n"
        f"**nDCG@{hi}** — 정답을 상위에 몰아놓은 정도를 0~1 로 정규화한 값으로, "
        f"첫 정답만 보는 MRR 과 달리 정답이 여러 개일 때 나머지 정답의 위치까지 반영한다."
    )

    st.subheader("정확도")
    melted = summary.melt(
        id_vars="model", value_vars=hit_cols, var_name="지표", value_name="값"
    )
    bars = (
        alt.Chart(melted)
        .mark_bar(cornerRadiusEnd=4, size=20)   # 데이터 끝만 둥글게, 24px 이하
        .encode(
            y=alt.Y("model:N", sort=order, title=None),
            x=alt.X("값:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".0%")),
            # 색은 패널당 하나뿐이라 범례 없이 패널 제목이 지표를 말해 준다
            color=alt.Color(
                "지표:N", legend=None,
                scale=alt.Scale(domain=hit_cols, range=[p["series1"], p["series2"]]),
            ),
            tooltip=[alt.Tooltip("model:N", title="모델"),
                     alt.Tooltip("지표:N"),
                     alt.Tooltip("값:Q", format=".3f")],
        )
    )
    # 막대 끝에 값 — 축을 읽지 않아도 되게. 텍스트는 계열 색이 아닌 muted 잉크.
    labels = bars.mark_text(align="left", dx=6, color=p["muted"], fontSize=11).encode(
        text=alt.Text("값:Q", format=".3f"), color=alt.value(p["muted"])
    )
    st.altair_chart(
        themed(
            (bars + labels)
            .properties(height=42 * len(order) + 20)
            .facet(column=alt.Column(
                "지표:N", title=None,
                header=alt.Header(labelColor=p["muted"], labelFontSize=13))),
            p,
        ),
        width="stretch",
    )

    st.subheader("정확도 대비 비용")
    st.caption(
        "오른쪽 위가 좋습니다. 점 크기는 인덱스 용량 — 작은 모델이 큰 모델과 비슷한 "
        "정확도를 낸다면 서비스에서는 그쪽이 정답일 수 있습니다."
    )
    scat = summary.copy()
    base = alt.Chart(scat).encode(
        x=alt.X("chunks_per_s:Q", title="chunks/s (클수록 빠름)",
                scale=alt.Scale(zero=False, padding=20)),
        y=alt.Y(f"{sort_col}:Q", title=sort_col,
                scale=alt.Scale(zero=False, padding=20)),
        tooltip=[alt.Tooltip("model:N", title="모델"),
                 alt.Tooltip(f"{sort_col}:Q", format=".3f"),
                 alt.Tooltip("chunks_per_s:Q", format=",.0f", title="chunks/s"),
                 alt.Tooltip("dim:Q", title="차원"),
                 alt.Tooltip("index_MB:Q", title="인덱스 MB")],
    )
    dots = base.mark_circle(
        opacity=1, stroke=p["surface"], strokeWidth=2   # 겹칠 때를 대비한 표면색 링
    ).encode(
        size=alt.Size("index_MB:Q", legend=None, scale=alt.Scale(range=[120, 700])),
        color=alt.value(p["series1"]),                  # 단일 계열 → 범례 불필요
    )
    names = base.mark_text(align="left", dx=14, fontSize=11, color=p["muted"]).encode(
        text="model:N"
    )
    st.altair_chart(
        themed((dots + names).properties(height=380), p), width="stretch"
    )


# ── 2. 언어별 ─────────────────────────────────────────────────
with tab_lang:
    st.subheader("언어별 정확도")
    st.caption(
        "질문은 전부 한국어이고 문서만 언어가 다릅니다. 다음은 교차 언어 검색 성능입니다"
    )

    # Hit 은 "맞혔나", MRR·nDCG 는 "얼마나 위에 올렸나"를 본다.
    # 셋 다 0~1 범위라 같은 히트맵 색 눈금을 그대로 쓸 수 있다.
    lang_metrics = hit_cols + [
        c for c in (f"MRR@{hi}", f"nDCG@{hi}") if c in qdf.columns
    ]
    klang = st.radio("기준", lang_metrics, horizontal=True, index=0, key="lang_k")
    is_hit = klang.startswith("Hit@")   # 이때만 "맞힌 문항 수"가 말이 된다

    lang_order = [l for l in LANG_LABEL if l in set(qdf["lang"])]
    label_order = [LANG_LABEL[l] for l in lang_order]

    g = qdf.groupby(["model", "lang"])[klang].agg(["mean", "size"]).reset_index()
    g.columns = ["model", "lang", "값", "문항수"]
    # Hit@k 는 0/1 이라 평균 x 문항수 = 맞힌 개수지만, MRR·nDCG 는 부분점수라
    # 같은 계산이 "맞힌 문항 수"를 뜻하지 않는다. 그래서 Hit 일 때만 만든다.
    if is_hit:
        g["맞힌수"] = (g["값"] * g["문항수"]).round().astype(int)
    g["언어"] = g["lang"].map(LANG_LABEL)

    # 램프가 테마에 따라 뒤집히므로 셀 글자색 조건도 함께 뒤집는다
    hi_ink = p["on_weak"] if dark else p["on_strong"]
    lo_ink = p["on_strong"] if dark else p["on_weak"]

    cells = (
        alt.Chart(g)
        .mark_rect(stroke=p["surface"], strokeWidth=2)   # 셀 사이 2px 표면 간격
        .encode(
            x=alt.X("언어:N", sort=label_order, title=None,
                    axis=alt.Axis(labelAngle=0)),
            y=alt.Y("model:N", sort=order, title=None),
            color=alt.Color(
                "값:Q",
                scale=alt.Scale(range=p["seq"], domain=[0, 1]),
                legend=alt.Legend(title=klang, format=".0%", gradientLength=150),
            ),
            tooltip=[alt.Tooltip("model:N", title="모델"),
                     alt.Tooltip("언어:N"),
                     alt.Tooltip("값:Q", format=".3f", title=klang)]
                    + ([alt.Tooltip("맞힌수:Q", title="맞힌 문항")] if is_hit else [])
                    + [alt.Tooltip("문항수:Q", title="전체 문항")],
        )
    )
    cell_labels = cells.mark_text(fontSize=11).encode(
        text=alt.Text("값:Q", format=".2f"),
        color=alt.condition(alt.datum["값"] >= 0.5, alt.value(hi_ink), alt.value(lo_ink)),
    )
    st.altair_chart(
        themed((cells + cell_labels).properties(height=44 * len(order) + 20), p),
        width="stretch",
    )

    meaning = (
        f"{klang} — 상위 {klang.split('@')[1]}개 안에 정답 청크가 들어온 질문 비율"
        if is_hit else
        (f"{klang} — 정답 청크가 몇 등으로 올라왔는지의 역수 평균 (1등 1.0 · 2등 0.5)"
         if klang.startswith("MRR@") else
         f"{klang} — 정답을 얼마나 위쪽에 몰아놨는지 (정답이 여러 개일 때도 반영)")
    )
    st.caption(
        f"**{meaning}.**  진한 칸일수록 잘 찾은 언어입니다. 세로로 훑으면 "
        "*어느 모델이 그 언어에 강한지*, 가로로 훑으면 *그 모델이 어떤 언어에서 "
        "무너지는지* 보입니다."
    )

    st.subheader("표로 보기")
    pivot = g.pivot(index="model", columns="언어", values="값")
    pivot = pivot.reindex(index=[m for m in order if m in pivot.index],
                          columns=[c for c in label_order if c in pivot.columns])
    counts = g.drop_duplicates("lang").set_index("언어")["문항수"]
    st.dataframe(
        pivot.style.format("{:.3f}").background_gradient(cmap="Blues", vmin=0, vmax=1),
        width="stretch",
    )
    st.caption(
        "언어별 문항 수 — "
        + " · ".join(f"{c} {int(counts[c])}개" for c in pivot.columns if c in counts.index)
    )

    st.subheader("언어별 1위")
    winners = (
        g.loc[g.groupby("lang")["값"].idxmax()]
        .assign(순서=lambda x: x["lang"].map({l: i for i, l in enumerate(lang_order)}))
        .sort_values("순서")
    )
    st.dataframe(
        winners[["언어", "model", "값"] + (["맞힌수"] if is_hit else []) + ["문항수"]]
        .rename(columns={"model": "1위 모델", "값": klang})
        .set_index("언어")
        .style.format({klang: "{:.3f}"}),
        width="stretch",
    )


# ── 2. 질문별 비교 ────────────────────────────────────────────
with tab_questions:
    st.subheader("질문 × 모델")
    st.caption(
        "칸의 숫자는 **정답 청크가 몇 등으로 올라왔는지**입니다. "
        "`1` 이면 1등으로 맞힌 것, `—` 는 상위 5개 안에 못 넣은 것입니다."
    )

    matrix = qdf.pivot(index="qid", columns="model", values="gold_rank")
    matrix = matrix[[m for m in order if m in matrix.columns]]

    def tint(v):
        if pd.isna(v):
            return f"background-color: {MISS_TINT}"
        return f"background-color: {RANK_TINT.get(int(v), MISS_TINT)}"

    st.dataframe(
        matrix.style.map(tint).format(lambda v: "—" if pd.isna(v) else f"{int(v)}"),
        width="stretch",
        height=min(600, 38 * len(matrix) + 40),
    )

    st.divider()
    st.subheader("질문 하나 뜯어보기")

    qmap = qdf.drop_duplicates("qid").set_index("qid")["question"].to_dict()
    qid = st.selectbox(
        "질문", list(qmap), format_func=lambda q: f"{q} — {qmap.get(q, '')}"
    )

    st.info(qmap[qid])
    detail_by_model = {d["model"]: d for d in details if d["qid"] == qid}

    for m in order:
        d = detail_by_model.get(m)
        if not d:
            continue
        rank = next((i for i, t in enumerate(d["top5"], start=1) if t["is_gold"]), None)
        badge = f"{rank}등" if rank else "실패"
        with st.expander(f"**{m}** — 정답 {badge}", expanded=(m == order[0])):
            rows = []
            for i, t in enumerate(d["top5"], start=1):
                rows.append(
                    {
                        "순위": i,
                        "정답": "●" if t["is_gold"] else "",
                        "유사도": round(t["score"], 4),
                        "위치": t["locator"],
                        "섹션": t.get("섹션") or "",
                        # gold 청크는 정답 근거 문구 주변을, 나머지는 앞부분을 보여준다
                        "내용": t.get("match") or t["preview"],
                    }
                )
            st.dataframe(pd.DataFrame(rows).set_index("순위"),
                         width="stretch")


# ── 3. 오답 분석 ──────────────────────────────────────────────
with tab_misses:
    k = st.radio("기준", hit_cols, horizontal=True, index=0)

    miss = qdf[qdf[k] == 0]
    per_model = (
        miss.groupby("model").size().reindex(order, fill_value=0)
        .rename("틀린 문제 수").reset_index()
    )

    st.subheader(f"모델별 {k} 오답 수")
    mbars = (
        alt.Chart(per_model)
        .mark_bar(cornerRadiusEnd=4, size=20, color=p["series1"])
        .encode(
            y=alt.Y("model:N", sort=order, title=None),
            x=alt.X("틀린 문제 수:Q", title=None,
                    scale=alt.Scale(domain=[0, n_questions])),
            tooltip=["model:N", "틀린 문제 수:Q"],
        )
    )
    mlabels = mbars.mark_text(align="left", dx=6, fontSize=11).encode(
        text=alt.Text("틀린 문제 수:Q"), color=alt.value(p["muted"])
    )
    st.altair_chart(
        themed((mbars + mlabels).properties(height=42 * len(order) + 20), p),
        width="stretch",
    )

    st.subheader(f"모든 모델이 {k} 에서 틀린 문제")
    common = (
        miss.groupby("qid").size().pipe(lambda s: s[s == len(order)]).index.tolist()
    )
    if not common:
        st.success("없습니다. 모든 문제를 최소 한 모델은 맞혔습니다.")
    else:
        st.warning(
            f"{len(common)}개. **모델 성능 문제가 아닐 가능성이 높습니다** — "
            "질문 문장이나 `must_include` 라벨을 먼저 점검하세요."
        )
        st.dataframe(
            qdf[qdf["qid"].isin(common)][["qid", "question", "n_gold"]]
            .drop_duplicates("qid").set_index("qid"),
            width="stretch",
        )

    st.subheader("문제별 난이도")
    st.caption(f"{k} 기준으로 몇 개 모델이 틀렸는지. 위쪽이 어려운 문제입니다.")
    hardness = (
        miss.groupby("qid").size().rename("틀린 모델 수")
        .reset_index().sort_values("틀린 모델 수", ascending=False)
    )
    if hardness.empty:
        st.caption("오답이 없습니다.")
    else:
        hardness["question"] = hardness["qid"].map(qmap)
        st.dataframe(hardness.set_index("qid"), width="stretch")


# ── 4. 청크 탐색 ──────────────────────────────────────────────
with tab_chunks:
    if not chunks:
        st.info("`results/chunks.json` 이 없습니다. 벤치마크를 다시 돌리면 생성됩니다.")
    else:
        cdf = pd.json_normalize(chunks)
        cdf["글자수"] = cdf["text"].str.len()

        f0, f1, f2, f3 = st.columns(4)
        if "meta.lang" in cdf.columns:
            langs = [l for l in LANG_LABEL if l in set(cdf["meta.lang"].dropna())]
            lang_pick = f0.selectbox(
                "언어", ["전체"] + langs,
                format_func=lambda l: l if l == "전체" else f"{LANG_LABEL[l]} ({l})",
            )
        else:
            lang_pick = "전체"
        pool = cdf if lang_pick == "전체" else cdf[cdf["meta.lang"] == lang_pick]
        src = f1.selectbox("문서", ["전체"] + sorted(pool["source"].unique()))
        if "meta.doc_type" in cdf.columns:
            dtypes = sorted(x for x in cdf["meta.doc_type"].dropna().unique())
            dtype = f2.selectbox("종류", ["전체"] + dtypes)
        else:
            dtype = "전체"
        if "meta.섹션" in cdf.columns:
            secs = sorted(x for x in cdf["meta.섹션"].dropna().unique())
            sec = f3.selectbox("섹션", ["전체"] + secs)
        else:
            sec = "전체"

        view = pool
        if src != "전체":
            view = view[view["source"] == src]
        if dtype != "전체":
            view = view[view["meta.doc_type"] == dtype]
        if sec != "전체":
            view = view[view["meta.섹션"] == sec]

        st.caption(f"{len(view)} / {len(cdf)} 청크")
        cols = [c for c in ["id", "locator", "meta.섹션", "글자수", "text"] if c in view.columns]
        st.dataframe(
            view[cols].rename(columns={"meta.섹션": "섹션", "text": "본문"}).set_index("id"),
            width="stretch",
            height=520,
        )
