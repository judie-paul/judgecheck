"""Streamlit results explorer. Run with ``make app`` or ``streamlit run src/judgecheck/app.py``."""

import os
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from judgecheck.explore import (
    FILTERS,
    bias_rows,
    drilldown,
    find_studies,
    load_study,
    summary_rows,
    verdict_label,
)

st.set_page_config(page_title="JudgeCheck explorer", layout="wide")


def _bars(frame: pd.DataFrame, value: str, low: str, high: str, title: str) -> alt.LayerChart:
    base = alt.Chart(frame).encode(y=alt.Y("judge:N", sort=None, title=None))
    bars = base.mark_bar().encode(x=alt.X(f"{value}:Q", title=title))
    errors = base.mark_errorbar().encode(x=alt.X(f"{low}:Q", title=title), x2=alt.X2(f"{high}:Q"))
    return alt.LayerChart(layer=[bars, errors])


root = Path(os.environ.get("JUDGECHECK_REPORTS", "reports"))
studies = find_studies(root)
st.title("JudgeCheck: can LLM judges be trusted?")
if not studies:
    st.info(f"No finished runs under `{root}`. Run `make pipeline` first.")
    st.stop()

choice = st.sidebar.selectbox("Run", studies, format_func=lambda path: path.name or str(path))
study = load_study(choice)
report = study.report
st.caption(
    f"Dataset revision `{report.dataset_revision[:8]}` · {report.comparisons} comparisons · "
    f"seed {report.seed} · 95% bootstrap intervals resampled by comparison"
)
st.sidebar.markdown(
    f"**Expert agreement ceiling**  \n"
    f"S1 {report.human.agreement_s1.value:.3f} · S2 {report.human.agreement_s2.value:.3f}"
    if report.human.agreement_s1.value is not None and report.human.agreement_s2.value is not None
    else "**Expert agreement ceiling**  \nundefined (no comparison has two votes)"
)

overview, position, verbosity, explorer = st.tabs(
    ["Agreement", "Position bias", "Verbosity bias", "Comparisons"]
)

with overview:
    rows = pd.DataFrame(summary_rows(study))
    st.subheader("Agreement with experts")
    st.caption("Consensus of both display orders. S1 counts ties, S2 drops them.")
    for metric in ("S1", "S2"):
        shown = rows.dropna(subset=[metric])
        if shown.empty:
            st.write(f"{metric}: undefined for every judge.")
        else:
            st.altair_chart(
                _bars(shown, metric, f"{metric} low", f"{metric} high", metric),
                use_container_width=True,
            )
    st.dataframe(rows, hide_index=True, use_container_width=True)

bias = pd.DataFrame(bias_rows(study))

with position:
    st.subheader("Does the verdict survive swapping the answers?")
    st.caption("A first-position pick rate of 0.5 is fair; 1.0 means always the first answer.")
    shown = bias.dropna(subset=["first-position pick rate"])
    if not shown.empty:
        st.altair_chart(
            _bars(
                shown,
                "first-position pick rate",
                "first-position pick rate low",
                "first-position pick rate high",
                "first-position pick rate",
            ),
            use_container_width=True,
        )
    st.dataframe(
        bias[["judge", "order-consistent", "always first", "always second"]],
        hide_index=True,
        use_container_width=True,
    )

with verbosity:
    st.subheader("Does the judge favor the longer answer?")
    st.caption("Gap = judge's longer-pick rate minus the experts' on the same comparisons.")
    shown = bias.dropna(subset=["longer gap"])
    if not shown.empty:
        st.altair_chart(
            _bars(shown, "longer gap", "longer gap low", "longer gap high", "longer gap"),
            use_container_width=True,
        )
    st.dataframe(
        bias[["judge", "judge picks longer", "experts pick longer", "longer gap"]],
        hide_index=True,
        use_container_width=True,
    )

with explorer:
    judge_name = st.selectbox("Judge", [judge.judge for judge in report.judges])
    filter_name = st.selectbox("Show", list(FILTERS), format_func=FILTERS.__getitem__)
    items = drilldown(study, judge_name, filter_name)
    st.write(f"{len(items)} comparisons")
    if items:
        item = st.selectbox("Comparison", items, format_func=lambda c: c.id)
        record = study.records[judge_name][item.id]
        left, right = st.columns(2)
        left.markdown("**Judge**")
        left.write(f"Model a shown first: {verdict_label(item, record.ab, 'ab')}")
        left.write(f"Model b shown first: {verdict_label(item, record.ba, 'ba')}")
        left.write(f"Consensus: {record.consensus or 'undefined'}")
        right.markdown("**Experts**")
        for vote in item.votes:
            label = {"a": item.model_a, "b": item.model_b, "tie": "tie"}[vote.verdict]
            right.write(f"{vote.judge}: {label}")
        for column, model, conversation in (
            (left, item.model_a, item.conversation_a),
            (right, item.model_b, item.conversation_b),
        ):
            with column.expander(f"{model}: conversation", expanded=True):
                for message in conversation:
                    st.markdown(f"**{message.role}**")
                    st.text(message.content)
