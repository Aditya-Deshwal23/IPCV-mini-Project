"""Evaluation result browser and small reproducible rerun.

Syllabus: Unit 7 - evaluation and reproducibility
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from docuvision.evaluation.run_eval import run_evaluation
from docuvision.ui.common import syllabus_info


def render() -> None:
    """Show measured evaluation tables, charts, report, and rerun control.

    Args:
        None.

    Returns:
        None.

    Syllabus: Unit 7 - evaluation and reproducibility
    """
    syllabus_info("PSNR, SSIM, OCR error, ablation, compression, reproducibility", "Evaluation")
    results_dir = Path("results")
    if st.button("Re-run a small evaluation (N=2)", key="evaluation_rerun"):
        try:
            with st.spinner("Generating synthetic pages and measuring methods..."):
                generated = run_evaluation(
                    n=2,
                    dataset_dir=Path("data/evaluation-smoke"),
                    results_dir=Path("results/small-run"),
                    seed=2026,
                )
            st.session_state["evaluation_generated"] = {
                key: str(path) for key, path in generated.items()
            }
            st.success("Small evaluation finished; generated CSV and chart artifacts are saved.")
        except (OSError, ValueError, RuntimeError) as error:
            st.error(f"Evaluation failed: {error}")

    generated_paths = st.session_state.get("evaluation_generated", {})
    if generated_paths:
        st.caption("Most recent small-run artifacts")
        st.json(generated_paths)
    if not results_dir.exists():
        st.info("No results directory exists yet. Run the small evaluation to generate measured artifacts.")
        return
    summary_paths = sorted(results_dir.rglob("summary.md"))
    for summary_path in summary_paths:
        st.subheader("Generated summary")
        st.caption(summary_path.relative_to(results_dir).as_posix())
        st.markdown(summary_path.read_text(encoding="utf-8"))
    csv_paths = sorted(results_dir.rglob("*.csv"))
    if not csv_paths:
        st.info("No root-level evaluation CSVs found yet.")
    for csv_path in csv_paths:
        st.subheader(csv_path.stem.replace("_", " ").title())
        st.caption(csv_path.relative_to(results_dir).as_posix())
        try:
            frame = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
            st.error(f"Could not load {csv_path.name}: {error}")
            continue
        st.dataframe(frame, hide_index=True, use_container_width=True)
        numeric = frame.select_dtypes(include="number")
        if not numeric.empty:
            st.bar_chart(numeric)
    chart_paths = sorted(results_dir.rglob("*.png"))
    if chart_paths:
        st.subheader("Evaluation charts")
        for chart_path in chart_paths:
            st.image(str(chart_path), caption=chart_path.name, use_container_width=True)
