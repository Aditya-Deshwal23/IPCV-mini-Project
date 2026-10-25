"""Tests for keeping generated evaluation tables synchronized in documentation.

Syllabus: Unit 7 - evaluation and reproducibility
"""

from pathlib import Path

import pytest

from scripts.sync_results_docs import sync_summary


def test_sync_summary_copies_generated_content_and_is_idempotent(tmp_path: Path) -> None:
    """Verify the source summary is copied exactly and repeated sync is stable.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        None.

    Syllabus: Unit 7 - evaluation and reproducibility
    """
    results_directory = tmp_path / "results"
    results_directory.mkdir()
    summary = results_directory / "summary.md"
    document = tmp_path / "README.md"
    summary.write_text(
        "# Generated\n\n| measured | value |\n| --- | --- |\n\n"
        "![plot](chart.png)\n",
        encoding="utf-8",
    )
    document.write_text(
        "Before\n<!-- BEGIN GENERATED EVALUATION SUMMARY -->\n"
        "_placeholder_\n<!-- END GENERATED EVALUATION SUMMARY -->\nAfter\n",
        encoding="utf-8",
    )

    sync_summary(summary, (document,))
    first_output = document.read_text(encoding="utf-8")
    sync_summary(summary, (document,))

    assert "# Generated" in first_output
    assert "| measured | value |" in first_output
    assert "![plot](results/chart.png)" in first_output
    assert document.read_text(encoding="utf-8") == first_output


def test_sync_summary_requires_generated_input(tmp_path: Path) -> None:
    """Report a missing evaluator summary instead of inserting fake results.

    Args:
        tmp_path: Pytest temporary directory fixture.

    Returns:
        None.

    Syllabus: Unit 7 - evaluation and reproducibility
    """
    document = tmp_path / "README.md"
    document.write_text(
        "<!-- BEGIN GENERATED EVALUATION SUMMARY -->\n"
        "<!-- END GENERATED EVALUATION SUMMARY -->",
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError, match="run the evaluator first"):
        sync_summary(tmp_path / "missing.md", (document,))
