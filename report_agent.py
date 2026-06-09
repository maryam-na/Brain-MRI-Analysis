from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


PATHOLOGY_TERMS = ("pathology", "atrophy", "dementia", "impairment", "abnormal", "severe", "moderate", "mild")


def generate_report(z_scores: pd.Series | dict) -> str:
    values = pd.Series(z_scores, dtype="float64")
    min_z = values.min()
    if min_z < -2:
        severity = "severe"
        finding = "severe volumetric abnormality"
    elif min_z < -1:
        severity = "mild to moderate"
        finding = "mild to moderate volumetric reduction"
    else:
        severity = "normal"
        finding = "measurements are within expected limits"

    return (
        f"Clinical MRI volumetric summary: {finding}. "
        f"Overall pattern is classified as {severity} relative to the cohort reference."
    )


def _contains_any(text: str, terms: Iterable[str]) -> bool:
    return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)


def check_report_consistency(z_scores, report_text: str) -> dict:
    """Validate report wording against z-score severity thresholds."""
    values = pd.Series(z_scores, dtype="float64").dropna()
    report = report_text.lower()
    min_z = values.min()
    issues = []

    if min_z < -2:
        if "severe" not in report:
            issues.append("Z < -2 requires 'severe' language in the report.")
    elif -2 < min_z < -1:
        if not _contains_any(report, ("moderate", "mild")):
            issues.append("-2 < Z < -1 requires 'moderate' or 'mild' language in the report.")
        if "severe" in report:
            issues.append("Mild/moderate z-score range should not be described as severe.")
    else:
        if _contains_any(report, PATHOLOGY_TERMS):
            issues.append("Normal z-score range should not include pathology language.")

    return {
        "consistent": len(issues) == 0,
        "min_z": float(min_z),
        "issues": issues,
    }
