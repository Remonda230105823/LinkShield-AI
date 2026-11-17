"""
LinkShield AI — Data Cleaning Module
=====================================

Person 2 deliverable: input validation, whitespace handling,
URL normalization for duplicate detection, duplicate/conflict handling,
and cleaning report generation.

Input contract (agreed with Person 1 and Person 3)
--------------------------------------------------
The cleaned modelling data has exactly two columns:

- ``url``          : raw URL string.
- ``is_phishing``  : integer target, 1 = phishing, 0 = legitimate.

The original UCI ``label`` column is the opposite (1 = legitimate,
0 = phishing); ``convert_original_label`` performs the conversion. Every
other UCI column, including the precomputed features and ``FILENAME``, is
excluded from the model input.

This module treats every URL as an inert text string.
No URL is visited, resolved, or accessed over the network.

Author: Person 2 (Feature Engineering & Preprocessing)
"""

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urlunparse

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_COLUMNS = {"url", "is_phishing"}
VALID_LABELS = {0, 1}
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Label conversion (original UCI label -> project label)
# ---------------------------------------------------------------------------

def convert_original_label(original_label: pd.Series) -> pd.Series:
    """
    Convert the original UCI ``label`` to the project target ``is_phishing``.

    Original UCI mapping : 1 = legitimate, 0 = phishing.
    Project mapping      : 1 = phishing,   0 = legitimate.

    Raises ``ValueError`` if any value is missing or not exactly 0 or 1, so
    the mapping can never be applied silently to unexpected data.
    """
    if original_label.isna().any():
        raise ValueError("Original label contains missing values.")
    if not set(original_label.unique()) <= VALID_LABELS:
        raise ValueError(
            f"Original label must be binary {{0, 1}}, got "
            f"{sorted(set(original_label.unique()))}"
        )
    return (1 - original_label.astype(int)).rename("is_phishing")


# ---------------------------------------------------------------------------
# A. Schema Validation
# ---------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate that the DataFrame has exactly the expected modelling columns
    (``url`` and ``is_phishing``), that labels are binary integers in {0, 1},
    and that URL values are strings.

    Returns a report dict with validation results.
    """
    report: Dict[str, Any] = {
        "columns_present": list(df.columns),
        "expected_columns_found": False,
        "unexpected_columns": [],
        "label_dtype": str(df["is_phishing"].dtype) if "is_phishing" in df.columns else None,
        "labels_valid": False,
        "url_dtype": str(df["url"].dtype) if "url" in df.columns else None,
        "urls_are_strings": False,
        "missing_url_count": 0,
        "missing_label_count": 0,
        "invalid_label_count": 0,
        "non_string_url_count": 0,
        "total_rows": len(df),
    }

    # Check expected columns
    present = set(df.columns)
    report["expected_columns_found"] = EXPECTED_COLUMNS.issubset(present)
    report["unexpected_columns"] = sorted(present - EXPECTED_COLUMNS)

    if not report["expected_columns_found"]:
        missing = EXPECTED_COLUMNS - present
        raise ValueError(f"Missing required columns: {missing}")

    # Check for unexpected modelling columns (warn, don't fail)
    if report["unexpected_columns"]:
        logger.warning(
            "Unexpected columns found (will be ignored for modelling): %s",
            report["unexpected_columns"],
        )

    # Missing values
    report["missing_url_count"] = int(df["url"].isna().sum())
    report["missing_label_count"] = int(df["is_phishing"].isna().sum())

    # Validate labels are binary integers
    non_null_labels = df["is_phishing"].dropna()
    invalid_mask = ~non_null_labels.isin(VALID_LABELS)
    report["invalid_label_count"] = int(invalid_mask.sum())
    report["labels_valid"] = (
        report["invalid_label_count"] == 0
        and report["missing_label_count"] == 0
    )

    # Validate URL values are strings
    url_series = df["url"].dropna()
    non_string_mask = ~url_series.apply(lambda x: isinstance(x, str))
    report["non_string_url_count"] = int(non_string_mask.sum())
    report["urls_are_strings"] = report["non_string_url_count"] == 0

    return report


def coerce_url_to_string(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Convert non-string URL values to strings using ``str()``.

    Returns (modified DataFrame copy, count of coerced values).

    Documented rule: any non-null, non-string URL value is converted to its
    Python ``str()`` representation. Null/NaN values remain as NaN.
    """
    df = df.copy()
    non_null = df["url"].notna()
    non_string = non_null & ~df["url"].apply(lambda x: isinstance(x, str))
    count = int(non_string.sum())
    if count > 0:
        df.loc[non_string, "url"] = df.loc[non_string, "url"].apply(str)
        logger.info("Coerced %d non-string URL values to strings.", count)
    return df, count


# ---------------------------------------------------------------------------
# B. Whitespace Handling
# ---------------------------------------------------------------------------

def handle_whitespace(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Strip leading and trailing whitespace from URLs.
    Identify empty or whitespace-only URLs after stripping.

    Does NOT modify meaningful internal URL characters (spaces, tabs inside
    the URL are preserved as-is).

    Returns (modified DataFrame copy, whitespace report dict).
    """
    df = df.copy()
    ws_report: Dict[str, int] = {
        "leading_trailing_stripped": 0,
        "empty_after_strip": 0,
        "whitespace_only": 0,
    }

    original_urls = df["url"].copy()
    stripped = df["url"].astype(str).str.strip()

    # Count URLs where stripping changed the value
    changed = (original_urls.astype(str) != stripped) & original_urls.notna()
    ws_report["leading_trailing_stripped"] = int(changed.sum())

    # Identify empty / whitespace-only
    empty_mask = stripped.eq("")
    ws_report["empty_after_strip"] = int(empty_mask.sum())

    # Whitespace-only = non-empty original that became empty after strip
    ws_only = empty_mask & original_urls.astype(str).str.len().gt(0)
    ws_report["whitespace_only"] = int(ws_only.sum())

    df["url"] = stripped

    return df, ws_report


# ---------------------------------------------------------------------------
# C. URL Normalization for Duplicate Detection
# ---------------------------------------------------------------------------

def normalize_url(url: str) -> str:
    """
    Conservative URL normalization used **only** for duplicate detection.

    Normalization steps (documented):
    1. Strip leading/trailing whitespace.
    2. Parse with ``urllib.parse.urlparse``.
    3. Lowercase the **scheme** (e.g., ``HTTP`` → ``http``).
    4. Lowercase the **hostname** (domain names are case-insensitive per RFC).
    5. Remove default ports: port 80 for ``http``, port 443 for ``https``.
    6. Preserve path, query, and fragment **as-is** (case-sensitive).
       - Paths and query values may be case-sensitive on many servers.
    7. Reassemble the URL.

    What this does NOT do:
    - Does not lowercase the entire URL (path/query may be case-sensitive).
    - Does not remove trailing slashes (``/path/`` != ``/path``).
    - Does not sort query parameters.
    - Does not decode percent-encoded characters.
    - Does not resolve relative URLs.
    - Does not strip ``www.`` prefix (``www.example.com`` != ``example.com``).

    Parameters
    ----------
    url : str
        A raw URL string.

    Returns
    -------
    str
        The normalized URL string.
    """
    url = url.strip()

    # If no scheme, prepend http:// for parsing only so urlparse works
    temp_url = url
    has_original_scheme = bool(re.match(r'^[a-zA-Z][a-zA-Z0-9+\-.]*://', url))
    if not has_original_scheme:
        temp_url = "http://" + url

    parsed = urlparse(temp_url)

    # Lowercase scheme
    scheme = parsed.scheme.lower()

    # Lowercase hostname
    hostname = parsed.hostname or ""
    hostname = hostname.lower()

    # Handle port: remove default ports
    port = parsed.port
    if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
        port = None

    # Reconstruct netloc
    netloc = hostname
    if parsed.username:
        userinfo = parsed.username
        if parsed.password:
            userinfo += ":" + parsed.password
        netloc = userinfo + "@" + netloc
    if port is not None:
        netloc += ":" + str(port)

    # Preserve path, params, query, fragment as-is
    normalized = urlunparse((
        scheme,
        netloc,
        parsed.path,
        parsed.params,
        parsed.query,
        parsed.fragment,
    ))

    # If original had no scheme, remove the prepended one
    if not has_original_scheme:
        normalized = re.sub(r'^http://', '', normalized, count=1)

    return normalized


# ---------------------------------------------------------------------------
# D. Duplicate Handling
# ---------------------------------------------------------------------------

def detect_duplicates(
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Detect exact and normalized duplicate URL groups.

    Returns a dict containing:
    - ``exact_duplicate_count``: number of duplicate rows (not including first)
    - ``normalized_groups``: number of normalized URL groups with >1 member
    - ``conflicting_label_groups``: number of normalized groups with both labels
    - ``conflicting_row_count``: total rows in conflicting groups
    - ``same_label_dup_rows``: rows that are non-first in same-label groups
    """
    report: Dict[str, Any] = {}

    # Exact duplicates (count of rows beyond the first occurrence)
    report["exact_duplicate_count"] = int(df.duplicated(subset=["url"], keep="first").sum())

    # Normalized duplicates
    df = df.copy()
    df["_normalized_url"] = df["url"].apply(normalize_url)

    # Group by normalized URL
    grouped = df.groupby("_normalized_url")

    # Find groups with more than one member
    group_sizes = grouped.size()
    dup_groups = group_sizes[group_sizes > 1]
    report["normalized_dup_groups"] = len(dup_groups)
    report["normalized_dup_total_rows"] = int(dup_groups.sum())

    # Find conflicting-label groups
    label_nunique = grouped["is_phishing"].nunique()
    conflicting = label_nunique[label_nunique > 1]
    report["conflicting_label_groups"] = len(conflicting)

    # Count total rows in conflicting groups
    if len(conflicting) > 0:
        conflicting_urls = set(conflicting.index)
        conflicting_mask = df["_normalized_url"].isin(conflicting_urls)
        report["conflicting_row_count"] = int(conflicting_mask.sum())
    else:
        report["conflicting_row_count"] = 0

    # Same-label duplicates: rows beyond the first in groups with uniform label
    same_label_groups = label_nunique[label_nunique == 1].index
    same_label_mask = df["_normalized_url"].isin(set(same_label_groups))
    same_label_df = df[same_label_mask]
    same_label_dup_rows = int(
        same_label_df.duplicated(subset=["_normalized_url"], keep="first").sum()
    )
    report["same_label_dup_rows"] = same_label_dup_rows

    return report


def handle_duplicates(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Handle duplicates according to the project rules:

    1. If normalized duplicates have the same label → retain one deterministic
       representative (first occurrence in the original index order).
    2. If a normalized URL group has conflicting labels → quarantine the
       **entire** ambiguous group.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame with ``url`` and ``is_phishing`` columns.

    Returns
    -------
    clean_df : pd.DataFrame
        Deduplicated DataFrame with quarantined rows removed.
    quarantined_df : pd.DataFrame
        DataFrame of quarantined rows with a ``quarantine_reason`` column.
    dup_report : dict
        Duplicate handling statistics.
    """
    df = df.copy()
    df["_normalized_url"] = df["url"].apply(normalize_url)

    # Identify conflicting-label groups
    label_nunique = df.groupby("_normalized_url")["is_phishing"].nunique()
    conflicting_urls = set(label_nunique[label_nunique > 1].index)

    dup_report: Dict[str, Any] = {
        "conflicting_label_groups": len(conflicting_urls),
    }

    # Quarantine conflicting groups
    conflicting_mask = df["_normalized_url"].isin(conflicting_urls)
    quarantined = df[conflicting_mask].copy()
    quarantined["quarantine_reason"] = "conflicting_labels_in_normalized_group"
    dup_report["quarantined_rows"] = len(quarantined)

    # Keep non-conflicting rows
    clean = df[~conflicting_mask].copy()

    # Deduplicate same-label normalized groups: keep first occurrence
    before_dedup = len(clean)
    clean = clean.drop_duplicates(subset=["_normalized_url"], keep="first")
    dup_report["same_label_dups_removed"] = before_dedup - len(clean)
    dup_report["clean_rows_after_dedup"] = len(clean)

    # Drop helper column
    clean = clean.drop(columns=["_normalized_url"])
    quarantined = quarantined.drop(columns=["_normalized_url"])

    return clean, quarantined, dup_report


# ---------------------------------------------------------------------------
# E. Cleaning Report & Full Pipeline
# ---------------------------------------------------------------------------

def run_cleaning_pipeline(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Run the full cleaning pipeline and produce a comprehensive report.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with ``url`` and ``is_phishing`` columns.

    Returns
    -------
    clean_df : pd.DataFrame
        Final cleaned DataFrame ready for splitting.
    quarantined_df : pd.DataFrame
        Quarantined rows with reasons.
    report : dict
        Comprehensive cleaning report with row counts at every stage.
    """
    report: Dict[str, Any] = {"stages": {}}
    report["original_rows"] = len(df)
    report["original_class_counts"] = df["is_phishing"].value_counts().to_dict()

    # --- Stage 1: Schema validation ---
    schema_report = validate_schema(df)
    report["stages"]["schema_validation"] = schema_report

    # --- Stage 2: Coerce non-string URLs ---
    df, coerced_count = coerce_url_to_string(df)
    report["stages"]["url_coercion"] = {"coerced_count": coerced_count}

    # --- Stage 3: Whitespace handling ---
    df, ws_report = handle_whitespace(df)
    report["stages"]["whitespace"] = ws_report

    # --- Stage 4: Remove missing/empty/invalid ---
    # Remove rows with missing URLs
    missing_url_mask = df["url"].isna()
    missing_url_count = int(missing_url_mask.sum())

    # Remove rows with empty URLs (after stripping)
    empty_url_mask = df["url"].eq("")
    empty_url_count = int(empty_url_mask.sum())

    # Remove rows with missing labels
    missing_label_mask = df["is_phishing"].isna()
    missing_label_count = int(missing_label_mask.sum())

    # Remove rows with invalid labels
    invalid_label_mask = ~df["is_phishing"].isin(VALID_LABELS) & ~missing_label_mask
    invalid_label_count = int(invalid_label_mask.sum())

    remove_mask = missing_url_mask | empty_url_mask | missing_label_mask | invalid_label_mask
    invalid_total = int(remove_mask.sum())

    df = df[~remove_mask].copy()

    report["stages"]["invalid_removal"] = {
        "missing_urls": missing_url_count,
        "empty_urls": empty_url_count,
        "missing_labels": missing_label_count,
        "invalid_labels": invalid_label_count,
        "total_removed": invalid_total,
        "rows_after": len(df),
    }

    # --- Stage 5: Duplicate detection (pre-handling stats) ---
    dup_detection = detect_duplicates(df)
    report["stages"]["duplicate_detection"] = dup_detection

    # --- Stage 6: Duplicate handling ---
    clean_df, quarantined_df, dup_handling = handle_duplicates(df)
    report["stages"]["duplicate_handling"] = dup_handling

    # --- Final counts ---
    report["final_clean_rows"] = len(clean_df)
    report["quarantined_rows"] = len(quarantined_df)
    final_counts = clean_df["is_phishing"].value_counts().to_dict()
    report["final_class_counts"] = {
        "legitimate (0)": final_counts.get(0, 0),
        "phishing (1)": final_counts.get(1, 0),
    }

    # Keep only modelling columns
    clean_df = clean_df[["url", "is_phishing"]].reset_index(drop=True)
    quarantined_df = quarantined_df.reset_index(drop=True)

    return clean_df, quarantined_df, report


def save_cleaning_outputs(
    clean_df: pd.DataFrame,
    quarantined_df: pd.DataFrame,
    report: Dict[str, Any],
    output_dir: Path,
    reports_dir: Path,
) -> None:
    """Save cleaned data, quarantined data, and cleaning report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save cleaned URLs
    clean_path = output_dir / "cleaned_urls.csv"
    clean_df.to_csv(clean_path, index=False)
    logger.info("Saved %d cleaned URLs to %s", len(clean_df), clean_path)

    # Save quarantined rows
    if len(quarantined_df) > 0:
        quarantine_path = output_dir / "quarantined_urls.csv"
        quarantined_df.to_csv(quarantine_path, index=False)
        logger.info(
            "Saved %d quarantined URLs to %s", len(quarantined_df), quarantine_path
        )

    # Save report
    report_path = reports_dir / "data_quality_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved cleaning report to %s", report_path)
