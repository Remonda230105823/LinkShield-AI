"""
LinkShield AI — Unit Tests for Data Cleaning Module
=====================================================

Tests cover: schema validation, whitespace handling, normalization,
same-label duplicate handling, conflicting-label quarantine, and
train/test leakage detection.

Author: Person 2 (Feature Engineering & Preprocessing)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_cleaning import (
    EXPECTED_COLUMNS,
    VALID_LABELS,
    coerce_url_to_string,
    detect_duplicates,
    handle_duplicates,
    handle_whitespace,
    normalize_url,
    run_cleaning_pipeline,
    validate_schema,
)


# -----------------------------------------------------------------------
# Schema validation
# -----------------------------------------------------------------------
class TestSchemaValidation:
    def test_valid_schema(self):
        df = pd.DataFrame({
            "url": ["https://example.com", "https://test.com"],
            "is_phishing": [0, 1],
        })
        report = validate_schema(df)
        assert report["expected_columns_found"] is True
        assert report["labels_valid"] is True
        assert report["missing_url_count"] == 0
        assert report["missing_label_count"] == 0

    def test_missing_column(self):
        df = pd.DataFrame({"url": ["https://example.com"]})
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_schema(df)

    def test_invalid_labels(self):
        df = pd.DataFrame({
            "url": ["https://a.com", "https://b.com"],
            "is_phishing": [0, 2],
        })
        report = validate_schema(df)
        assert report["invalid_label_count"] == 1
        assert report["labels_valid"] is False

    def test_unexpected_columns_warning(self):
        df = pd.DataFrame({
            "url": ["https://a.com"],
            "is_phishing": [0],
            "extra_col": [42],
        })
        report = validate_schema(df)
        assert "extra_col" in report["unexpected_columns"]

    def test_non_string_urls(self):
        df = pd.DataFrame({
            "url": [123, 456],
            "is_phishing": [0, 1],
        })
        report = validate_schema(df)
        assert report["non_string_url_count"] == 2


# -----------------------------------------------------------------------
# Whitespace handling
# -----------------------------------------------------------------------
class TestWhitespace:
    def test_leading_trailing_strip(self):
        df = pd.DataFrame({
            "url": ["  https://example.com  ", "https://test.com"],
            "is_phishing": [0, 1],
        })
        result, ws_report = handle_whitespace(df)
        assert result["url"].iloc[0] == "https://example.com"
        assert ws_report["leading_trailing_stripped"] == 1

    def test_empty_after_strip(self):
        df = pd.DataFrame({
            "url": ["   ", "https://test.com"],
            "is_phishing": [0, 1],
        })
        result, ws_report = handle_whitespace(df)
        assert ws_report["empty_after_strip"] == 1
        assert ws_report["whitespace_only"] == 1

    def test_no_modification_to_internal(self):
        """Internal spaces/chars should not be modified."""
        url = "https://example.com/path with spaces"
        df = pd.DataFrame({"url": [url], "is_phishing": [0]})
        result, _ = handle_whitespace(df)
        assert result["url"].iloc[0] == url


# -----------------------------------------------------------------------
# URL normalization
# -----------------------------------------------------------------------
class TestNormalization:
    def test_scheme_case(self):
        assert normalize_url("HTTP://Example.com") == "http://example.com"

    def test_hostname_case(self):
        assert normalize_url("https://EXAMPLE.COM/Path") == "https://example.com/Path"

    def test_default_port_http(self):
        assert normalize_url("http://example.com:80/p") == "http://example.com/p"

    def test_default_port_https(self):
        assert normalize_url("https://example.com:443/p") == "https://example.com/p"

    def test_non_default_port_preserved(self):
        result = normalize_url("http://example.com:8080/p")
        assert ":8080" in result

    def test_path_case_preserved(self):
        """Path case must be preserved (case-sensitive on servers)."""
        result = normalize_url("https://example.com/MyPath")
        assert "/MyPath" in result

    def test_query_preserved(self):
        result = normalize_url("https://example.com/p?Key=Val")
        assert "Key=Val" in result

    def test_no_scheme_url(self):
        """URL without scheme should normalize hostname but not add scheme."""
        result = normalize_url("Example.Com/path")
        assert result == "example.com/path"

    def test_whitespace_strip(self):
        assert normalize_url("  https://example.com  ") == "https://example.com"

    def test_deterministic(self):
        url = "HTTPS://WWW.Example.COM:443/Path?Q=1#Frag"
        assert normalize_url(url) == normalize_url(url)


# -----------------------------------------------------------------------
# 16. Same-label normalized duplicate handling
# -----------------------------------------------------------------------
class TestSameLabelDuplicates:
    def test_same_label_dedup(self):
        """Normalized duplicates with the same label → keep first."""
        df = pd.DataFrame({
            "url": [
                "https://example.com/path",
                "HTTPS://Example.COM/path",  # Same after normalization
                "https://other.com",
            ],
            "is_phishing": [0, 0, 1],
        })
        clean, quarantined, report = handle_duplicates(df)
        assert len(clean) == 2  # One representative + the other URL
        assert len(quarantined) == 0
        assert report["same_label_dups_removed"] == 1

    def test_keeps_first_occurrence(self):
        df = pd.DataFrame({
            "url": [
                "HTTP://example.com:80/path",
                "http://example.com/path",
            ],
            "is_phishing": [1, 1],
        })
        clean, _, _ = handle_duplicates(df)
        assert len(clean) == 1
        # Should keep the first row's URL
        assert clean["url"].iloc[0] == "HTTP://example.com:80/path"


# -----------------------------------------------------------------------
# 17. Conflicting-label normalized duplicate quarantine
# -----------------------------------------------------------------------
class TestConflictingLabels:
    def test_conflicting_quarantine(self):
        """Normalized duplicates with different labels → quarantine ALL."""
        df = pd.DataFrame({
            "url": [
                "https://example.com/path",
                "HTTPS://Example.COM/path",
                "https://safe.com",
            ],
            "is_phishing": [0, 1, 0],
        })
        clean, quarantined, report = handle_duplicates(df)
        # The conflicting pair should be quarantined
        assert len(quarantined) == 2
        assert report["conflicting_label_groups"] == 1
        # The safe URL should remain
        assert len(clean) == 1
        assert clean["url"].iloc[0] == "https://safe.com"

    def test_quarantine_reason_documented(self):
        df = pd.DataFrame({
            "url": ["http://a.com", "HTTP://A.COM"],
            "is_phishing": [0, 1],
        })
        _, quarantined, _ = handle_duplicates(df)
        assert "quarantine_reason" in quarantined.columns
        assert all(quarantined["quarantine_reason"].str.contains("conflicting"))


# -----------------------------------------------------------------------
# 18. Train/test duplicate leakage check
# -----------------------------------------------------------------------
class TestLeakageCheck:
    def test_no_exact_leakage(self):
        """After dedup, no exact URL should appear in both train and test."""
        from sklearn.model_selection import train_test_split

        df = pd.DataFrame({
            "url": [f"https://site{i}.com" for i in range(100)],
            "is_phishing": [0] * 50 + [1] * 50,
        })
        train, test = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["is_phishing"]
        )
        overlap = set(train["url"]) & set(test["url"])
        assert len(overlap) == 0

    def test_no_normalized_leakage(self):
        """After dedup, no normalized URL should appear in both splits."""
        from sklearn.model_selection import train_test_split

        # Unique URLs (no duplicates)
        df = pd.DataFrame({
            "url": [f"https://uniquesite{i}.com/p" for i in range(100)],
            "is_phishing": [0] * 50 + [1] * 50,
        })
        train, test = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["is_phishing"]
        )
        train_norm = set(train["url"].apply(normalize_url))
        test_norm = set(test["url"].apply(normalize_url))
        overlap = train_norm & test_norm
        assert len(overlap) == 0


# -----------------------------------------------------------------------
# Full pipeline test
# -----------------------------------------------------------------------
class TestCleaningPipeline:
    def test_full_pipeline(self):
        df = pd.DataFrame({
            "url": [
                "https://example.com",
                "HTTPS://EXAMPLE.COM",  # Normalized dup, same label
                "https://conflict.com",
                "HTTPS://Conflict.COM",  # Normalized dup, diff label
                "https://unique.com",
                "",  # Empty
            ],
            "is_phishing": [0, 0, 0, 1, 1, 0],
        })
        clean, quarantined, report = run_cleaning_pipeline(df)

        # Empty URL removed
        assert report["stages"]["invalid_removal"]["empty_urls"] == 1

        # Conflicting pair quarantined
        assert len(quarantined) == 2

        # Clean should have 2: one from example.com group + unique.com
        assert len(clean) == 2
        assert set(clean.columns) == {"url", "is_phishing"}


# =========================================================================
# Original label conversion (UCI label -> is_phishing)
# =========================================================================

class TestLabelConversion:
    def test_original_label_is_inverted(self):
        from src.data_cleaning import convert_original_label
        original = pd.Series([1, 0, 1, 0])          # 1 = legitimate, 0 = phishing
        converted = convert_original_label(original)
        assert converted.tolist() == [0, 1, 0, 1]   # 1 = phishing, 0 = legitimate
        assert converted.name == "is_phishing"

    def test_non_binary_label_rejected(self):
        from src.data_cleaning import convert_original_label
        with pytest.raises(ValueError):
            convert_original_label(pd.Series([0, 1, 2]))

    def test_missing_label_rejected(self):
        from src.data_cleaning import convert_original_label
        with pytest.raises(ValueError):
            convert_original_label(pd.Series([0, 1, None]))
