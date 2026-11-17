"""
LinkShield AI — Unit Tests for URLFeatureExtractor and Preprocessing
=====================================================================

Tests cover: normal URLs, edge cases, consistency, and preprocessing
with unseen TLDs.  No URL is visited during testing.

Author: Person 2 (Feature Engineering & Preprocessing)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.url_features import (
    FEATURE_NAMES,
    SUSPICIOUS_KEYWORDS,
    URLFeatureExtractor,
    _is_ip_address,
    _shannon_entropy,
    _extract_tld,
)
from src.preprocessing import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
)


@pytest.fixture
def extractor():
    return URLFeatureExtractor()


# -----------------------------------------------------------------------
# 1. Normal HTTPS URL
# -----------------------------------------------------------------------
class TestNormalHTTPS:
    def test_basic_https(self, extractor):
        urls = pd.Series(["https://www.example.com/path?q=1"])
        result = extractor.fit_transform(urls)
        assert result["has_https"].iloc[0] == 1
        assert result["has_www"].iloc[0] == 1
        assert result["has_query"].iloc[0] == 1
        assert result["tld"].iloc[0] == "com"
        assert result["url_length"].iloc[0] == len("https://www.example.com/path?q=1")


# -----------------------------------------------------------------------
# 2. HTTP URL
# -----------------------------------------------------------------------
class TestHTTPURL:
    def test_basic_http(self, extractor):
        urls = pd.Series(["http://example.com/page"])
        result = extractor.fit_transform(urls)
        assert result["has_https"].iloc[0] == 0
        assert result["tld"].iloc[0] == "com"


# -----------------------------------------------------------------------
# 3. URL containing @
# -----------------------------------------------------------------------
class TestAtSymbol:
    def test_url_with_at(self, extractor):
        urls = pd.Series(["http://user@evil.com/path"])
        result = extractor.fit_transform(urls)
        assert result["num_at_symbols"].iloc[0] >= 1


# -----------------------------------------------------------------------
# 4. URL with multiple subdomains
# -----------------------------------------------------------------------
class TestMultipleSubdomains:
    def test_subdomains(self, extractor):
        urls = pd.Series(["https://a.b.c.example.com/"])
        result = extractor.fit_transform(urls)
        # a.b.c.example.com = 5 parts => 3 subdomains
        assert result["num_subdomains"].iloc[0] == 3


# -----------------------------------------------------------------------
# 5. URL with IP hostname
# -----------------------------------------------------------------------
class TestIPHostname:
    def test_ipv4(self, extractor):
        urls = pd.Series(["http://192.168.1.1/admin"])
        result = extractor.fit_transform(urls)
        assert result["has_ip"].iloc[0] == 1
        assert result["num_subdomains"].iloc[0] == 0
        assert result["tld"].iloc[0] == ""

    def test_ipv6(self, extractor):
        urls = pd.Series(["http://[::1]/admin"])
        result = extractor.fit_transform(urls)
        assert result["has_ip"].iloc[0] == 1


# -----------------------------------------------------------------------
# 6. URL with query string
# -----------------------------------------------------------------------
class TestQueryString:
    def test_query(self, extractor):
        urls = pd.Series(["https://example.com/search?q=test&page=2"])
        result = extractor.fit_transform(urls)
        assert result["has_query"].iloc[0] == 1
        assert result["num_ampersands"].iloc[0] >= 1
        assert result["num_equal_signs"].iloc[0] >= 2
        assert result["query_length"].iloc[0] > 0


# -----------------------------------------------------------------------
# 7. URL with fragment
# -----------------------------------------------------------------------
class TestFragment:
    def test_fragment(self, extractor):
        urls = pd.Series(["https://example.com/page#section1"])
        result = extractor.fit_transform(urls)
        assert result["has_fragment"].iloc[0] == 1
        assert result["fragment_length"].iloc[0] > 0


# -----------------------------------------------------------------------
# 8. URL with no explicit scheme
# -----------------------------------------------------------------------
class TestNoScheme:
    def test_no_scheme(self, extractor):
        urls = pd.Series(["www.example.com/path"])
        result = extractor.fit_transform(urls)
        assert result["has_https"].iloc[0] == 0
        assert result["tld"].iloc[0] == "com"
        assert result["domain_length"].iloc[0] > 0


# -----------------------------------------------------------------------
# 9. Very long URL
# -----------------------------------------------------------------------
class TestLongURL:
    def test_very_long(self, extractor):
        long_url = "https://example.com/" + "a" * 5000
        urls = pd.Series([long_url])
        result = extractor.fit_transform(urls)
        assert result["url_length"].iloc[0] == len(long_url)
        assert result["path_length"].iloc[0] > 5000


# -----------------------------------------------------------------------
# 10. Empty or missing input
# -----------------------------------------------------------------------
class TestEmptyMissing:
    def test_empty_string(self, extractor):
        urls = pd.Series([""])
        result = extractor.fit_transform(urls)
        assert result["url_length"].iloc[0] == 0
        assert result["tld"].iloc[0] == ""
        assert len(result) == 1

    def test_none_value(self, extractor):
        urls = pd.Series([None])
        result = extractor.fit_transform(urls)
        assert result["url_length"].iloc[0] == 0
        assert len(result) == 1

    def test_nan_value(self, extractor):
        urls = pd.Series([np.nan])
        result = extractor.fit_transform(urls)
        assert result["url_length"].iloc[0] == 0
        assert len(result) == 1


# -----------------------------------------------------------------------
# 11. Malformed URL
# -----------------------------------------------------------------------
class TestMalformedURL:
    def test_just_text(self, extractor):
        urls = pd.Series(["not-a-real-url"])
        result = extractor.fit_transform(urls)
        # Should not crash
        assert result["url_length"].iloc[0] == len("not-a-real-url")

    def test_only_scheme(self, extractor):
        urls = pd.Series(["https://"])
        result = extractor.fit_transform(urls)
        assert result["url_length"].iloc[0] > 0

    def test_random_characters(self, extractor):
        urls = pd.Series(["!@#$%^&*()"])
        result = extractor.fit_transform(urls)
        assert result["num_special_chars"].iloc[0] > 0


# -----------------------------------------------------------------------
# 12. Unknown TLD
# -----------------------------------------------------------------------
class TestUnknownTLD:
    def test_unusual_tld(self, extractor):
        urls = pd.Series(["https://example.zzzzz/path"])
        result = extractor.fit_transform(urls)
        assert result["tld"].iloc[0] == "zzzzz"


# -----------------------------------------------------------------------
# 13. Consistent feature column names and order
# -----------------------------------------------------------------------
class TestConsistentColumns:
    def test_column_names(self, extractor):
        urls = pd.Series(["https://example.com", "http://test.org/p"])
        result = extractor.fit_transform(urls)
        assert list(result.columns) == FEATURE_NAMES

    def test_column_order_single(self, extractor):
        urls = pd.Series(["https://a.com"])
        result = extractor.fit_transform(urls)
        assert list(result.columns) == FEATURE_NAMES

    def test_feature_names_out(self, extractor):
        assert extractor.get_feature_names_out() == FEATURE_NAMES


# -----------------------------------------------------------------------
# 14. Consistent output types
# -----------------------------------------------------------------------
class TestOutputTypes:
    def test_numeric_types(self, extractor):
        urls = pd.Series(["https://example.com/path?q=1"])
        result = extractor.fit_transform(urls)
        for col in NUMERICAL_FEATURES:
            assert np.issubdtype(result[col].dtype, np.number), (
                f"{col} is not numeric: {result[col].dtype}"
            )

    def test_tld_is_string(self, extractor):
        urls = pd.Series(["https://example.com"])
        result = extractor.fit_transform(urls)
        assert pd.api.types.is_string_dtype(result["tld"]) or result["tld"].dtype == object
        assert isinstance(result["tld"].iloc[0], str)

    def test_accepts_dataframe(self, extractor):
        df = pd.DataFrame({"url": ["https://a.com", "http://b.org"]})
        result = extractor.fit_transform(df)
        assert len(result) == 2
        assert list(result.columns) == FEATURE_NAMES

    def test_accepts_list(self, extractor):
        urls = ["https://a.com", "http://b.org"]
        result = extractor.fit_transform(urls)
        assert len(result) == 2

    def test_accepts_numpy_array(self, extractor):
        urls = np.array(["https://a.com", "http://b.org"])
        result = extractor.fit_transform(urls)
        assert len(result) == 2


# -----------------------------------------------------------------------
# 15. Repeated calls produce identical output
# -----------------------------------------------------------------------
class TestDeterminism:
    def test_repeated_calls(self, extractor):
        urls = pd.Series([
            "https://www.example.com/path?q=val#frag",
            "http://192.168.1.1/admin",
            "ftp://files.example.org/doc.pdf",
        ])
        r1 = extractor.fit_transform(urls)
        r2 = extractor.fit_transform(urls)
        pd.testing.assert_frame_equal(r1, r2)

    def test_entropy_determinism(self):
        assert _shannon_entropy("hello") == _shannon_entropy("hello")
        assert _shannon_entropy("") == 0.0


# -----------------------------------------------------------------------
# Suspicious keyword test
# -----------------------------------------------------------------------
class TestSuspiciousKeywords:
    def test_keyword_count(self, extractor):
        urls = pd.Series(["https://login-verify-account.example.com/login"])
        result = extractor.fit_transform(urls)
        # "login" x2, "verify" x1, "account" x1 = 4
        assert result["suspicious_keyword_count"].iloc[0] == 4

    def test_no_keywords(self, extractor):
        urls = pd.Series(["https://www.google.com"])
        result = extractor.fit_transform(urls)
        assert result["suspicious_keyword_count"].iloc[0] == 0


# -----------------------------------------------------------------------
# Helper function tests
# -----------------------------------------------------------------------
class TestHelpers:
    def test_is_ip_v4(self):
        assert _is_ip_address("192.168.1.1") is True
        assert _is_ip_address("10.0.0.1") is True

    def test_is_ip_v6(self):
        assert _is_ip_address("::1") is True
        assert _is_ip_address("[::1]") is True

    def test_not_ip(self):
        assert _is_ip_address("example.com") is False
        assert _is_ip_address("") is False

    def test_extract_tld(self):
        assert _extract_tld("www.example.com") == "com"
        assert _extract_tld("example.org") == "org"
        assert _extract_tld("192.168.1.1") == ""
        assert _extract_tld("") == ""
        assert _extract_tld("localhost") == ""

    def test_entropy_values(self):
        # All same characters → entropy = 0
        assert _shannon_entropy("aaaa") == 0.0
        # Two different chars equally distributed → entropy = 1.0
        assert abs(_shannon_entropy("ab") - 1.0) < 1e-10


# -----------------------------------------------------------------------
# Preprocessing tests
# -----------------------------------------------------------------------
class TestPreprocessor:
    def test_build_preprocessor(self):
        preprocessor = build_preprocessor()
        assert preprocessor is not None

    def test_preprocessor_fit_transform(self, extractor):
        """Test that the preprocessor can fit and transform features."""
        urls = pd.Series([
            "https://www.google.com",
            "http://phishing-site.evil.xyz/login",
            "https://bank.example.com/account",
        ])
        features = extractor.fit_transform(urls)
        preprocessor = build_preprocessor()
        result = preprocessor.fit_transform(features)
        assert result.shape[0] == 3
        assert result.shape[1] > 0

    def test_unseen_tld(self, extractor):
        """
        Test that the preprocessor handles an unseen TLD at predict time
        without crashing.
        """
        # Fit on known TLDs
        train_urls = pd.Series([
            "https://example.com/path",
            "https://example.org/path",
        ])
        train_features = extractor.fit_transform(train_urls)
        preprocessor = build_preprocessor()
        preprocessor.fit(train_features)

        # Transform with an unseen TLD
        test_urls = pd.Series(["https://example.nevertld/path"])
        test_url_features = extractor.transform(test_urls)
        result = preprocessor.transform(test_url_features)

        # Should not crash and should have the same number of columns
        assert result.shape[0] == 1
        assert result.shape[1] > 0

    def test_preprocessor_column_lists(self):
        """Test that numerical and categorical feature lists are consistent."""
        all_features = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
        expected = set(FEATURE_NAMES)
        assert all_features == expected
