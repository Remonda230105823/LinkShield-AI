"""
LinkShield AI — URL Feature Extraction Module
===============================================

Person 2 deliverable: a reusable scikit-learn-compatible transformer that
extracts deterministic lexical features from raw URL strings.

All features are derived from the URL text alone.  No URL is visited,
resolved, or accessed over the network.  No DNS, WHOIS, or external-API
lookups are performed.

Author: Person 2 (Feature Engineering & Preprocessing)
"""

import ipaddress
import math
import re
import string
from collections import Counter
from typing import List, Optional, Union
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Fixed, documented list of suspicious keywords.
#: These are common in phishing URLs but their presence is an indicator,
#: **not proof** that a URL is malicious.
SUSPICIOUS_KEYWORDS: List[str] = [
    "login",
    "verify",
    "update",
    "secure",
    "account",
    "bank",
    "signin",
    "password",
    "confirm",
    "wallet",
    "billing",
    "support",
]

#: Characters considered "special" for the ``num_special_chars`` feature.
#: This includes common URL-special characters and other non-alphanumeric,
#: non-standard punctuation sometimes seen in obfuscated URLs.
SPECIAL_CHARS = set("@!#$%^&*()_+-=[]{}|;':\",./<>?`~\\")

# Ordered list of feature names produced by the extractor.
FEATURE_NAMES: List[str] = [
    # URL structure
    "url_length",
    "domain_length",
    "path_length",
    "query_length",
    "fragment_length",
    "num_dots",
    "num_hyphens",
    "num_underscores",
    "num_slashes",
    "num_at_symbols",
    "num_question_marks",
    "num_ampersands",
    "num_equal_signs",
    "num_percent_signs",
    "num_digits",
    "num_special_chars",
    # Hostname features
    "num_subdomains",
    "has_ip",
    "has_https",
    "has_www",
    # Other lexical features
    "has_fragment",
    "has_query",
    "url_entropy",
    "suspicious_keyword_count",
    # Categorical
    "tld",
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _safe_parse(url: str):
    """
    Parse a URL with ``urllib.parse.urlparse``.

    If the URL has no scheme, prepend ``http://`` as a **parsing-only**
    fallback so that ``urlparse`` can correctly identify the netloc, path,
    query, and fragment.  The prepended scheme is recorded so that
    ``has_https`` reflects the *original* URL.

    Returns (ParseResult, had_original_scheme: bool).
    """
    has_scheme = bool(re.match(r'^[a-zA-Z][a-zA-Z0-9+\-.]*://', url))
    if not has_scheme:
        parsed = urlparse("http://" + url)
    else:
        parsed = urlparse(url)
    return parsed, has_scheme


def _is_ip_address(hostname: str) -> bool:
    """
    Check whether *hostname* is a valid IPv4 or IPv6 address using
    Python's ``ipaddress`` module.
    """
    if not hostname:
        return False
    # Strip brackets for IPv6
    h = hostname.strip("[]")
    try:
        ipaddress.ip_address(h)
        return True
    except ValueError:
        return False


def _extract_tld(hostname: str) -> str:
    """
    Extract the top-level domain from a hostname.

    Simple heuristic: take the last dot-separated segment.
    Returns empty string if no TLD can be determined (e.g. IP address or
    single-label hostname).
    """
    if not hostname or _is_ip_address(hostname):
        return ""
    parts = hostname.rstrip(".").split(".")
    if len(parts) < 2:
        return ""
    return parts[-1].lower()


def _shannon_entropy(text: str) -> float:
    """
    Compute character-level Shannon entropy of *text*.

    Formula:  H = -Σ p(c) · log₂(p(c))

    where p(c) = count(c) / len(text) for each unique character c.

    Returns 0.0 for empty strings.
    """
    if not text:
        return 0.0
    length = len(text)
    counts = Counter(text)
    entropy = 0.0
    for count in counts.values():
        p = count / length
        if p > 0:
            entropy -= p * math.log2(p)
    return entropy


def _count_suspicious_keywords(url_lower: str) -> int:
    """
    Count total occurrences of suspicious keywords in the lowercased URL.

    Matching is **case-insensitive** (the URL is already lowered).
    Each keyword is counted as many times as it appears (non-overlapping).
    Different keywords are summed together.
    """
    total = 0
    for kw in SUSPICIOUS_KEYWORDS:
        total += url_lower.count(kw)
    return total


# ---------------------------------------------------------------------------
# Main Transformer
# ---------------------------------------------------------------------------

class URLFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Scikit-learn-compatible transformer that extracts deterministic lexical
    features from raw URL strings.

    Accepts:
    - A pandas Series of URL strings.
    - A single-column pandas DataFrame with URL strings.
    - A list or 1-D numpy array of URL strings.

    Returns a pandas DataFrame with one row per input URL and columns
    matching ``FEATURE_NAMES``.

    Missing or non-string values are handled safely: they produce a row
    of zeros / empty strings for the TLD, with ``url_length = 0``.

    This transformer is **stateless** — ``fit()`` is a no-op.  It never
    accesses the target/label.
    """

    def fit(self, X, y=None):
        """No-op.  The extractor is stateless."""
        return self

    def transform(self, X, y=None) -> pd.DataFrame:
        """
        Extract features from *X*.

        Parameters
        ----------
        X : array-like of URL strings
            pandas Series, single-column DataFrame, list, or 1-D ndarray.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``FEATURE_NAMES``.
        """
        urls = self._coerce_input(X)
        rows = [self._extract_single(u) for u in urls]
        df = pd.DataFrame(rows, columns=FEATURE_NAMES)

        # Enforce numeric dtypes for numeric features
        numeric_cols = [c for c in FEATURE_NAMES if c != "tld"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Enforce int types for binary/count columns
        int_cols = [
            "url_length", "domain_length", "path_length", "query_length",
            "fragment_length", "num_dots", "num_hyphens", "num_underscores",
            "num_slashes", "num_at_symbols", "num_question_marks",
            "num_ampersands", "num_equal_signs", "num_percent_signs",
            "num_digits", "num_special_chars", "num_subdomains",
            "has_ip", "has_https", "has_www", "has_fragment", "has_query",
            "suspicious_keyword_count",
        ]
        for col in int_cols:
            df[col] = df[col].astype(int)

        # TLD as string / categorical
        df["tld"] = df["tld"].astype(str)

        return df

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Return the ordered list of output feature names."""
        return list(FEATURE_NAMES)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _coerce_input(X) -> List[str]:
        """Coerce input to a list of strings, handling None/NaN safely."""
        if isinstance(X, pd.DataFrame):
            if X.shape[1] != 1:
                raise ValueError(
                    f"Expected a single-column DataFrame, got {X.shape[1]} columns."
                )
            series = X.iloc[:, 0]
        elif isinstance(X, pd.Series):
            series = X
        elif isinstance(X, np.ndarray):
            series = pd.Series(X.ravel())
        elif isinstance(X, list):
            series = pd.Series(X)
        else:
            raise TypeError(f"Unsupported input type: {type(X)}")

        # Convert to list, replacing NaN / None with empty string
        return [str(v) if pd.notna(v) and v is not None else "" for v in series]

    @staticmethod
    def _extract_single(url: str) -> list:
        """Extract all features for a single URL string."""
        if not url:
            # Return safe defaults for missing/empty URLs
            return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                    0, 0, 0, 0, 0, 0, 0.0, 0, ""]

        parsed, had_scheme = _safe_parse(url)
        hostname = parsed.hostname or ""
        path = parsed.path or ""
        query = parsed.query or ""
        fragment = parsed.fragment or ""
        url_lower = url.lower()

        # --- URL structure ---
        url_length = len(url)
        domain_length = len(hostname)
        path_length = len(path)
        query_length = len(query)
        fragment_length = len(fragment)

        num_dots = url.count(".")
        num_hyphens = url.count("-")
        num_underscores = url.count("_")
        num_slashes = url.count("/")
        num_at_symbols = url.count("@")
        num_question_marks = url.count("?")
        num_ampersands = url.count("&")
        num_equal_signs = url.count("=")
        num_percent_signs = url.count("%")
        num_digits = sum(c.isdigit() for c in url)
        num_special_chars = sum(c in SPECIAL_CHARS for c in url)

        # --- Hostname features ---
        # Subdomains: number of dots in hostname gives (num_parts - 1)
        # e.g. "a.b.example.com" has 3 dots → 4 parts → 2 subdomains
        #      (everything before "example.com")
        # Heuristic: subdomains = max(0, parts - 2) for normal domains
        # For IPs: 0 subdomains
        if _is_ip_address(hostname):
            num_subdomains = 0
        else:
            parts = hostname.rstrip(".").split(".")
            num_subdomains = max(0, len(parts) - 2)

        has_ip = int(_is_ip_address(hostname))

        # has_https: based on the ORIGINAL URL scheme
        if had_scheme:
            has_https = int(parsed.scheme.lower() == "https")
        else:
            has_https = 0

        has_www = int(hostname.lower().startswith("www."))

        # --- Other lexical features ---
        has_fragment = int(len(fragment) > 0)
        has_query = int(len(query) > 0)
        url_entropy = _shannon_entropy(url)
        suspicious_kw_count = _count_suspicious_keywords(url_lower)

        # --- Categorical ---
        tld = _extract_tld(hostname)

        return [
            url_length,
            domain_length,
            path_length,
            query_length,
            fragment_length,
            num_dots,
            num_hyphens,
            num_underscores,
            num_slashes,
            num_at_symbols,
            num_question_marks,
            num_ampersands,
            num_equal_signs,
            num_percent_signs,
            num_digits,
            num_special_chars,
            num_subdomains,
            has_ip,
            has_https,
            has_www,
            has_fragment,
            has_query,
            url_entropy,
            suspicious_kw_count,
            tld,
        ]
