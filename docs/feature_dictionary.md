# Feature Dictionary — LinkShield AI

This document describes every feature produced by `URLFeatureExtractor` in `src/url_features.py`.

All features are **lexical** — derived from the URL text alone without visiting, resolving, or scanning the URL.

> [!IMPORTANT]
> URL lexical features are useful indicators but **cannot guarantee detection of all phishing URLs**. A sophisticated attacker can craft a URL that looks lexically benign. These features should be used as part of a broader security workflow, not as a sole detection mechanism.

---

## Numerical Features

### URL Structure Features

| # | Feature | Meaning | Calculation | Type | Example | Phishing Relevance | Limitations |
|---|---------|---------|-------------|------|---------|---------------------|-------------|
| 1 | `url_length` | Total character length of the raw URL string | `len(url)` | int | `https://example.com` → 20 | Phishing URLs tend to be longer due to obfuscation, random paths, or encoded payloads | Not all long URLs are phishing; short URLs can also be malicious |
| 2 | `domain_length` | Character length of the hostname (netloc without port/userinfo) | `len(parsed.hostname)` | int | `example.com` → 11 | Phishing domains may use long, confusing hostnames | Some legitimate domains (CDNs, subdomains) are also long |
| 3 | `path_length` | Character length of the URL path component | `len(parsed.path)` | int | `/login/verify` → 14 | Deep paths may indicate obfuscation attempts | Legitimate sites can have deep path structures |
| 4 | `query_length` | Character length of the query string (after `?`, before `#`) | `len(parsed.query)` | int | `q=test&page=2` → 14 | Long query strings may carry encoded phishing payloads | Search engines and apps use long query strings legitimately |
| 5 | `fragment_length` | Character length of the fragment (after `#`) | `len(parsed.fragment)` | int | `section1` → 8 | Rarely used in phishing; unusual fragments may be suspicious | Single-page apps use fragments extensively |
| 6 | `num_dots` | Count of `.` characters in the full URL | `url.count(".")` | int | `a.b.c.com/p` → 3 | More dots may indicate subdomain abuse | Multi-level legitimate domains also have many dots |
| 7 | `num_hyphens` | Count of `-` characters in the full URL | `url.count("-")` | int | `my-bank-login.com` → 2 | Hyphens are common in impersonation domains | Many legitimate domains use hyphens |
| 8 | `num_underscores` | Count of `_` characters in the full URL | `url.count("_")` | int | `page_login` → 1 | Less common in URLs; may indicate generated paths | Some CMS systems use underscores in paths |
| 9 | `num_slashes` | Count of `/` characters in the full URL | `url.count("/")` | int | `https://a.com/b/c/d` → 5 | Many slashes may indicate deep, obfuscated paths | REST APIs and deep site structures are legitimate |
| 10 | `num_at_symbols` | Count of `@` characters in the full URL | `url.count("@")` | int | `http://user@evil.com` → 1 | `@` in URLs is a classic spoofing technique (browsers may ignore text before `@`) | Rare in URLs; almost always suspicious |
| 11 | `num_question_marks` | Count of `?` characters in the full URL | `url.count("?")` | int | `?q=1` → 1 | Multiple `?` is syntactically unusual and may indicate obfuscation | Normally only one `?` per URL |
| 12 | `num_ampersands` | Count of `&` characters in the full URL | `url.count("&")` | int | `?a=1&b=2&c=3` → 2 | Many parameters may carry phishing payloads | E-commerce and analytics URLs have many parameters |
| 13 | `num_equal_signs` | Count of `=` characters in the full URL | `url.count("=")` | int | `?a=1&b=2` → 2 | Accompanies query parameters | Normal in URLs with query strings |
| 14 | `num_percent_signs` | Count of `%` characters in the full URL | `url.count("%")` | int | `%20%3D` → 2 | Heavy percent-encoding may indicate obfuscation of suspicious content | Some legitimate URLs with international characters use percent-encoding |
| 15 | `num_digits` | Count of digit characters (0-9) in the full URL | `sum(c.isdigit() for c in url)` | int | `http://192.168.1.1` → 10 | Many digits may indicate IP-based URLs or random tokens | Version numbers, dates, IDs are legitimate |
| 16 | `num_special_chars` | Count of characters in the set `@!#$%^&*()_+-=[]{}|;':\",./<>?~\`` | Character-set membership count | int | `http://a.com/@b` → varies | High special-char density may indicate obfuscation | URL delimiters (`:`, `/`, `.`) inflate the count |

### Hostname Features

| # | Feature | Meaning | Calculation | Type | Example | Phishing Relevance | Limitations |
|---|---------|---------|-------------|------|---------|---------------------|-------------|
| 17 | `num_subdomains` | Number of subdomain levels in the hostname | `max(0, len(hostname.split(".")) - 2)` for non-IP hosts; 0 for IPs | int | `a.b.c.example.com` → 3 | Subdomain abuse (e.g., `login.bank.evil.com`) is common in phishing | Some CDNs and cloud services use deep subdomains |

### Other Lexical Features

| # | Feature | Meaning | Calculation | Type | Example | Phishing Relevance | Limitations |
|---|---------|---------|-------------|------|---------|---------------------|-------------|
| 23 | `url_entropy` | Character-level Shannon entropy of the URL string | H = −Σ p(c) · log₂(p(c)) where p(c) = count(c)/len(url) | float | Random string → ~4.0; repetitive → ~1.0 | High entropy suggests random/obfuscated URLs (common in phishing) | Some legitimate URLs (UUIDs, tokens) also have high entropy |
| 24 | `suspicious_keyword_count` | Total occurrences of suspicious keywords in the lowercased URL | Sum of `url_lower.count(kw)` for each keyword in the fixed list | int | `login-verify-account.com/login` → 4 | Keywords like "login", "verify", "bank" are commonly used to deceive users | Legitimate banking and security pages also contain these words |

**Suspicious keyword list** (fixed, case-insensitive matching):
`login`, `verify`, `update`, `secure`, `account`, `bank`, `signin`, `password`, `confirm`, `wallet`, `billing`, `support`

Each keyword is counted as many times as it appears (non-overlapping). Different keywords are summed.

---

## Binary Indicator Features

| # | Feature | Meaning | Calculation | Type | Example | Phishing Relevance | Limitations |
|---|---------|---------|-------------|------|---------|---------------------|-------------|
| 18 | `has_ip` | Whether the hostname is a raw IP address (IPv4 or IPv6) | `ipaddress.ip_address()` validation | int (0/1) | `http://192.168.1.1` → 1 | IP-based URLs bypass domain-name trust and are common in phishing | Some internal/legitimate tools use IP addresses |
| 19 | `has_https` | Whether the URL's **original** scheme is `https` | Checks the original scheme (not the parsing fallback) | int (0/1) | `https://example.com` → 1 | Lack of HTTPS may indicate a quickly-set-up phishing page | Many phishing sites now use HTTPS via free certificates |
| 20 | `has_www` | Whether the hostname starts with `www.` | `hostname.lower().startswith("www.")` | int (0/1) | `www.example.com` → 1 | Omitting `www.` is slightly more common in phishing | Many legitimate sites omit `www.` |
| 21 | `has_fragment` | Whether the URL contains a fragment (`#`) | `len(fragment) > 0` | int (0/1) | `page#section` → 1 | Unusual fragment usage may indicate obfuscation | Single-page apps use fragments heavily |
| 22 | `has_query` | Whether the URL contains a query string (`?`) | `len(query) > 0` | int (0/1) | `page?id=1` → 1 | Query strings can carry phishing parameters | Very common in legitimate URLs |

---

## Categorical Features

| # | Feature | Meaning | Calculation | Type | Example | Phishing Relevance | Limitations |
|---|---------|---------|-------------|------|---------|---------------------|-------------|
| 25 | `tld` | Top-level domain of the hostname | Last dot-separated segment of the hostname, lowercased. Empty string for IP addresses or single-label hostnames. | str (categorical) | `example.com` → `"com"`, `192.168.1.1` → `""` | Some TLDs (`.xyz`, `.tk`, `.top`) are disproportionately used in phishing due to low cost and minimal registration requirements | Not all URLs on these TLDs are malicious; TLD alone is insufficient for classification |

---

## Feature Extraction Edge Cases

| Scenario | Behaviour |
|----------|-----------|
| Missing/None/NaN URL | All numeric features = 0, `tld` = empty string |
| Empty string URL | Same as missing |
| URL without scheme | `http://` prepended for parsing only; `has_https` = 0 |
| URL with non-default port | Port preserved in normalization; does not affect feature values |
| IP-based hostname | `has_ip` = 1, `num_subdomains` = 0, `tld` = empty string |
| Unknown/unusual TLD | Extracted as-is; handled by `OneHotEncoder(handle_unknown="ignore")` in preprocessing |

---

## Summary

- **24 numerical/binary features** + **1 categorical feature** = **25 total features**.
- All features are deterministic: same input always produces same output.
- No internet access, DNS lookups, or website visits required.
- Feature extraction is separated from preprocessing (scaling/encoding).
- The `tld` feature is one-hot-encoded during preprocessing, expanding the effective dimensionality.

---

## Why domain age is not a feature

The project brief mentions domain age. It is deliberately excluded in Phase 1:

- Domain age needs registration data from WHOIS/RDAP or another external
  service. A pasted URL string does not contain it.
- Looking it up would mean network calls for every URL, with rate limits,
  missing answers and results that change over time, so a run could not be
  reproduced exactly.
- The project rule is that URLs are inert text and nothing is resolved or
  requested.

No domain-age value was estimated or invented. If a reliable offline source is
agreed later, it can be added as a separate, clearly documented feature.
