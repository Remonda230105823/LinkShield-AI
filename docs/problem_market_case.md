# LinkShield AI — Problem Statement and Market Case

## Problem statement
Phishing attackers use deceptive URLs to impersonate trusted services and trick users into revealing sensitive information. A newly created phishing URL may not yet appear on a known blocklist. A retail bank security operations or email-security team therefore needs additional signals to prioritize suspicious links reported by customers or received in messages.

LinkShield AI is a Phase 1 academic prototype that analyzes URL text without visiting the destination. It aims to classify URLs as legitimate or phishing and later provide interpretable risk signals. It is an additional triage aid, not a replacement for bank security controls.

## Primary customer and workflow
**Primary customer:** Retail bank security operations / email-security team.

Suspicious URLs arrive through messages, email-security workflows or customer reports. Analysts review the alerts and decide which require investigation. Missed phishing URLs may expose customers; false alarms consume analyst time and may interfere with legitimate communications.

## Value proposition and limitations
The proposed pre-click score may help prioritize analyst review. Phase 1 does not claim measured accuracy, reduced fraud losses, production deployment, or actual bank adoption. It does not inspect webpage contents, execute malware, or open submitted URLs.

## Predefined success metrics
- **Primary:** Phishing recall (detecting actual phishing; false negatives matter).
- **Supporting:** Phishing precision and phishing F1-score (false positives and trade-off).
- **Initial decision threshold:** 0.50 for baseline only; do not tune using test data.
- **Operational proxy:** Estimated analyst review hours per 10,000 screened URLs = (TP + FP) × assumed minutes per alert / 60. Report FN separately. Any 1/3/5-minute scenarios are illustrative assumptions, not measured bank data.
- Person 3 must report actual held-out test metrics; no performance target is asserted as achieved here.
