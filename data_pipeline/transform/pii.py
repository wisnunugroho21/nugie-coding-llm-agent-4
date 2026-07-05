"""
Stage 4a — PII reduction (paper Sec. 2.1).

The paper uses "complex regular expressions to detect such information and
replace them with placeholders such as '<name>' and '<password>'", targeting
passwords, emails and IP addresses. We implement matchable, high-precision
patterns for emails, IPv4/IPv6 addresses, and secret assignments (password /
token / api_key / private keys), replacing each with a typed placeholder and
recording how many of each were redacted in `doc.meta`.

Precision over recall: these patterns are deliberately conservative so we don't
mangle ordinary code. Real deployments would layer a detect-secrets / model-based
pass on top; the interface here (regex list -> counted substitution) is the same.
"""

from __future__ import annotations

import re

from ..config import TransformConfig
from ..models import CodeDocument

_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_IPV4 = re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b")
_IPV6 = re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b")
# key/secret assignment: `password = "..."`, `api_key: '...'`, `TOKEN="..."`
_SECRET = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|token)\b"
    r"(\s*[:=]\s*)"
    r"(\"[^\"]+\"|'[^']+'|[^\s,;]+)"
)
_PRIVATE_KEY = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----",
    re.DOTALL,
)


def redact_pii(doc: CodeDocument, cfg: TransformConfig) -> CodeDocument:
    counts = {"email": 0, "ip": 0, "secret": 0, "private_key": 0}
    text = doc.content

    text, n = _PRIVATE_KEY.subn(cfg.key_placeholder, text)
    counts["private_key"] += n

    def _secret_sub(m: re.Match) -> str:
        # Preserve the surrounding quotes so redacted assignments stay
        # syntactically valid (a bare <password> would break AST parsing and
        # then get dropped by the later Python AST filter).
        value = m.group(3)
        quote = value[0] if value[:1] in ("'", '"') else ""
        placeholder = f"{quote}{cfg.secret_placeholder}{quote}"
        return f"{m.group(1)}{m.group(2)}{placeholder}"

    text, n = _SECRET.subn(_secret_sub, text)
    counts["secret"] += n

    text, n = _EMAIL.subn(cfg.email_placeholder, text)
    counts["email"] += n

    text, n = _IPV4.subn(cfg.ip_placeholder, text)
    counts["ip"] += n
    text, n = _IPV6.subn(cfg.ip_placeholder, text)
    counts["ip"] += n

    doc.content = text
    if any(counts.values()):
        doc.meta.setdefault("pii_redactions", {})
        for k, v in counts.items():
            doc.meta["pii_redactions"][k] = doc.meta["pii_redactions"].get(k, 0) + v
    return doc
