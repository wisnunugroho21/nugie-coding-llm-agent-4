"""
Stage 6 — High-resource language downsampling (paper Sec. 2.1).

The paper caps over-represented languages so they don't dominate the mixture:
Java 409GB -> 200GB and HTML 213GB -> 64GB, for example. We express each cap as a
per-language *keep fraction* and drop the rest by seeded random sampling, which is
reproducible and preserves the within-language distribution. Languages absent from
`keep_fraction` are kept in full.
"""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator

from ..config import DownsampleConfig
from ..models import CodeDocument


def downsample(
    docs: Iterable[CodeDocument], cfg: DownsampleConfig
) -> Iterator[CodeDocument]:
    rng = random.Random(cfg.seed)
    for doc in docs:
        frac = cfg.keep_fraction.get(doc.language, 1.0)
        if frac >= 1.0 or rng.random() < frac:
            yield doc
