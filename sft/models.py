"""
The unit of SFT data: `InstructionExample` (one instruction/response pair).

Fields carry the provenance the two-stage composer and decontamination need:
`source` (RealUser-Instruct, Evol-Instruct, Educational-Instruct, ...), `stage`
(1 or 2), `language`, and `meta` for anything a synthesizer wants to record
(seed, whether it was test-validated, etc.).
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass
class InstructionExample:
    instruction: str
    response: str
    source: str = ""            # e.g. "educational_instruct"
    language: str = "Python"
    stage: int = 0              # 1 or 2 once composed
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    @property
    def text(self) -> str:
        """Instruction + response concatenated (used for decontamination scans)."""
        return self.instruction + "\n" + self.response

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "InstructionExample":
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)
