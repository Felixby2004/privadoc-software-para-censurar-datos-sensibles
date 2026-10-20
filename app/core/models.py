from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class OCRLine:
    text: str
    confidence: float
    box: list[int]


@dataclass
class SensitiveField:
    id: str
    name: str
    value: str
    risk: str
    recommendation: str
    box: list[int]
    confidence: float
    category: str = "personal"
    default_redact: bool = True
    source: str = "ocr"
    mask_box: list[int] | None = None


@dataclass
class DocumentResult:
    id: str
    page_index: int
    type: str
    side: str
    side_label: str
    orientation: int
    detection_confidence: float
    classification_confidence: float
    registration_confidence: float
    reference_used: bool
    width: int
    height: int
    preview: str
    fields: list[SensitiveField] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Private runtime-only properties. main.py strips keys beginning with _.
    _page_to_canonical: Any = None
    _canonical_to_page: Any = None
    _image: Any = None

    def public_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "page_index": self.page_index,
            "type": self.type,
            "side": self.side,
            "side_label": self.side_label,
            "orientation": self.orientation,
            "detection_confidence": self.detection_confidence,
            "classification_confidence": self.classification_confidence,
            "registration_confidence": self.registration_confidence,
            "reference_used": self.reference_used,
            "width": self.width,
            "height": self.height,
            "preview": self.preview,
            "fields": [asdict(f) for f in self.fields],
            "warnings": list(self.warnings),
        }
