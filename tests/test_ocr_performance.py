from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest

from app.core.models import OCRLine
import app.core.ocr as ocr


class _Raw:
    def __init__(self, boxes=None, txts=None, scores=None):
        self.boxes = boxes
        self.txts = txts
        self.scores = scores


def test_full_rapidocr_disables_orientation_classifier(monkeypatch):
    calls = []

    class Engine:
        def __call__(self, img, **kwargs):
            calls.append(kwargs)
            return _Raw(
                boxes=np.array([[[0, 0], [20, 0], [20, 10], [0, 10]]], dtype=np.float32),
                txts=("DNI",),
                scores=(0.95,),
            )

    monkeypatch.setattr(ocr, "_get_engine", lambda: Engine())
    lines = ocr._rapid(np.zeros((20, 40, 3), dtype=np.uint8))

    assert lines and lines[0].text == "DNI"
    assert calls == [{"use_det": True, "use_cls": False, "use_rec": True}]


def test_single_line_ocr_skips_detector_and_classifier(monkeypatch):
    calls = []

    class Engine:
        def __call__(self, img, **kwargs):
            calls.append(kwargs)
            return _Raw(boxes=None, txts=("71481817",), scores=(0.94,))

    monkeypatch.setattr(ocr, "_get_engine", lambda: Engine())
    line, _ = ocr.run_ocr_line(np.zeros((36, 180, 3), dtype=np.uint8))

    assert line is not None
    assert line.text == "71481817"
    assert line.box == [0, 0, 180, 36]
    assert calls == [{"use_det": False, "use_cls": False, "use_rec": True}]


def test_resilient_ocr_stops_after_successful_scaled_pass(monkeypatch):
    calls = {"n": 0}

    def fake_engine(img):
        calls["n"] += 1
        # Base and CLAHE are intentionally sparse. The scaled pass succeeds.
        if calls["n"] < 3:
            lines = [
                OCRLine(f"L{calls['n']}-A", 0.90, [10, 10, 80, 30]),
                OCRLine(f"L{calls['n']}-B", 0.90, [100, 40, 180, 65]),
            ]
        else:
            lines = [
                OCRLine(f"FIELD-{i}", 0.90, [10 + i * 30, 20 + i * 35, 35 + i * 30, 42 + i * 35])
                for i in range(10)
            ]
        return lines, "fake"

    monkeypatch.setattr(ocr, "_run_single_engine", fake_engine)
    lines, _ = ocr.run_ocr_resilient(np.zeros((800, 800, 3), dtype=np.uint8))

    assert len(lines) >= 9
    # Regression: the old implementation evaluated stale `merged` data and made a 4th OCR call.
    assert calls["n"] == 3


@pytest.mark.parametrize(
    "module_name",
    ["dni_blue", "dni_yellow", "dni_electronic"],
)
def test_structured_name_roi_uses_recognition_only_fast_path(monkeypatch, module_name):
    mod = __import__(f"app.documents.{module_name}.processor", fromlist=["*"])
    image = np.full((mod.CANON_H, mod.CANON_W, 3), 220, dtype=np.uint8)
    image[130:145, 250:480] = 40

    def fake_line(img):
        return OCRLine("CHAVEZ", 0.94, [0, 0, img.shape[1], img.shape[0]]), "fake-rec"

    monkeypatch.setattr(mod, "run_ocr_line", fake_line)
    monkeypatch.setattr(
        mod,
        "_ocr_crop_lines",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no debe ejecutar detector en fast path válido")),
    )
    monkeypatch.setattr(
        mod,
        "_tesseract_text_variants",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("Tesseract no debe ejecutarse si RapidOCR ya validó el valor")),
    )

    line = mod._best_crop_value(image, [220, 120, 520, 170], "letters")

    assert line is not None
    assert line.text == "CHAVEZ"


@pytest.mark.parametrize(
    "module_name",
    ["dni_blue", "dni_yellow", "dni_electronic"],
)
def test_strong_template_match_skips_orientation_ocr(monkeypatch, module_name):
    mod = __import__(f"app.documents.{module_name}.processor", fromlist=["*"])
    card = np.zeros((mod.CANON_H, mod.CANON_W, 3), dtype=np.uint8)

    def fake_rotate(img, angle):
        out = img.copy()
        out[0, 0, 0] = 0 if angle == 0 else 1
        return out

    def fake_register(img, side):
        angle0 = int(img[0, 0, 0]) == 0
        if angle0 and side == "front":
            conf = 0.92
        elif angle0 and side == "back":
            conf = 0.55
        else:
            conf = 0.42
        return img, np.eye(3), conf, True

    monkeypatch.setattr(mod, "rotate_image", fake_rotate)
    monkeypatch.setattr(mod, "_register_reference", fake_register)
    monkeypatch.setattr(
        mod,
        "_safe_ocr_for_scoring",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("OCR de orientación no debe ejecutarse con plantilla concluyente")),
    )

    _, angle, side, confidence = mod.choose_orientation_and_side(card)

    assert angle == 0
    assert side == "front"
    assert confidence > 0.7


def test_structured_resilient_ocr_uses_at_most_two_global_detector_passes(monkeypatch):
    calls = {"n": 0}

    def fake_engine(img):
        calls["n"] += 1
        # Deliberately sparse on both passes. Structured mode must hand off to
        # field-level ROI rescue instead of starting more full-card detector passes.
        return [OCRLine(f"TOKEN-{calls['n']}", 0.72, [10, 10, 80, 30])], "fake"

    monkeypatch.setattr(ocr, "_run_single_engine", fake_engine)
    lines, engine = ocr.run_ocr_resilient(
        np.zeros((640, 1000, 3), dtype=np.uint8),
        structured=True,
    )

    assert lines
    assert calls["n"] == 2
    assert "estructurado" in engine
