from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from app.core.models import OCRLine
import app.documents.receipts.processor as receipts


def test_receipt_structural_rescue_runs_when_core_anchors_are_missing(monkeypatch):
    page = np.full((1000, 600, 3), 255, dtype=np.uint8)
    initial = [OCRLine('PLIN', 0.9, [240, 80, 320, 120])]
    calls = {'n': 0}

    def fake_targeted(page_arg, roi):
        calls['n'] += 1
        return [
            OCRLine('Enviado a', 0.9, [60, 380, 160, 410]),
            OCRLine('934 569 777 - Yape', 0.9, [60, 450, 260, 480]),
            OCRLine('Código de operación', 0.9, [60, 650, 250, 680]),
            OCRLine('50235027', 0.9, [60, 690, 160, 720]),
        ]

    monkeypatch.setattr(receipts, '_targeted_ocr', fake_targeted)
    lines = receipts._supplement_receipt_lines(page, initial)

    assert calls['n'] == 1
    assert any('Enviado a' in x.text for x in lines)
    assert any('50235027' in x.text for x in lines)


def test_receipt_structural_rescue_skips_when_core_anchors_are_present(monkeypatch):
    page = np.full((1000, 600, 3), 255, dtype=np.uint8)
    initial = [
        OCRLine('Enviado a', 0.9, [40, 300, 120, 330]),
        OCRLine('Desde', 0.9, [40, 520, 100, 550]),
        OCRLine('Número de operación', 0.9, [40, 700, 220, 730]),
    ]
    calls = {'n': 0}

    def fake_targeted(page_arg, roi):
        calls['n'] += 1
        return []

    monkeypatch.setattr(receipts, '_targeted_ocr', fake_targeted)
    lines = receipts._supplement_receipt_lines(page, initial)

    assert calls['n'] == 0
    assert len(lines) == len(initial)
