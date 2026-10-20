from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np

from app.core.models import OCRLine
import app.core.ocr as ocr


def test_refine_ocr_by_layout_adds_text_from_weak_expected_roi(monkeypatch):
    image = np.full((200, 400, 3), 255, dtype=np.uint8)
    initial = [OCRLine('REPUBLICA DEL PERU', 0.90, [10, 10, 180, 30])]
    fields = [{
        'id': 'dni_number',
        'category': 'identifier',
        'value_roi': [0.70, 0.10, 0.95, 0.25],
    }]

    def fake_run(crop):
        return [OCRLine('71481817', 0.88, [5, 4, 80, 24])], 'fake'

    monkeypatch.setattr(ocr, 'run_ocr', fake_run)
    refined = ocr.refine_ocr_by_layout(image, initial, fields, max_regions=4)

    matches = [ln for ln in refined if '71481817' in ln.text]
    assert matches
    assert matches[0].box[0] >= 270
    assert matches[0].box[1] >= 15


def test_refine_ocr_by_layout_skips_roi_when_good_text_already_exists(monkeypatch):
    image = np.full((200, 400, 3), 255, dtype=np.uint8)
    initial = [OCRLine('71481817', 0.92, [285, 25, 365, 48])]
    fields = [{
        'id': 'dni_number',
        'category': 'identifier',
        'value_roi': [0.70, 0.10, 0.95, 0.25],
    }]
    calls = {'n': 0}

    def fake_run(crop):
        calls['n'] += 1
        return [], 'fake'

    monkeypatch.setattr(ocr, 'run_ocr', fake_run)
    refined = ocr.refine_ocr_by_layout(image, initial, fields, max_regions=4)

    assert calls['n'] == 0
    assert any(ln.text == '71481817' for ln in refined)
