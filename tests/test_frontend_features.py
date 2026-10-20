from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / 'frontend' / 'index.html').read_text(encoding='utf-8')
APP = (ROOT / 'frontend' / 'app.js').read_text(encoding='utf-8')
CSS = (ROOT / 'frontend' / 'styles.css').read_text(encoding='utf-8')


def test_history_is_removed_from_ui_and_code():
    assert 'Historial local' not in INDEX
    assert 'data-view="history"' not in INDEX
    assert 'id="historyScreen"' not in INDEX
    assert 'privadoc.history.v1' not in APP
    assert 'saveHistoryEntry' not in APP
    assert 'renderHistory' not in APP


def test_settings_remain_available():
    assert 'data-view="settings"' in INDEX
    assert 'id="settingsScreen"' in INDEX
    assert 'privadoc.settings.v1' in APP


def test_feedback_card_exists_but_is_not_intrusive_before_download():
    assert 'id="feedbackCard"' in INDEX
    assert 'feedback-card hidden' in INDEX
    assert 'id="feedbackBtn"' in INDEX
    assert 'Tu opinión nos ayuda a validar PRIVADOC' in INDEX
    assert 'feedbackCard' in APP
    assert "#downloadBtn" in APP
    assert "setTimeout(revealFeedbackPrompt" in APP
    assert "classList.remove('hidden')" in APP


def test_feedback_google_forms_url_is_one_line_configuration():
    assert 'FEEDBACK_FORM_URL' in APP
    assert 'GOOGLE_FORMS_URL_AQUI' in APP
    assert 'window.open(FEEDBACK_FORM_URL' in APP


def test_mobile_layout_covers_done_and_feedback_sections():
    assert '@media(max-width:760px)' in CSS
    assert '.done-grid' in CSS
    assert '.feedback-card' in CSS
    assert '.feedback-actions' in CSS
    assert '.feedback-actions .btn' in CSS


def test_frontend_version_is_updated():
    assert 'v0.10.0' in INDEX
    assert 'styles.css?v=0.10.0' in INDEX
    assert 'app.js?v=0.10.0' in INDEX
