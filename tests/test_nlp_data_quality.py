"""Known edit-distance examples keep WER independent of the ASR output."""
import pytest
from tools.report_nlp_data_quality import word_error_rate, words


def test_normalization_is_explicit():
    assert words('  ВСЁ, хорошо!\nДа—нет. 42 ') == ['всё', 'хорошо', 'да', 'нет', '42']
    assert word_error_rate('всё', 'все')['wer'] == 1
    assert word_error_rate('Ёж, идёт!', 'ёж идёт')['wer'] == 0


@pytest.mark.parametrize('reference,hypothesis,edits', [
    ('один два три', 'один четыре три', 1),
    ('один два три', 'один три', 1),
    ('один два три', 'один два четыре три', 1),
    ('один два три', '', 3),
    ('да да нет', 'да нет нет', 1),
])
def test_known_edit_counts(reference, hypothesis, edits):
    result = word_error_rate(reference, hypothesis)
    assert result['word_edits'] == edits
    assert result['reference_words'] == 3
    assert result['wer'] == pytest.approx(edits / 3)


def test_wer_can_exceed_one():
    assert word_error_rate('да', 'нет нет нет')['wer'] == 3


def test_empty_reference_is_rejected():
    with pytest.raises(ValueError, match='Reference'):
        word_error_rate(' ... ', 'слово')
