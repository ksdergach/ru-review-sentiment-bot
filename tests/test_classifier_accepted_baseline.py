"""Shared interface on the accepted T04/N02 baseline bundle (models/nlp_baseline, not in Git).

The SHA-256 of the file is compared with reports/nlp/training/run_metadata.json before
the file is deserialized. Skipped when the file is absent or is a different (for example,
locally retrained) bundle: such a file is not evidence for the accepted baseline.
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BUNDLE = ROOT / "models" / "nlp_baseline" / "tfidf_logreg_baseline.joblib"
REPORT = json.loads((ROOT / "reports/nlp/training/run_metadata.json").read_text(encoding="utf-8"))
if not BUNDLE.is_file():
    pytest.skip("accepted baseline bundle is not installed: models/nlp_baseline/tfidf_logreg_baseline.joblib",
                allow_module_level=True)
_actual = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
if _actual != REPORT["bundle_sha256"]:
    pytest.skip(f"models/nlp_baseline holds a different bundle (SHA-256 {_actual[:12]}…), "
                f"not the accepted {REPORT['bundle_sha256'][:12]}…", allow_module_level=True)

from src import bot, classifier as c, train_baseline as base  # noqa: E402
from tools import predict_text  # noqa: E402

TEXTS = base._smoke_texts()
RECORDED = json.loads((ROOT / "reports/nlp/training/bundle_roundtrip.json").read_text(encoding="utf-8"))["fixed_texts"]


@pytest.fixture(scope="module")
def accepted():
    model = c.BaselineClassifier()
    model.predict(TEXTS[0])
    return model


def test_accepted_baseline_contract_and_recorded_predictions(accepted):
    bundle = accepted._get_bundle()
    raw = base.predict_texts(bundle, TEXTS)
    assert [base.sha256_text(t) for t in TEXTS] == RECORDED["text_sha256"]
    assert bundle["label_mapping"] == {"negative": 0, "neutral": 1, "positive": 2}
    assert [int(v) for v in bundle["classifier"].classes_] == RECORDED["probability_class_ids"] == [0, 1, 2]
    for i, text in enumerate(TEXTS):
        result = accepted.predict(text)
        assert set(result) == {"label", "label_id", "confidence", "probabilities", "truncated"}
        assert list(result["probabilities"]) == c.EXPECTED_CLASS_NAMES
        values = list(result["probabilities"].values())
        assert all(math.isfinite(v) and 0 <= v <= 1 for v in values)
        assert math.isclose(sum(values), 1, abs_tol=1e-6)
        assert result["label_id"] == raw["label_ids"][i] == RECORDED["predicted_label_ids"][i]
        assert result["label"] == RECORDED["predicted_labels"][i]
        assert values == pytest.approx(raw["probabilities"][i], abs=1e-12)
        assert result["confidence"] == max(values) and result["truncated"] is False


def test_accepted_baseline_data_version(accepted):
    version = accepted.data_version
    for key in ("train_ids_sha256", "validation_ids_sha256", "train_parquet_sha256",
                "validation_parquet_sha256", "tfidf_state_sha256", "winner_C"):
        assert version[key] == REPORT[key]
    assert version["final_test_used"] is False


def test_bot_cli_and_evaluation_paths_agree(accepted, monkeypatch, capsys):
    monkeypatch.setattr(c, "_default_classifier", accepted)
    batch = c.predict_batch(TEXTS)
    for text, evaluated in zip(TEXTS, batch):
        direct = c.predict(text)
        assert evaluated == direct
        assert bot.classify_text_for_bot(text) == bot.format_prediction(direct)
        assert predict_text.main([text]) == 0
        assert json.loads(capsys.readouterr().out) == direct
