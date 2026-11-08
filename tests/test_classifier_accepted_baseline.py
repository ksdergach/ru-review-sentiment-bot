"""Shared interface on the accepted T04/N02 baseline bundle (models/nlp_baseline, not in Git).

The SHA-256 of the file is compared with reports/nlp/training/run_metadata.json before
the file is deserialized. Skipped when the file is absent or is a different (for example,
locally retrained) bundle: such a file is not evidence for the accepted baseline.
Uses the fixed smoke texts and the prepared validation split, never the final test split.
"""
import hashlib
import json
import math
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
BUNDLE = ROOT / "models" / "nlp_baseline" / "tfidf_logreg_baseline.joblib"
REPORT_PATH = ROOT / "reports/nlp/training/run_metadata.json"
REPORT = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
if not BUNDLE.is_file():
    pytest.skip("accepted baseline bundle is not installed: models/nlp_baseline/tfidf_logreg_baseline.joblib",
                allow_module_level=True)
_actual = hashlib.sha256(BUNDLE.read_bytes()).hexdigest()
if _actual != REPORT["bundle_sha256"]:
    pytest.skip(f"models/nlp_baseline holds a different bundle (SHA-256 {_actual[:12]}…), "
                f"not the accepted {REPORT['bundle_sha256'][:12]}…", allow_module_level=True)

import joblib  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from src import bot, classifier as c, train_baseline as base  # noqa: E402
from tools import predict_text, report_nlp_baseline as n02  # noqa: E402

TEXTS = base._smoke_texts()
RECORDED = json.loads((ROOT / "reports/nlp/training/bundle_roundtrip.json").read_text(encoding="utf-8"))["fixed_texts"]
METRICS = json.loads((ROOT / "reports/nlp/baseline_metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module", autouse=True)
def guards():
    """Network and fitting are refused for the whole module, before the first load.

    Attempts are recorded as well as refused: the classifier wraps load errors, so a
    swallowed attempt must still fail the test.
    """
    attempts = []
    loads = []

    def refuse(name):
        def call(*args, **kwargs):
            attempts.append(name)
            raise AssertionError(f"inference must not use {name}")
        return call

    original_load = c.load_bundle

    def counting_load(*args, **kwargs):
        loads.append(Path(args[0]))
        return original_load(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        for name in ("connect", "connect_ex"):
            patch.setattr(socket.socket, name, refuse(f"socket.{name}"))
        patch.setattr(socket, "create_connection", refuse("socket.create_connection"))
        patch.setattr(socket, "getaddrinfo", refuse("socket.getaddrinfo"))
        for cls in (CountVectorizer, TfidfVectorizer, LogisticRegression):
            for name in ("fit", "fit_transform"):
                if name in vars(cls):
                    patch.setattr(cls, name, refuse(f"{cls.__name__}.{name}"))
        for name in ("run_training", "fit_vectorizer_on_train", "build_classifier", "save_bundle"):
            patch.setattr(base, name, refuse(f"train_baseline.{name}"))
        patch.setattr(c, "load_bundle", counting_load)
        yield {"attempts": attempts, "loads": loads}
    assert attempts == []


@pytest.fixture(scope="module")
def accepted(guards):
    assert guards["loads"] == []
    model = c.BaselineClassifier()
    model.predict(TEXTS[0])
    assert guards["loads"] == [BUNDLE] and guards["attempts"] == []
    return model


def test_first_load_ran_under_guards(accepted, guards):
    assert guards["loads"].count(BUNDLE) == 1 and guards["attempts"] == []
    # The loaded state is the N02 model itself, not just a file with the same name.
    assert n02.model_state_sha256(accepted._get_bundle()) == METRICS["model"]["state_sha256"]


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


def test_full_validation_matches_recorded_n02_predictions(accepted):
    config = base.load_config(ROOT / "configs/nlp_baseline.json")
    path = ROOT / config["validation_path"]
    if not path.is_file() or base.sha256_file(path) != REPORT["validation_parquet_sha256"]:
        pytest.skip(f"prepared validation split matching the run report is not available: {config['validation_path']}")
    frame = base.load_prepared_split(path, split_name="validation", text_column=config["text_column"],
                                     label_column=config["label_column"], id_column=config["id_column"])
    assert base.ids_sha256(frame, id_column=config["id_column"],
                           label_column=config["label_column"]) == REPORT["validation_ids_sha256"]
    texts = frame[config["text_column"]].tolist()
    truth = frame[config["label_column"]].astype(int).tolist()
    results = accepted.predict_batch(texts, max_text_length=None)
    reference = base.predict_texts(accepted._get_bundle(), texts)
    assert len(results) == len(texts) == METRICS["rows"] == 10396
    predicted = [r["label_id"] for r in results]
    assert predicted == reference["label_ids"]
    worst = max(abs(a - b) for r, ref in zip(results, reference["probabilities"])
                for a, b in zip(r["probabilities"].values(), ref))
    assert worst <= 1e-12
    records = [{"record_id": str(rid), "true_label_id": y, "predicted_label_id": p}
               for rid, y, p in zip(frame[config["id_column"]], truth, predicted)]
    assert base.sha256_json(records) == METRICS["validation_predictions_sha256"]
    assert base.validation_confusion_matrix(truth, predicted) == METRICS["confusion_matrix"]["matrix"]
    assert base.validation_metrics(truth, predicted)["macro_f1"] == METRICS["macro_f1"]


def _signed_copy(tmp_path, change):
    """Copy of the accepted bundle with one change and its own report, so the checksum passes."""
    bundle = joblib.load(BUNDLE)
    change(bundle)
    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, path, compress=3)
    report = tmp_path / "run_metadata.json"
    report.write_text(json.dumps({**REPORT, "bundle_sha256": base.sha256_file(path)}), encoding="utf-8")
    return c.BaselineClassifier(path, report_path=report)


def _swap_mapping(bundle):
    bundle["label_mapping"] = {"negative": 2, "neutral": 1, "positive": 0}


def _reverse_classes(bundle):
    lr = bundle["classifier"]
    lr.classes_ = lr.classes_[::-1].copy()
    lr.coef_ = lr.coef_[::-1].copy()
    lr.intercept_ = lr.intercept_[::-1].copy()


@pytest.mark.parametrize("change, reason", [(_swap_mapping, "label_mapping"), (_reverse_classes, "classes_")])
def test_class_order_is_checked_on_saved_artifact(tmp_path, change, reason):
    with pytest.raises(c.ModelUnavailableError, match=reason):
        _signed_copy(tmp_path, change).predict(TEXTS[0])


def test_concurrent_first_load_of_accepted_bundle(guards):
    from concurrent.futures import ThreadPoolExecutor

    before = guards["loads"].count(BUNDLE)
    model = c.BaselineClassifier()
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(model.predict, TEXTS * 4))
    assert guards["loads"].count(BUNDLE) == before + 1
    assert [r["label_id"] for r in results] == RECORDED["predicted_label_ids"] * 4
