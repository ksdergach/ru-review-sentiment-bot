"""T13: финальное сравнение на искусственном репозитории; настоящий test не читается.

Ожидаемые метрики посчитаны вручную, независимо от sklearn и src.train_baseline.
"""
import csv
import json
import multiprocessing
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import evaluate as ev, train_baseline as tb  # noqa: E402
from test_classifier import make_real_bundle, report_of  # noqa: E402

NAMES = {0: "negative", 1: "neutral", 2: "positive"}
REAL_SET_TORCH_THREADS = ev._set_torch_threads
CLEAN_GIT = {"commit": "b" * 40, "frozen_commit_is_ancestor": True, "tracked_changes": False,
             "evaluator_in_head": True, "evaluator_matches_head": True}
# Последние две пары — повторы: точный (регистр и пробелы) и близкий, подтверждённый в T03.
TEST_ROWS = [
    ("test.parquet#row=0", "Ужасный фильм, зря потратил вечер", 0),
    ("test.parquet#row=1", "Скучно и затянуто", 0),
    ("test.parquet#row=2", "Обычное кино на один раз", 1),
    ("test.parquet#row=3", "Прекрасная игра актёров", 2),
    ("test.parquet#row=4", "Хороший фильм", 2),
    ("test.parquet#row=5", "хороший   фильм", 2),
    ("test.parquet#row=6", "Очень понравился сюжет и музыка", 2),
    ("test.parquet#row=7", "Очень понравился сюжет и музыка!", 2),
]
TEXTS = [text for _, text, _ in TEST_ROWS]
LABELS = [label for _, _, label in TEST_ROWS]
# Baseline ни разу не предсказывает neutral.
PREDICTIONS = {"baseline": [0, 2, 0, 2, 2, 2, 0, 0], "rubert": [0, 0, 1, 2, 2, 2, 2, 1]}
VERSION = {"train_ids_sha256": "1" * 64, "train_rows": 6, "validation_ids_sha256": "2" * 64,
           "validation_rows": 2, "final_test_used": False}


def write_csv(path: Path, header, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


class Clock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class FakeModel:
    """Общий интерфейс src.classifier: ленивая загрузка, data_version и predict_batch."""

    def __init__(self, backend, data_version, clock=None, load_cost=0.0, batch_cost=0.0):
        self.backend, self.version, self.clock = backend, data_version, clock
        self.load_cost, self.batch_cost = load_cost, batch_cost
        self.data_version, self.calls = None, []

    def _get_bundle(self):
        if self.clock:
            self.clock.now += self.load_cost
        self.data_version = self.version

    def predict_batch(self, texts, *, max_text_length):
        assert self.data_version is not None and max_text_length is None
        self.calls.append(list(texts))
        if self.clock:
            self.clock.now += self.batch_cost / (2 if tuple(texts) == ev.WARMUP_TEXTS else 1)
        answers = dict(zip(TEXTS, PREDICTIONS[self.backend]))
        return [{"label_id": answers.get(t, 0), "truncated": False} for t in texts]


def rubert_version(version):
    checksums = {"data_prep_split_ids_path": version["t03_split_ids_sha256"],
                 "data_prep_audit_path": version["t03_audit_sha256"]}
    return {**{k: version[k] for k in ev.SHARED_VERSION_KEYS[:4]}, "checksums": checksums, "final_test_used": False}


MAPPING = {"negative": 0, "neutral": 1, "positive": 2}
PREP = {**{k: v for k, v in VERSION.items() if k != "final_test_used"},
        "train_parquet_sha256": "3" * 64, "validation_parquet_sha256": "4" * 64}
RUBERT_FILES = ("config.json", "label_mapping.json", "metadata.json", "model.safetensors", "tokenizer.json",
                "tokenizer_config.json", "training_config.json")
BASELINE_REPORT = "reports/nlp/training/run_metadata.json"
RUBERT_REPORT = "reports/transformer/run_metadata.json"


def sha(root, rel):
    return tb.sha256_file(root / rel)


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def baseline_report(root, model_rel, model_sha):
    return {"bundle_path": model_rel, "bundle_sha256": model_sha, "config_path": "configs/nlp_baseline.json",
            "config_sha256": sha(root, "configs/nlp_baseline.json"), "final_test_used": False,
            "selection_metric": "validation_macro_f1",
            "t03_audit_sha256": sha(root, "reports/data_prep/audit.json"),
            "t03_split_ids_sha256": sha(root, "reports/data_prep/split_ids.csv"),
            "split_independence": {"movie_id_overlap": 0, "normalized_text_overlap": 0, "record_id_overlap": 0},
            **PREP}


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Искусственный репозиторий той же структуры, что утверждённая фиксация T12."""
    pytest.importorskip("pyarrow")
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    tb.write_json(root / "configs/nlp_baseline.json", {
        "data_prep_audit_path": "reports/data_prep/audit.json",
        "data_prep_split_ids_path": "reports/data_prep/split_ids.csv",
        "text_processing": {"manual_normalization": "none", "truncation": None}, "label_mapping": MAPPING})
    tb.write_json(root / "configs/transformer.json", {"model_name": "tiny/rubert", "revision": "c" * 40,
                                                      "max_length": 64, "truncation_side": "right",
                                                      "label_mapping": MAPPING})
    tb.write_json(root / "configs/data_prep.json", {"report_dir": "reports/data_prep", "near_duplicates": {
        "decision_file": "configs/near_duplicate_decisions.csv"}})
    for rel in ("requirements.txt", "src/__init__.py", "src/classifier.py", "src/train_baseline.py",
                "src/train_transformer.py", "src/prepare_data.py", "reports/freeze.md",
                "reports/freeze_checks.json", "reports/freeze_review_fix.json"):
        write_bytes(root / rel, f"frozen {rel}\n".encode())
    git(root, "init", "-q")
    git(root, "add", "src/classifier.py")
    git(root, "commit", "-qm", "frozen classifier")
    frozen_commit = git(root, "rev-parse", "HEAD")
    shutil.copyfile(ev.__file__, root / "src/evaluate.py")
    git(root, "add", "src/evaluate.py")
    git(root, "commit", "-qm", "evaluator")
    # The original test is only hashed, never decoded: arbitrary bytes suffice.
    write_bytes(root / "NLP_dataset/test-00000-of-00001.parquet", b"original test bytes")
    prep = root / "reports/data_prep"
    counts = {name: LABELS.count(i) for i, name in NAMES.items()}
    tb.write_json(prep / "audit.json", {
        "status": "completed", "final_summary": {"test": {"rows": len(TEST_ROWS), **counts}},
        "source_files": {"test": {"path": "NLP_dataset/test-00000-of-00001.parquet",
                                  "sha256_after": sha(root, "NLP_dataset/test-00000-of-00001.parquet")}},
        "final_movie_id_overlap": {"train__test": 0, "train__validation": 0, "validation__test": 0},
        "final_exact_text_overlap": {"train__test": 0, "train__validation": 0, "validation__test": 0},
        "exact_dedup": {"test_exact_duplicate_groups": 1}, "near_duplicates": {"near_pairs_within_test": 1}})
    write_csv(prep / "split_ids.csv", ["record_id", "split", "label_name", "label_id"],
              [["train.parquet#row=0", "train", "positive", 2]]
              + [[rid, "test", NAMES[label], label] for rid, _, label in TEST_ROWS])
    write_csv(prep / "near_duplicate_candidates.csv",
              ["pair_id", "left_record_id", "left_split", "right_record_id", "right_split", "similarity"],
              [["p1", TEST_ROWS[6][0], "test", TEST_ROWS[7][0], "test", "0.95"],
               ["p2", "train.parquet#row=0", "train", "train.parquet#row=9", "train", "0.93"],
               ["p3", TEST_ROWS[0][0], "test", TEST_ROWS[1][0], "test", "0.91"]])
    write_csv(root / "configs/near_duplicate_decisions.csv", ["pair_id", "decision", "note"],
              [["p1", "duplicate", ""], ["p2", "duplicate", ""], ["p3", "not_duplicate", ""]])
    frame = pd.DataFrame({"record_id": [r for r, _, _ in TEST_ROWS], "review_text": TEXTS, "label_id": LABELS,
                          "label_name": [NAMES[x] for x in LABELS], "review_language": "ru",
                          "movie_id": list(range(len(TEST_ROWS)))})
    test_path = root / "data/processed/test.parquet"
    test_path.parent.mkdir(parents=True)
    frame.to_parquet(test_path, index=False)

    baseline_rel = "models/nlp_baseline/baseline.joblib"
    write_bytes(root / baseline_rel, b"synthetic baseline bundle")
    for name in RUBERT_FILES:
        write_bytes(root / "models/transformer" / name, f"synthetic {name}".encode())
    tb.write_json(root / BASELINE_REPORT, baseline_report(root, baseline_rel, sha(root, baseline_rel)))
    rubert_files = {name: sha(root, f"models/transformer/{name}") for name in RUBERT_FILES}
    tb.write_json(root / RUBERT_REPORT, {
        "bundle_files_sha256": rubert_files, "final_test_used": False, "selection_metric": "validation_macro_f1",
        "data_version": {**{k: VERSION[k] for k in ("train_rows", "validation_rows", "train_ids_sha256",
                                                    "validation_ids_sha256")},
                         "final_test_used": False, "train_ids_equal_baseline": True,
                         "split_independence": {"movie_id_overlap": 0, "record_id_overlap": 0},
                         "checksums": {"train_path": PREP["train_parquet_sha256"],
                                       "validation_path": PREP["validation_parquet_sha256"],
                                       "data_prep_audit_path": sha(root, "reports/data_prep/audit.json"),
                                       "data_prep_split_ids_path": sha(root, "reports/data_prep/split_ids.csv")}}})
    variables = {"T13_SYNTHETIC_THREADS": "1"}
    for name, value in variables.items():
        monkeypatch.setenv(name, value)
    freeze = {
        "schema": ev.FREEZE_SCHEMA, "protocol_version": ev.PROTOCOL_VERSION, "protocol_path": "reports/freeze.md",
        "status": "approved", "pending_approval": [], "approved_by": ["team"], "approved_at": "2026-12-01",
        "confirmed_decisions": {"bot_model": "baseline"}, "frozen_commit": frozen_commit,
        "final_test_used_before_freeze": False, "bot_model": "baseline",
        "bot_model_selected_by": "validation_and_resources", "label_mapping": MAPPING,
        "data_prep": {"audit_sha256": sha(root, "reports/data_prep/audit.json"),
                      "split_ids_sha256": sha(root, "reports/data_prep/split_ids.csv"), **PREP},
        "test": {"path": "data/processed/test.parquet", "parquet_sha256": tb.sha256_file(test_path),
                 "rows": len(TEST_ROWS), "ids_sha256": tb.ids_sha256(frame, id_column="record_id", label_column="label_id"),
                 "source_parquet_sha256": sha(root, "NLP_dataset/test-00000-of-00001.parquet")},
        "models": {
            "baseline": {"model_path": baseline_rel, "model_sha256": sha(root, baseline_rel),
                         "bytes": (root / baseline_rel).stat().st_size, "report_path": BASELINE_REPORT,
                         "report_sha256": sha(root, BASELINE_REPORT), "config_path": "configs/nlp_baseline.json",
                         "manual_normalization": "none", "truncation": None},
            "rubert": {"model_path": "models/transformer", "report_path": RUBERT_REPORT,
                       "report_sha256": sha(root, RUBERT_REPORT), "bundle_files_sha256": rubert_files,
                       "config_path": "configs/transformer.json", "base_model": "tiny/rubert",
                       "base_revision": "c" * 40, "max_length_tokens": 64, "truncation_side": "right"},
        },
        "inference": {"batch_size": 3, "device": "cpu", "num_threads": 2},
        "runtime_environment_variables": variables,
        "text_protocol": {"models": ["baseline", "rubert"], "runs": 1, "max_text_length": None,
                          "labels": [0, 1, 2], "zero_division": 0},
        "environment": {"machine": ev.machine_info(), "packages": {"numpy": ev.installed_version("numpy")}},
    }
    pinned = [*ev.REQUIRED_PINNED, BASELINE_REPORT, RUBERT_REPORT]
    freeze["file_sha256"] = {rel: sha(root, rel) for rel in pinned}
    tb.write_json(root / "reports/freeze.json", freeze)
    monkeypatch.setattr(ev, "ACCEPTED_T12_SHA256", {rel: sha(root, rel) for rel in (
        "reports/freeze.json", "reports/freeze.md", "reports/freeze_checks.json", "reports/freeze_review_fix.json")})
    threads = []
    monkeypatch.setattr(ev, "_set_torch_threads", threads.append)
    version = {**VERSION, "t03_split_ids_sha256": freeze["data_prep"]["split_ids_sha256"],
               "t03_audit_sha256": freeze["data_prep"]["audit_sha256"]}
    return root, freeze, version


def edit_freeze(root, update):
    """Изменить фиксацию так, будто это принятая версия: проверяется следующая защита."""
    path = root / "reports/freeze.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    update(data)
    tb.write_json(path, data)
    ev.ACCEPTED_T12_SHA256["reports/freeze.json"] = tb.sha256_file(path)


def repin(root, *rels):
    """Согласованно обновить file_sha256 (и report_sha256) после изменения закреплённых файлов."""
    def update(data):
        for rel in rels:
            data["file_sha256"][rel] = sha(root, rel)
            for entry in data["models"].values():
                if entry["report_path"] == rel:
                    entry["report_sha256"] = sha(root, rel)
    edit_freeze(root, update)


def edit_json(root, rel, update):
    data = json.loads((root / rel).read_text(encoding="utf-8"))
    update(data)
    tb.write_json(root / rel, data)


def fake_factory(version, created=None, **costs):
    def factory(backend, model_path, *, report_path):
        model = FakeModel(backend, version if backend == "baseline" else rubert_version(version), **costs)
        if created is not None:
            created[backend] = model
        return model
    return factory


def run(root, version, **kwargs):
    kwargs.setdefault("factory", fake_factory(version))
    return ev.evaluate(root / "reports/freeze.json", root / "out", root=root, **kwargs)


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #
def test_metrics_match_hand_computed_values():
    truth = [0, 0, 0, 1, 1, 2, 2, 2, 2, 2]
    predicted = [0, 0, 2, 1, 0, 2, 2, 2, 1, 0]
    metrics = ev.compute_metrics(truth, predicted)
    assert metrics["confusion_matrix"] == [[2, 0, 1], [1, 1, 0], [1, 1, 3]]
    assert metrics["rows"] == 10 and metrics["accuracy"] == pytest.approx(0.6)
    expected = {"negative": (1 / 2, 2 / 3, 4 / 7, 3), "neutral": (1 / 2, 1 / 2, 1 / 2, 2),
                "positive": (3 / 4, 3 / 5, 2 / 3, 5)}
    for name, (precision, recall, f1, support) in expected.items():
        assert metrics["per_class"][name] == {"precision": pytest.approx(precision), "recall": pytest.approx(recall),
                                              "f1": pytest.approx(f1), "support": support}
    assert metrics["macro_f1"] == pytest.approx(73 / 126)
    assert metrics["predicted_counts"] == {"negative": 4, "neutral": 2, "positive": 4}


def test_neutral_without_predictions_stays_in_matrix_and_macro_f1():
    metrics = ev.compute_metrics([0, 1, 1, 2], [0, 0, 2, 2])
    assert metrics["confusion_matrix"] == [[1, 0, 0], [1, 0, 1], [0, 0, 1]]
    assert metrics["per_class"]["neutral"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 2}
    assert metrics["predicted_counts"]["neutral"] == 0
    assert metrics["macro_f1"] == pytest.approx(4 / 9) and metrics["accuracy"] == pytest.approx(0.5)
    # Neutral отсутствует и в истине, и в ответах: матрица остаётся 3×3.
    assert ev.compute_metrics([0, 2], [0, 2])["confusion_matrix"] == [[1, 0, 0], [0, 0, 0], [0, 0, 1]]


@pytest.mark.parametrize("truth,predicted", [([0, 1], [0]), ([], []), ([0, 3], [0, 1]), ([0, 1], [0, -1]),
                                             ([0, 1, 2], [0, 1.9, 2]), ([0, 1, 2], [0, 1.0, 2]),
                                             ([0, 1.5, 2], [0, 1, 2]), ([0, 1], [0, True]), ([0, 1], [0, "1"])])
def test_metrics_reject_invalid_input(truth, predicted):
    with pytest.raises(ev.DataMismatchError):
        ev.compute_metrics(truth, predicted)


# --------------------------------------------------------------------------- #
# Одинаковые входы и версии данных
# --------------------------------------------------------------------------- #
def make_runs():
    version = {"parquet_sha256": "c" * 64, "ids_sha256": "d" * 64, "rows": 3}
    return {b: {"record_ids": ["a", "b", "c"], "labels": [0, 1, 2], "predictions": [0, 1, 2],
                "test_version": dict(version)} for b in ev.BACKENDS}


def test_same_inputs_accepted():
    ev.check_same_inputs(make_runs())


@pytest.mark.parametrize("key,value", [
    ("record_ids", ["a", "c", "b"]), ("record_ids", ["a", "b", "x"]), ("labels", [0, 1, 1]),
    ("record_ids", ["a", "b"]), ("predictions", [0, 1]),
    ("test_version", {"parquet_sha256": "e" * 64, "ids_sha256": "d" * 64, "rows": 3}),
])
def test_different_inputs_rejected(key, value):
    runs = make_runs()
    runs["rubert"][key] = value
    with pytest.raises(ev.DataMismatchError):
        ev.check_same_inputs(runs)


def test_missing_model_run_rejected():
    runs = make_runs()
    del runs["rubert"]
    with pytest.raises(ev.DataMismatchError, match="Expected runs"):
        ev.check_same_inputs(runs)


def test_model_versions_normalized_and_compared(repo):
    _, freeze, version = repo
    versions = {"baseline": ev.model_data_version("baseline", version),
                "rubert": ev.model_data_version("rubert", rubert_version(version))}
    assert versions["baseline"] == versions["rubert"]
    ev.check_model_versions(versions, freeze)
    with pytest.raises(ev.DataMismatchError, match="train_rows"):
        ev.check_model_versions({**versions, "rubert": {**versions["rubert"], "train_rows": 7}}, freeze)
    with pytest.raises(ev.DataMismatchError, match="T03 version"):
        ev.check_model_versions({**versions, "rubert": {**versions["rubert"], "t03_audit_sha256": "f" * 64}}, freeze)
    with pytest.raises(ev.DataMismatchError, match="Expected models"):
        ev.check_model_versions({"baseline": versions["baseline"]}, freeze)
    with pytest.raises(ev.DataMismatchError, match="final test"):
        ev.model_data_version("baseline", {**version, "final_test_used": True})
    with pytest.raises(ev.DataMismatchError, match="incomplete"):
        ev.model_data_version("rubert", version)


# --------------------------------------------------------------------------- #
# Финальный запуск на искусственных данных
# --------------------------------------------------------------------------- #
def test_final_run_writes_aggregated_reports_without_texts(repo):
    root, _, version = repo
    created = {}
    report = run(root, version, factory=fake_factory(version, created))
    out = root / "out"
    assert sorted(p.name for p in out.iterdir()) == sorted([
        "final_metrics.json", "final_table.csv", "final_report.md", "predictions.csv",
        "confusion_matrix_baseline.csv", "confusion_matrix_rubert.csv",
        "confusion_matrix_baseline.svg", "confusion_matrix_rubert.svg"])
    assert not [p for p in out.parent.iterdir() if p.name.startswith(".out.")]
    # Hand-computed: baseline never predicts neutral.
    baseline = report["models"]["baseline"]["metrics"]
    assert baseline["confusion_matrix"] == [[1, 0, 1], [1, 0, 0], [2, 0, 3]]
    assert baseline["accuracy"] == pytest.approx(4 / 8)
    # F1: negative 1/3 (P 1/4, R 1/2), neutral 0, positive 2/3 (P 3/4, R 3/5).
    assert baseline["macro_f1"] == pytest.approx(1 / 3)
    assert baseline["per_class"]["neutral"] == {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 1}
    rubert = report["models"]["rubert"]["metrics"]
    assert rubert["confusion_matrix"] == [[2, 0, 0], [0, 1, 0], [0, 1, 4]]
    assert rubert["accuracy"] == pytest.approx(7 / 8)
    assert report["test"]["rows"] == len(TEST_ROWS) and report["bot_model"] == "baseline"
    assert report["run"]["command"] == "python -m src.evaluate --run-final --freeze reports/freeze.json --out-dir out"
    assert report["run"]["freeze_sha256"] == sha(root, "reports/freeze.json")
    assert report["run"]["protocol_version"] == "t12/1" and report["run"]["frozen_commit"] == git(root, "rev-parse", "HEAD^")
    # RuBERT is better on this synthetic test, but the bot model chosen on validation stays.
    assert rubert_better(report) and report["bot_model"] == "baseline"
    assert report["bot_model_changed_by_test"] is False
    assert baseline["errors"] == {"total": 4, "by_true_class": {"negative": 1, "neutral": 1, "positive": 2},
                                  "by_predicted_class": {"negative": 3, "neutral": 0, "positive": 1}}
    assert report["run"]["git_commit"] == git(root, "rev-parse", "HEAD")
    # Both models processed the same texts in the same order and batch layout.
    assert created["baseline"].calls == created["rubert"].calls
    assert created["baseline"].calls[0] == list(ev.WARMUP_TEXTS)
    assert sum(created["baseline"].calls[1:], []) == TEXTS
    assert [len(batch) for batch in created["rubert"].calls[1:]] == [3, 3, 2]

    with (out / "predictions.csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["record_id"] for r in rows] == [r for r, _, _ in TEST_ROWS]
    assert [int(r["baseline_label_id"]) for r in rows] == PREDICTIONS["baseline"]
    with (out / "final_table.csv").open(encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
    assert [r["model"] for r in table] == ["baseline", "rubert"]
    assert float(table[0]["neutral_precision"]) == 0.0 and table[0]["bot_model"] == "True"
    with (out / "confusion_matrix_baseline.csv").open(encoding="utf-8") as fh:
        assert list(csv.reader(fh)) == [["true\\predicted", "negative", "neutral", "positive"],
                                        ["negative", "1", "0", "1"], ["neutral", "1", "0", "0"],
                                        ["positive", "2", "0", "3"]]
    written = "\n".join(p.read_text(encoding="utf-8") for p in out.iterdir())
    for text in TEXTS:
        assert text not in written
    assert "Хороший" not in written and "понравился" not in written
    saved = json.loads((out / "final_metrics.json").read_text(encoding="utf-8"))
    assert saved["models"]["rubert"]["metrics"] == rubert
    assert saved["outputs_sha256"] == {name: tb.sha256_file(out / name) for name in ev.OUTPUT_FILES
                                       if name != "final_metrics.json"}


def rubert_better(report):
    models = report["models"]
    return models["rubert"]["metrics"]["macro_f1"] > models["baseline"]["metrics"]["macro_f1"]


def test_repeat_sensitivity_keeps_first_copy_and_main_test_unchanged(repo):
    root, _, version = repo
    report = run(root, version)
    sens = report["repeat_sensitivity"]
    # Exact group rows 4/5 and confirmed near pair 6/7; the not_duplicate pair is ignored.
    assert sens["applied"] and sens["groups"] == 2 and sens["rows_dropped"] == 2 and sens["rows"] == 6
    assert report["models"]["baseline"]["metrics"]["rows"] == len(TEST_ROWS)
    # Without rows 5 and 7: baseline [0,2,0,2,2,0] vs truth [0,0,1,2,2,2].
    assert sens["models"]["baseline"]["confusion_matrix"] == [[1, 0, 1], [1, 0, 0], [1, 0, 2]]
    assert sens["models"]["baseline"]["accuracy"] == pytest.approx(3 / 6)
    assert sens["models"]["baseline"]["delta_accuracy"] == pytest.approx(0.0)
    assert sens["models"]["rubert"]["accuracy"] == pytest.approx(6 / 6)
    assert sens["models"]["rubert"]["delta_accuracy"] == pytest.approx(1 - 7 / 8)


def test_repeats_must_match_t03_audit(repo):
    root, _, version = repo
    audit = root / "reports/data_prep/audit.json"
    data = json.loads(audit.read_text(encoding="utf-8"))
    data["near_duplicates"]["near_pairs_within_test"] = 2
    tb.write_json(audit, data)
    sync_t03(root, version)
    with pytest.raises(ev.DataMismatchError, match="T03 audit"):
        run(root, version)
    assert not (root / "out").exists()


def test_load_warmup_and_inference_timed_separately(repo):
    root, _, version = repo
    clock = Clock()
    report = run(root, version, clock=clock,
                 factory=fake_factory(version, clock=clock, load_cost=5.0, batch_cost=1.0))
    for backend in ev.BACKENDS:
        timing = report["models"][backend]["timing"]
        assert timing["load_seconds"] == pytest.approx(5.0)
        assert timing["warmup_seconds"] == pytest.approx(0.5)
        assert timing["inference_seconds"] == pytest.approx(3.0)
        assert timing["batches"] == 3 and timing["batch_size"] == 3 and timing["rows"] == len(TEST_ROWS)
        assert timing["ms_per_text"] == pytest.approx(3000 / len(TEST_ROWS))
    assert report["inference_protocol"]["batch_size"] == 3
    assert report["inference_protocol"]["order"] == ["baseline", "rubert"]
    assert {"machine", "libraries", "runtime_environment_variables"} <= set(report["environment"])
    assert report["environment"]["runtime_environment_variables"] == {"T13_SYNTHETIC_THREADS": "1"}


def test_shared_classifier_interface_with_real_baseline_bundle(repo):
    """Baseline проходит через настоящий src.classifier.BaselineClassifier."""
    root, freeze, version = repo
    model_rel = "models/real_baseline.joblib"
    model_path = make_real_bundle(root / "models", training_update=lambda t: t.update({
        **{k: v for k, v in baseline_report(root, model_rel, None).items()
           if k not in ("bundle_sha256", "config_path")}}))
    report_rel = report_of(model_path).relative_to(root).as_posix()

    def point_to_real_bundle(f):
        f["models"]["baseline"].update(model_path=model_rel, model_sha256=sha(root, model_rel),
                                       bytes=model_path.stat().st_size, report_path=report_rel)
        f["file_sha256"].pop(BASELINE_REPORT)
    edit_freeze(root, point_to_real_bundle)
    repin(root, report_rel)
    fake = fake_factory(version)

    def factory(backend, model_path, *, report_path):
        if backend == "baseline":
            return ev.c.create_classifier(backend, model_path, report_path=report_path)
        return fake(backend, model_path, report_path=report_path)

    report = run(root, version, factory=factory)
    # Expected labels from the training module directly, not through src.classifier.
    expected = tb.predict_texts(tb.load_bundle(model_path), TEXTS)["label_ids"]
    with (root / "out/predictions.csv").open(encoding="utf-8") as fh:
        assert [int(r["baseline_label_id"]) for r in csv.DictReader(fh)] == expected
    assert report["models"]["baseline"]["metrics"] == ev.compute_metrics(LABELS, expected)


def test_rubert_threads_set_from_freeze_before_loading(repo):
    root, _, version = repo
    run(root, version)
    assert ev._set_torch_threads.__self__ == [2]


# --------------------------------------------------------------------------- #
# Графики и запись результатов
# --------------------------------------------------------------------------- #
def test_confusion_matrix_svg_on_synthetic_matrix():
    from xml.etree import ElementTree
    svg = ev.confusion_matrix_svg([[5, 0, 1], [2, 0, 2], [0, 0, 7]], "Confusion matrix: synthetic")
    tree = ElementTree.fromstring(svg)
    ns = {"svg": "http://www.w3.org/2000/svg"}
    cells = [r for r in tree.iterfind(".//svg:g/svg:rect", ns)]
    assert len(cells) == 9
    # Row-major order; zero cells use the neutral fill, full-row cells the darkest step.
    fills = [r.get("fill") for r in cells]
    assert fills[1] == fills[4] == fills[7] == ev._ZERO_FILL and fills[8] == ev._SEQUENTIAL[-1]
    texts = [t.text for t in tree.iterfind(".//svg:g/svg:text", ns)]
    assert texts[0::2] == ["5", "0", "1", "2", "0", "2", "0", "0", "7"]
    assert texts[1::2][:3] == ["83.3%", "0.0%", "16.7%"]
    labels = [t.text for t in tree.iterfind("svg:text", ns)]
    assert {"negative", "neutral", "positive", "Истинный класс", "Предсказанный класс"} <= set(labels)


def test_existing_results_are_not_overwritten(repo, monkeypatch):
    root, _, version = repo
    (root / "out").mkdir()
    (root / "out/final_metrics.json").write_text("{}", encoding="utf-8")
    forbid_test_access(monkeypatch)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run(root, version, factory=forbid_models())
    assert (root / "out/final_metrics.json").read_text(encoding="utf-8") == "{}"


def test_write_failure_leaves_no_final_or_partial_output(repo, monkeypatch):
    root, _, version = repo

    def broken(*args, **kwargs):
        raise OSError("disk full")
    monkeypatch.setattr(ev, "confusion_matrix_svg", broken)
    with pytest.raises(OSError, match="disk full"):
        run(root, version)
    assert not (root / "out").exists()
    assert not [p.name for p in root.iterdir() if p.name.startswith(".out.")]


# --------------------------------------------------------------------------- #
# Повторы внутри test: граничные случаи
# --------------------------------------------------------------------------- #
def repeat_inputs(tmp_path, texts, pairs, decisions):
    frame = pd.DataFrame({"record_id": [f"t{i}" for i in range(len(texts))], "review_text": texts})
    write_csv(tmp_path / "candidates.csv", ["pair_id", "left_record_id", "left_split", "right_record_id", "right_split"],
              [[pid, a, "test", b, "test"] for pid, a, b in pairs])
    write_csv(tmp_path / "decisions.csv", ["pair_id", "decision"], list(decisions.items()))
    return frame, tmp_path / "candidates.csv", tmp_path / "decisions.csv"


def audit_of(exact, near):
    return {"exact_dedup": {"test_exact_duplicate_groups": exact}, "near_duplicates": {"near_pairs_within_test": near}}


def test_no_repeats_gives_no_sensitivity(tmp_path):
    frame, cand, dec = repeat_inputs(tmp_path, ["a", "b", "c"], [], {})
    groups = ev.repeat_groups_in_test(frame, cand, dec, audit_of(0, 0))
    assert groups == []
    assert ev.sensitivity(make_runs(), groups, {})["applied"] is False


def test_linked_repeats_form_one_group_keeping_first(tmp_path):
    # t3~t1 and t1~t0 are confirmed near pairs; t4 repeats t2 exactly (case and spaces).
    frame, cand, dec = repeat_inputs(tmp_path, ["one", "two", "Три", "four", "три "],
                                     [("p1", "t3", "t1"), ("p2", "t1", "t0")], {"p1": "duplicate", "p2": "duplicate"})
    groups = ev.repeat_groups_in_test(frame, cand, dec, audit_of(1, 2))
    assert sorted(groups) == [["t0", "t1", "t3"], ["t2", "t4"]]
    runs = {b: {"record_ids": [f"t{i}" for i in range(5)], "labels": [0, 0, 1, 0, 1],
                "predictions": [0, 2, 1, 2, 0], "test_version": {}} for b in ev.BACKENDS}
    main = {b: ev.compute_metrics(r["labels"], r["predictions"]) for b, r in runs.items()}
    result = ev.sensitivity(runs, groups, main)
    # Kept: t0 and t2; dropped t1, t3, t4. Main test is unchanged.
    assert result["rows"] == 2 and result["rows_dropped"] == 3 and result["groups"] == 2
    assert result["models"]["baseline"]["accuracy"] == pytest.approx(1.0)
    assert main["baseline"]["rows"] == 5 and main["baseline"]["accuracy"] == pytest.approx(2 / 5)


def test_rejected_or_unknown_repeat_pairs(tmp_path):
    frame, cand, dec = repeat_inputs(tmp_path, ["a", "b"], [("p1", "t0", "t1")], {"p1": "not_duplicate"})
    assert ev.repeat_groups_in_test(frame, cand, dec, audit_of(0, 0)) == []
    frame, cand, dec = repeat_inputs(tmp_path, ["a", "b"], [("p1", "t0", "t9")], {"p1": "duplicate"})
    with pytest.raises(ev.DataMismatchError, match="unknown test IDs"):
        ev.repeat_groups_in_test(frame, cand, dec, audit_of(0, 1))


# --------------------------------------------------------------------------- #
# Защита финального test
# --------------------------------------------------------------------------- #
def forbid_test_access(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("The final test must not be read before all preconditions pass")
    monkeypatch.setattr(tb, "load_prepared_split", forbidden)
    monkeypatch.setattr(ev, "load_frozen_test", forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)


def forbid_models():
    def factory(*args, **kwargs):
        pytest.fail("Models must not be loaded when preconditions fail")
    return factory


def refuse(root, error, message, monkeypatch, version=None):
    """Fail-closed: ни моделей, ни чтения test, ни итогового каталога."""
    with monkeypatch.context() as scoped:
        forbid_test_access(scoped)
        with pytest.raises(error, match=message):
            run(root, version, factory=forbid_models())
    assert not (root / "out").exists()


def sync_t03(root, version):
    """Согласованно перенести новую версию T03 в отчёты, фиксацию и версии моделей."""
    audit, split_ids = sha(root, "reports/data_prep/audit.json"), sha(root, "reports/data_prep/split_ids.csv")
    edit_json(root, BASELINE_REPORT, lambda r: r.update(t03_audit_sha256=audit, t03_split_ids_sha256=split_ids))
    edit_json(root, RUBERT_REPORT, lambda r: r["data_version"]["checksums"].update(
        data_prep_audit_path=audit, data_prep_split_ids_path=split_ids))
    edit_freeze(root, lambda f: f["data_prep"].update(audit_sha256=audit, split_ids_sha256=split_ids))
    repin(root, "reports/data_prep/audit.json", "reports/data_prep/split_ids.csv", BASELINE_REPORT, RUBERT_REPORT)
    version.update(t03_audit_sha256=audit, t03_split_ids_sha256=split_ids)


def test_accepted_t12_freeze_is_compatible_without_test_or_models(monkeypatch):
    """Настоящая фиксация T12 проходит статические проверки; test и модели не открываются."""
    forbid_test_access(monkeypatch)
    freeze, digest = ev.read_freeze(ev.DEFAULT_FREEZE_PATH)
    assert tb.sha256_file(ev.DEFAULT_FREEZE_PATH) == digest == ev.ACCEPTED_T12_SHA256["reports/freeze.json"]
    ev._check_accepted_t12(digest, ev.ROOT)
    assert freeze["frozen_commit"] == "5f22d93a8cff63b30fc129bb68a1cfbd4230995d"
    assert freeze["inference"] == {"batch_size": 64, "device": "cpu", "num_threads": 2}
    assert freeze["bot_model"] == "baseline" and freeze["test"]["rows"] == 9324
    for check in (ev._check_schema, ev._check_frozen_files, ev._check_data_prep,
                  ev._check_test_entry, ev._check_models, ev._check_inference):
        check(freeze, ev.ROOT)


def test_missing_freeze_refuses_final_run(repo, monkeypatch):
    root, _, _ = repo
    (root / "reports/freeze.json").unlink()
    refuse(root, ev.FreezeError, "T12 freeze not found", monkeypatch)
    with pytest.raises(ev.FreezeError):
        ev.check_only(root / "reports/freeze.json", root)


def test_freeze_differing_from_accepted_version_is_rejected(repo, monkeypatch):
    root, _, _ = repo
    (root / "reports/freeze.md").write_text("edited protocol\n", encoding="utf-8")
    refuse(root, ev.FreezeError, "accepted version: reports/freeze.md", monkeypatch)


def test_damaged_freeze_is_rejected(repo, monkeypatch):
    root, _, _ = repo
    (root / "reports/freeze.json").write_text('{"status": "approved"', encoding="utf-8")
    refuse(root, ev.FreezeError, "not valid JSON", monkeypatch)


@pytest.mark.parametrize("update,message", [
    (lambda f: f.update(status="draft"), "not approved"),
    (lambda f: f.update(schema="t12-freeze/v1"), "Unsupported freeze schema"),
    (lambda f: f.update(protocol_version="t12/2"), "protocol version"),
    (lambda f: f.update(pending_approval=["team"]), "pending approvals"),
    (lambda f: f.update(approved_by=[]), "no approvers"),
    (lambda f: f.pop("approved_at"), "approval date"),
    (lambda f: f.update(frozen_commit="abc"), "frozen_commit"),
    (lambda f: f.update(final_test_used_before_freeze=True), "final_test_used_before_freeze"),
    (lambda f: f.update(bot_model_selected_by="test"), "validation"),
    (lambda f: f["confirmed_decisions"].update(bot_model="rubert"), "confirmed bot model"),
    (lambda f: f.update(label_mapping={"negative": 0, "positive": 1, "neutral": 2}), "label mapping"),
    (lambda f: f["file_sha256"].pop("src/classifier.py"), "not pinned.*src/classifier.py"),
    (lambda f: f["models"].pop("rubert"), "exactly the models"),
    (lambda f: f["models"]["rubert"].update(report_sha256="0" * 64), "rubert run report differs"),
    (lambda f: f["models"]["baseline"].update(model_path="../outside"), "leaves the repository"),
    (lambda f: f["models"]["baseline"].update(model_sha256="0" * 64), "baseline run report differs.*bundle_sha256"),
    (lambda f: f["models"]["rubert"]["bundle_files_sha256"].update({"config.json": "0" * 64}), "bundle files differ"),
    (lambda f: f["data_prep"].update(split_ids_sha256="0" * 64), "split_ids_path differs"),
    (lambda f: f["data_prep"].update(train_ids_sha256="0" * 64), "baseline run report differs.*train_ids_sha256"),
    (lambda f: f["test"].update(rows=0), "rows must be positive"),
    (lambda f: f["test"].update(rows=7), "differs from T03 audit"),
    (lambda f: f["test"].update(ids_sha256="0" * 64), "split_ids.csv test IDs/labels differ"),
    (lambda f: f["test"].update(source_parquet_sha256="0" * 64), "source differs"),
    (lambda f: f["inference"].update(batch_size=0), "batch_size"),
    (lambda f: f["inference"].update(device="cuda"), "CPU"),
    (lambda f: f["inference"].update(num_threads=None), "num_threads"),
    (lambda f: f["text_protocol"].update(runs=2), "text protocol.*runs"),
    (lambda f: f["text_protocol"].update(max_text_length=2000), "text protocol.*max_text_length"),
    (lambda f: f["text_protocol"].update(labels=[0, 2, 1]), "text protocol.*labels"),
])
def test_invalid_freeze_refuses_before_models_and_test(repo, monkeypatch, update, message):
    root, _, _ = repo
    edit_freeze(root, update)
    refuse(root, ev.FreezeError, message, monkeypatch)


@pytest.mark.parametrize("rel", ["src/classifier.py", "src/train_baseline.py", "configs/transformer.json",
                                 "requirements.txt", "reports/data_prep/near_duplicate_candidates.csv"])
def test_changed_frozen_file_is_rejected(repo, monkeypatch, rel):
    root, _, _ = repo
    with (root / rel).open("a", encoding="utf-8") as fh:
        fh.write("\n# changed after the freeze\n")
    refuse(root, ev.FreezeError, f"Frozen files differ.*{rel}", monkeypatch)


def test_source_module_added_after_freeze_is_rejected(repo, monkeypatch):
    root, _, _ = repo
    write_bytes(root / "src/helpers.py", b"x = 1\n")
    refuse(root, ev.FreezeError, "not allowed: src/helpers.py", monkeypatch)


def test_changed_classifier_config_is_rejected_even_if_repinned(repo, monkeypatch):
    root, _, _ = repo
    edit_json(root, "configs/nlp_baseline.json", lambda c: c["text_processing"].update(truncation=512))
    refuse(root, ev.FreezeError, "Frozen files differ.*configs/nlp_baseline.json", monkeypatch)
    repin(root, "configs/nlp_baseline.json")
    refuse(root, ev.FreezeError, "baseline run report differs.*config_sha256", monkeypatch)


@pytest.mark.parametrize("rel,update,message", [
    (BASELINE_REPORT, lambda r: r.update(train_ids_sha256="9" * 64), "baseline run report differs.*train_ids_sha256"),
    (BASELINE_REPORT, lambda r: r.update(final_test_used=True), "baseline run report differs.*final_test_used"),
    (BASELINE_REPORT, lambda r: r["split_independence"].update(movie_id_overlap=1), "baseline run report shows"),
    (RUBERT_REPORT, lambda r: r["data_version"].update(validation_rows=3), "rubert data version differs.*validation_rows"),
    (RUBERT_REPORT, lambda r: r["data_version"]["checksums"].update(train_path="9" * 64), "rubert data version.*train_path"),
    (RUBERT_REPORT, lambda r: r.update(selection_metric="test_macro_f1"), "validation selection"),
    ("configs/transformer.json", lambda c: c.update(revision="d" * 40), "rubert configuration differs.*revision"),
    ("configs/transformer.json", lambda c: c.update(max_length=128), "rubert configuration differs.*max_length"),
])
def test_wrong_training_report_or_config_version_is_rejected(repo, monkeypatch, rel, update, message):
    root, _, _ = repo
    edit_json(root, rel, update)
    repin(root, rel)
    refuse(root, ev.FreezeError, message, monkeypatch)


@pytest.mark.parametrize("change,error,message", [
    (lambda root: write_bytes(root / "models/nlp_baseline/baseline.joblib", b"other baseline bundle"),
     ev.FreezeError, "Baseline bundle differs"),
    (lambda root: (root / "models/nlp_baseline/baseline.joblib").unlink(),
     ev.c.ModelUnavailableError, "baseline bundle not found"),
    (lambda root: write_bytes(root / "models/transformer/model.safetensors", b"other weights"),
     ev.FreezeError, "checksum differs from T12 freeze: model.safetensors"),
    (lambda root: write_bytes(root / "models/transformer/extra.bin", b"x"), ev.FreezeError, "bundle files differ"),
    (lambda root: shutil.rmtree(root / "models/transformer"), ev.c.ModelUnavailableError, "RuBERT bundle not found"),
])
def test_changed_model_bundle_is_rejected_before_loading(repo, monkeypatch, change, error, message):
    root, _, version = repo
    change(root)
    refuse(root, error, message, monkeypatch, version)


@pytest.mark.parametrize("kind,message", [("variable", "environment variables differ.*T13_SYNTHETIC_THREADS"),
                                          ("machine", "machine differs.*cpu"), ("package", "packages differ.*numpy")])
def test_other_environment_is_rejected(repo, monkeypatch, kind, message):
    root, _, _ = repo
    if kind == "variable":
        monkeypatch.setenv("T13_SYNTHETIC_THREADS", "4")
    elif kind == "machine":
        machine = {**ev.machine_info(), "cpu": "Other CPU"}
        monkeypatch.setattr(ev, "machine_info", lambda: machine)
    else:
        monkeypatch.setattr(ev, "installed_version", lambda name: "0.0")
    refuse(root, ev.FreezeError, message, monkeypatch)


@pytest.mark.parametrize("state,message", [
    ({**CLEAN_GIT, "commit": None}, "commit"),
    ({**CLEAN_GIT, "frozen_commit_is_ancestor": False}, "frozen commit"),
    ({**CLEAN_GIT, "tracked_changes": True}, "uncommitted"),
    ({**CLEAN_GIT, "evaluator_in_head": False, "evaluator_matches_head": False}, "not committed in Git HEAD"),
    ({**CLEAN_GIT, "evaluator_matches_head": False}, "differs from the version committed"),
])
def test_git_state_must_match_freeze(repo, monkeypatch, state, message):
    root, _, _ = repo
    monkeypatch.setattr(ev, "git_state", lambda root, frozen, **kwargs: state)
    refuse(root, ev.FreezeError, message, monkeypatch)


def test_incompatible_model_versions_refuse_before_test(repo, monkeypatch):
    root, _, version = repo
    forbid_test_access(monkeypatch)
    other = {**version, "validation_ids_sha256": "9" * 64}

    def factory(backend, model_path, *, report_path):
        return FakeModel(backend, version if backend == "baseline" else rubert_version(other))

    with pytest.raises(ev.DataMismatchError, match="validation_ids_sha256"):
        run(root, version, factory=factory)


def test_loaded_models_must_match_frozen_train_validation(repo, monkeypatch):
    """Обе модели согласованы между собой, но не с train/validation фиксации."""
    root, _, version = repo
    forbid_test_access(monkeypatch)
    with pytest.raises(ev.DataMismatchError, match="train/validation version differs from T12: train_rows"):
        run(root, version, factory=fake_factory({**version, "train_rows": 7}))


def test_changed_test_file_rejected_before_decoding(repo, monkeypatch):
    root, _, version = repo
    path = root / "data/processed/test.parquet"
    path.write_bytes(path.read_bytes() + b"\0")
    forbid = lambda *a, **k: pytest.fail("A changed test file must not be decoded")  # noqa: E731
    monkeypatch.setattr(tb, "load_prepared_split", forbid)
    with pytest.raises(ev.DataMismatchError, match="Test Parquet differs"):
        run(root, version)
    assert not (root / "out").exists()


@pytest.mark.parametrize("change", ["order", "label"])
def test_test_ids_order_and_labels_must_match(repo, monkeypatch, change):
    root, _, version = repo
    split_ids = root / "reports/data_prep/split_ids.csv"
    rows = list(csv.reader(split_ids.open(encoding="utf-8")))
    if change == "order":
        rows[2], rows[3] = rows[3], rows[2]
    else:
        rows[2][3] = "1"
    write_csv(split_ids, rows[0], rows[1:])
    sync_t03(root, version)
    # The freeze still pins the accepted ordered IDs/labels: refused before models and test.
    refuse(root, ev.FreezeError, "split_ids.csv test IDs/labels differ", monkeypatch, version)
    # Even a coherently changed freeze cannot make the prepared test match.
    frame = pd.read_csv(split_ids, dtype=str).query("split == 'test'")
    edit_freeze(root, lambda f: f["test"].update(ids_sha256=tb.ids_sha256(frame, id_column="record_id",
                                                                          label_column="label_id")))
    with pytest.raises(ev.DataMismatchError, match="Test IDs/labels differ"):
        run(root, version)
    assert not (root / "out").exists()


def test_test_class_counts_must_match_t03(repo):
    root, _, version = repo
    edit_json(root, "reports/data_prep/audit.json",
              lambda a: a["final_summary"]["test"].update(negative=1, neutral=2))
    sync_t03(root, version)
    with pytest.raises(ev.DataMismatchError, match="class counts"):
        run(root, version)


def test_cli_requires_explicit_mode_and_refuses_without_freeze(tmp_path, capsys):
    out = tmp_path / "out"
    with pytest.raises(SystemExit) as exited:
        ev.main(["--freeze", str(tmp_path / "freeze.json")])
    assert exited.value.code == 2
    assert ev.main(["--run-final", "--freeze", str(tmp_path / "freeze.json"), "--out-dir", str(out)]) == 2
    assert ev.main(["--check", "--freeze", str(tmp_path / "freeze.json")]) == 2
    assert "T12 freeze not found" in capsys.readouterr().err
    assert not out.exists()


def test_check_only_ready_without_reading_test(repo, monkeypatch):
    root, _, version = repo
    forbid_test_access(monkeypatch)
    result = ev.check_only(root / "reports/freeze.json", root, factory=fake_factory(version))
    assert result["status"] == "ready" and result["test_rows"] == 8
    assert result["final_test_decoded"] is False and result["final_inference_run"] is False
    assert [c["check"] for c in result["checks"]] == ["exclusive_run", "freeze_protocol", "frozen_files_and_reports",
                                                      "git", "environment", "output_dir",
                                                      "models", "test_file"]
    assert result["out_dir"] == "reports/evaluation" and not (root / "reports/evaluation").exists()
    assert result["freeze_sha256"] == sha(root, "reports/freeze.json")
    assert not [p for p in (root / "reports").iterdir() if p.name.startswith(".evaluation.")]


def test_check_only_lists_all_blockers(repo, monkeypatch):
    root, _, version = repo
    forbid_test_access(monkeypatch)
    monkeypatch.setenv("T13_SYNTHETIC_THREADS", "4")
    (root / "models/nlp_baseline/baseline.joblib").unlink()
    (root / "data/processed/test.parquet").unlink()
    result = ev.check_only(root / "reports/freeze.json", root, factory=forbid_models())
    status = {c["check"]: c["status"] for c in result["checks"]}
    assert result["status"] == "not_ready"
    assert status == {"freeze_protocol": "passed", "frozen_files_and_reports": "passed", "git": "passed",
                      "environment": "failed", "output_dir": "passed", "exclusive_run": "passed",
                      "models": "failed", "test_file": "failed"}


def test_check_only_stops_without_approved_protocol(repo, monkeypatch):
    root, _, _ = repo
    forbid_test_access(monkeypatch)
    edit_freeze(root, lambda f: f.update(status="draft"))
    result = ev.check_only(root / "reports/freeze.json", root, factory=forbid_models())
    assert result["status"] == "not_ready" and [c["check"] for c in result["checks"]] == ["exclusive_run", "freeze_protocol"]


# --------------------------------------------------------------------------- #
# Регрессии независимого ревью: фиксация, блокировка, git, запись, назначение
# --------------------------------------------------------------------------- #
def test_alternative_freeze_with_changed_parameters_is_rejected(repo, monkeypatch):
    """Сумма считается по выбранной фиксации, а не по неизменному reports/freeze.json."""
    root, _, _ = repo
    canonical = root / "reports/freeze.json"
    accepted = canonical.read_bytes()
    data = json.loads(accepted)
    data["inference"]["batch_size"] = 2
    alternative = root / "alt_freeze.json"
    tb.write_json(alternative, data)
    with monkeypatch.context() as scoped:
        forbid_test_access(scoped)
        with pytest.raises(ev.FreezeError, match="Selected T12 freeze differs from the accepted version"):
            ev.evaluate(alternative, root / "out", root=root, factory=forbid_models())
        result = ev.check_only(alternative, root, factory=forbid_models())
    assert result["status"] == "not_ready" and result["freeze_sha256"] == sha(root, "alt_freeze.json")
    assert [(c["check"], c["status"]) for c in result["checks"]] == [("exclusive_run", "passed"), ("freeze_protocol", "failed")]
    assert canonical.read_bytes() == accepted and not (root / "out").exists()


def test_identical_freeze_copy_is_reported_with_accepted_digest(repo):
    root, _, version = repo
    copy = root / "copy_freeze.json"
    copy.write_bytes((root / "reports/freeze.json").read_bytes())
    report = ev.evaluate(copy, root / "out", root=root, factory=fake_factory(version))
    assert report["run"]["freeze_sha256"] == ev.ACCEPTED_T12_SHA256["reports/freeze.json"]
    assert report["run"]["command"] == "python -m src.evaluate --run-final --freeze copy_freeze.json --out-dir out"


@pytest.mark.parametrize("destination", ["reports/evaluation", "trial 1", "-trial '2'", "../archive/result"])
def test_report_records_effective_arguments(repo, monkeypatch, destination):
    root, _, version = repo
    monkeypatch.chdir(root)
    out = Path(destination)
    out.parent.mkdir(exist_ok=True)
    external = destination.startswith("../")
    freeze = Path("../private protocol.json" if external else "reports/freeze.json")
    if external:
        shutil.copyfile(root / "reports/freeze.json", freeze)
    ev.evaluate(freeze, out, root=root, factory=fake_factory(version))
    saved = json.loads((out / "final_metrics.json").read_text())
    args = saved["run"]
    assert shlex.split(args["command"]) == args["argv"]
    assert args["argv"][:5] == ["python", "-m", "src.evaluate", "--run-final", "--freeze"]
    assert args["argv"][5:] == [args["freeze_path"], "--out-dir", args["out_dir"]]
    if external:
        assert args["required_local_paths"] == ["--freeze", "--out-dir"]
        assert (args["freeze_path"], args["out_dir"]) == ("<LOCAL_FREEZE_PATH>", "<LOCAL_OUT_DIR>")
        reports = "\n".join(p.read_text() for p in out.iterdir())
        assert all(private not in reports for private in (str(root.parent), "private protocol.json", "archive/result"))
    else:
        assert not args["required_local_paths"]
        assert args["freeze_path"] == "reports/freeze.json"
        assert args["out_dir"] == ("./" + destination if destination.startswith("-") else destination)
        assert (root / args["out_dir"]).samefile(out)


def hold_synthetic_run(mode, root, version, accepted, connection):
    """Дочерний процесс держит проверку на загрузке модели или запуск на предсказаниях."""
    ev.ACCEPTED_T12_SHA256 = accepted
    ev._set_torch_threads = lambda threads: None
    created = {}

    def pause():
        connection.send("using model")
        assert connection.poll(30) and connection.recv() == "continue"

    def factory(backend, model_path, *, report_path):
        if mode == "check" and backend == "baseline":
            pause()
        model = fake_factory(version, created)(backend, model_path, report_path=report_path)
        predict = model.predict_batch

        def predict_batch(texts, *, max_text_length):
            if mode == "final" and backend == "baseline" and len(model.calls) == 1:
                pause()
            return predict(texts, max_text_length=max_text_length)
        model.predict_batch = predict_batch
        return model

    if mode == "final":
        run(root, version, factory=factory)
        assert all(sum(model.calls[1:], []) == TEXTS for model in created.values())
    else:
        result = ev.check_only(root / "reports/freeze.json", root, factory=factory)
        assert result["status"] == "ready" and all(not model.calls for model in created.values())


@pytest.mark.parametrize("holder_mode", ["final", "check"])
def test_replacing_freeze_cannot_split_worktree_lock(repo, monkeypatch, holder_mode):
    root, _, version = repo
    freeze_path = root / "reports/freeze.json"
    git(root, "add", "reports/freeze.json")
    git(root, "commit", "-qm", "synthetic protocol")
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe()
    holder = ctx.Process(target=hold_synthetic_run,
                         args=(holder_mode, root, version, ev.ACCEPTED_T12_SHA256, child))
    holder.start()
    child.close()
    try:
        assert parent.poll(30) and parent.recv() == "using model"
        old_inode = freeze_path.stat().st_ino
        replacement = root / "identical.json"
        replacement.write_bytes(freeze_path.read_bytes())
        replacement.replace(freeze_path)
        assert freeze_path.stat().st_ino != old_inode
        assert sha(root, "reports/freeze.json") == ev.ACCEPTED_T12_SHA256["reports/freeze.json"]
        with monkeypatch.context() as scoped:
            forbid_test_access(scoped)
            scoped.setattr(ev, "read_freeze", lambda *args: pytest.fail("busy worktree: no protocol I/O"))
            scoped.setattr(ev, "load_models", lambda *a, **kw: pytest.fail("busy worktree: no model I/O"))
            with pytest.raises(ev.ConcurrentRunError, match="Another final evaluation"):
                ev.evaluate(freeze_path, root / "out2", root=root, factory=forbid_models())
            result = ev.check_only(freeze_path, root, out_dir=root / "out3", factory=forbid_models())
        assert result["status"] == "not_ready"
        assert [(c["check"], c["status"]) for c in result["checks"]] == [("exclusive_run", "failed")]
        assert "ConcurrentRunError" in result["checks"][0]["reason"]
    finally:
        if holder.is_alive():
            parent.send("continue")
        parent.close()
        holder.join(30)
        if holder.is_alive():
            holder.kill()
            holder.join()
    assert holder.exitcode == 0
    assert (root / "out/final_metrics.json").exists() == (holder_mode == "final")
    assert not (root / "out2").exists() and not (root / "out3").exists()
    assert not (root / "reports/evaluation").exists()
    assert ev.check_only(freeze_path, root, factory=fake_factory(version))["status"] == "ready"


@pytest.mark.parametrize("ending", ["normal", "killed"])
def test_lock_held_by_another_process_refuses_before_models(repo, monkeypatch, ending):
    root, _, version = repo
    holder = subprocess.Popen(
        [sys.executable, "-c", "from src.evaluate import final_run_lock; from pathlib import Path; import sys\n"
         "with final_run_lock(Path(sys.argv[1])):\n print('locked', flush=True); sys.stdin.read()", str(root)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    lock_path = Path(git(root, "rev-parse", "--absolute-git-dir")) / ev.LOCK_FILE
    try:
        assert holder.stdout.readline().strip() == "locked"
        inode = lock_path.stat().st_ino
        refuse(root, ev.ConcurrentRunError, "Another final evaluation", monkeypatch, version)
    finally:
        holder.kill() if ending == "killed" else holder.stdin.close()
        holder.wait(30)
    run(root, version)
    assert lock_path.stat().st_ino == inode and (root / "out/final_metrics.json").is_file()


def test_linked_worktree_lock_survives_exception(repo):
    root, _, _ = repo
    linked = root.parent / "linked tree"
    git(root, "worktree", "add", "--detach", str(linked))
    assert (linked / ".git").is_file()
    lock_path = Path(git(linked, "rev-parse", "--absolute-git-dir")) / ev.LOCK_FILE
    with ev.final_run_lock(root):
        with pytest.raises(ValueError, match="interrupted"):
            with ev.final_run_lock(linked):
                inode = lock_path.stat().st_ino
                with pytest.raises(ev.ConcurrentRunError):
                    with ev.final_run_lock(linked):
                        pytest.fail("linked worktree admitted a concurrent caller")
                raise ValueError("interrupted")
        with ev.final_run_lock(linked):
            assert lock_path.stat().st_ino == inode
    assert lock_path.is_file() and not git(linked, "status", "--porcelain")


def git(root, *args):
    return subprocess.run(["git", "-c", "user.name=T13", "-c", "user.email=t13@example.invalid",
                           "-c", "commit.gpgsign=false", *args], cwd=root, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.mark.parametrize("untracked", [False, True], ids=["missing", "untracked"])
def test_evaluator_missing_from_commit_is_rejected(repo, untracked):
    root, freeze, _ = repo
    frozen = freeze["frozen_commit"]
    git(root, "switch", "--detach", frozen)
    if untracked:
        shutil.copyfile(ev.__file__, root / "src/evaluate.py")
    state = ev.git_state(root, frozen)
    assert state["tracked_changes"] is False and state["evaluator_in_head"] is False
    with pytest.raises(ev.FreezeError, match="not committed in Git HEAD"):
        ev._check_git(freeze, root)


@pytest.mark.parametrize("change", ["worktree", "index", "index_only_differs_from_commit"])
def test_uncommitted_evaluator_rejected(repo, change):
    root, freeze, _ = repo
    path = root / "src/evaluate.py"
    original = path.read_bytes()
    path.write_bytes(original + b"# changed\n")
    if change != "worktree":
        git(root, "add", "src/evaluate.py")
    if change == "index_only_differs_from_commit":
        path.write_bytes(original)  # рабочие байты вновь совпадают с commit, индекс отличается
    with pytest.raises(ev.FreezeError, match="uncommitted"):
        ev._check_git(freeze, root)


def test_loaded_code_differs_from_clean_committed_file(repo, monkeypatch):
    root, freeze, _ = repo
    path = root / "src/evaluate.py"
    committed = path.read_bytes()
    path.write_bytes(committed + b"\ndef loaded_version(): return 'stale'\n")
    loaded = compile(path.read_bytes(), str(path), "exec", dont_inherit=True)
    path.write_bytes(committed)  # файл заменён после загрузки, до начала выполнения модуля
    monkeypatch.syspath_prepend(str(root))
    namespace = {"__file__": str(path), "__name__": "stale_evaluator"}
    exec(loaded, namespace)
    assert namespace["loaded_version"]() == "stale" and not git(root, "status", "--porcelain", "-uno")
    assert not namespace["git_state"](root, freeze["frozen_commit"])["evaluator_matches_head"]
    with pytest.raises(namespace["FreezeError"], match="differs from the version committed in Git HEAD"):
        namespace["_check_git"](freeze, root)


def test_committed_evaluator_is_accepted(repo):
    root, freeze, _ = repo
    commit = git(root, "rev-parse", "HEAD")
    assert ev._check_git(freeze, root) == commit
    assert ev._git_blob(root, f"{commit}:src/evaluate.py") == Path(ev.__file__).read_bytes()
    with pytest.raises(ev.FreezeError, match="frozen commit"):
        ev._check_git({"frozen_commit": "0" * 40}, root)


def test_head_moves_between_git_operations(repo, monkeypatch):
    root, freeze, _ = repo
    commit = git(root, "rev-parse", "HEAD")
    read_blob = ev._git_blob
    observed = {}

    def move_then_read(root, spec):
        git(root, "switch", "--detach", freeze["frozen_commit"])
        observed[spec] = read_blob(root, spec)
        return observed[spec]
    monkeypatch.setattr(ev, "_git_blob", move_then_read)
    with pytest.raises(ev.FreezeError, match="Git HEAD changed"):
        ev._check_git(freeze, root)
    assert observed == {f"{commit}:src/evaluate.py": Path(ev.__file__).read_bytes()}
    assert read_blob(root, "HEAD:src/evaluate.py") is None


def test_freeze_returns_only_the_verified_commit(repo, monkeypatch):
    root, freeze, _ = repo
    commit = git(root, "rev-parse", "HEAD")
    checked_state = ev.git_state

    def switch_after_verification(*args, **kwargs):
        state = checked_state(*args, **kwargs)
        git(root, "switch", "--detach", freeze["frozen_commit"])
        return state
    monkeypatch.setattr(ev, "git_state", switch_after_verification)
    _, _, verified = ev.load_freeze(root / "reports/freeze.json", root)
    assert verified == commit and ev._git_blob(root, "HEAD:src/evaluate.py") is None


@pytest.mark.parametrize("phase", ["predictions", "staging_complete", "staged_edit"])
def test_git_head_change_during_run_blocks_publication(repo, monkeypatch, phase):
    root, _, version = repo
    original = git(root, "rev-parse", "HEAD")

    def move_head():
        git(root, "commit", "--allow-empty", "-qm", "different commit, identical evaluator")

    if phase == "predictions":
        predict = FakeModel.predict_batch

        def move_during_prediction(model, texts, **kwargs):
            if model.backend == "rubert" and texts == TEXTS[-2:]:
                move_head()
            return predict(model, texts, **kwargs)
        monkeypatch.setattr(FakeModel, "predict_batch", move_during_prediction)
    else:
        write_json = tb.write_json

        def move_after_write(path, data):
            write_json(path, data)
            assert sorted(p.name for p in path.parent.iterdir()) == sorted(ev.OUTPUT_FILES)
            assert json.loads(path.read_text())["run"]["git_commit"] == original
            if phase == "staged_edit":
                with (root / "src/evaluate.py").open("ab") as evaluator:
                    evaluator.write(b"# changed during report generation\n")
                git(root, "add", "src/evaluate.py")
            else:
                move_head()
        monkeypatch.setattr(tb, "write_json", move_after_write)
    with pytest.raises(ev.FreezeError, match="uncommitted" if phase == "staged_edit" else "Git HEAD changed"):
        run(root, version)
    assert (git(root, "rev-parse", "HEAD") == original) == (phase == "staged_edit")
    assert not (root / "out").exists() and not list(root.glob(".out.*.tmp"))


def test_rename_noreplace_never_replaces_existing_directory(tmp_path):
    source, target = tmp_path / "staging", tmp_path / "final"
    write_bytes(source / "a.txt", b"new")
    target.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        ev._rename_noreplace(source, target)
    assert (source / "a.txt").read_bytes() == b"new" and list(target.iterdir()) == []
    target.rmdir()
    ev._rename_noreplace(source, target)
    assert (target / "a.txt").read_bytes() == b"new" and not source.exists()


@pytest.mark.parametrize("contents", [None, "other_run.json"])
def test_output_created_concurrently_is_not_replaced(repo, monkeypatch, contents):
    root, _, version = repo
    out = root / "out"
    publish = ev._rename_noreplace

    def racing(source, target):
        if target == out:  # another process creates the directory just before publication
            out.mkdir()
            if contents:
                (out / contents).write_text("{}", encoding="utf-8")
        return publish(source, target)
    monkeypatch.setattr(ev, "_rename_noreplace", racing)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run(root, version)
    assert sorted(p.name for p in out.iterdir()) == ([contents] if contents else [])
    assert not [p.name for p in root.iterdir() if p.name.startswith(".out.")]


class FakeLibc:
    """libc без операции переименования без замещения или с отказом файловой системы."""

    def __init__(self, code):
        self.code = code

    def __getattr__(self, name):
        if self.code is None:
            raise AttributeError(name)

        def function(*args):
            ev.ctypes.set_errno(self.code)
            return -1
        return function


@pytest.mark.parametrize("code,message", [(None, "no-replace rename is unavailable"),
                                          (ev.errno.EINVAL, "does not support atomic no-replace rename")])
def test_unsupported_no_replace_rename_refuses_before_inference(repo, monkeypatch, code, message):
    root, _, version = repo
    monkeypatch.setattr(ev.ctypes, "CDLL", lambda *args, **kwargs: FakeLibc(code))
    monkeypatch.setattr(ev.os, "rename", lambda *args: pytest.fail("no unsafe os.rename fallback"))
    monkeypatch.setattr(ev.os, "replace", lambda *args: pytest.fail("no unsafe os.replace fallback"))
    refuse(root, OSError, message, monkeypatch, version)
    result = ev.check_only(root / "reports/freeze.json", root, out_dir=root / "out", factory=fake_factory(version))
    output = next(c for c in result["checks"] if c["check"] == "output_dir")
    assert result["status"] == "not_ready" and output["status"] == "failed" and message in output["reason"]
    assert not [p.name for p in root.iterdir() if p.name.startswith(".out.")]


def test_existing_output_dir_fails_check_and_run_before_inference(repo, monkeypatch):
    root, _, version = repo
    (root / "out").mkdir()
    forbid_test_access(monkeypatch)
    result = ev.check_only(root / "reports/freeze.json", root, out_dir=root / "out", factory=fake_factory(version))
    output = next(c for c in result["checks"] if c["check"] == "output_dir")
    assert result["status"] == "not_ready" and "refusing to overwrite" in output["reason"]
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run(root, version, factory=forbid_models())
    assert list((root / "out").iterdir()) == []


def test_output_parent_that_is_a_file_fails_check_and_run(repo, monkeypatch):
    root, _, version = repo
    (root / "plain").write_text("not a directory", encoding="utf-8")
    out = root / "plain/evaluation"
    with monkeypatch.context() as scoped:
        forbid_test_access(scoped)
        result = ev.check_only(root / "reports/freeze.json", root, out_dir=out, factory=fake_factory(version))
        with pytest.raises(NotADirectoryError, match="Output parent"):
            ev.evaluate(root / "reports/freeze.json", out, root=root, factory=forbid_models())
    output = next(c for c in result["checks"] if c["check"] == "output_dir")
    assert result["status"] == "not_ready" and output["status"] == "failed" and "Output parent" in output["reason"]
    assert (root / "plain").read_text(encoding="utf-8") == "not a directory"


def test_missing_torch_gives_structured_not_ready(repo, monkeypatch):
    root, _, version = repo
    monkeypatch.setattr(ev, "_set_torch_threads", REAL_SET_TORCH_THREADS)
    monkeypatch.setitem(sys.modules, "torch", None)
    forbid_test_access(monkeypatch)
    result = ev.check_only(root / "reports/freeze.json", root, factory=fake_factory(version))
    models = next(c for c in result["checks"] if c["check"] == "models")
    assert result["status"] == "not_ready" and models["status"] == "failed"
    assert "torch is not installed" in models["reason"]
    with pytest.raises(ev.c.ModelUnavailableError, match="torch is not installed"):
        run(root, version)
    assert not (root / "out").exists()


class ConstantModel:
    def __init__(self, value):
        self.value = value

    def predict_batch(self, texts, *, max_text_length):
        return [{"label_id": self.value, "truncated": False} for _ in texts]


@pytest.mark.parametrize("value", [-1, 3, 100, 1.9, 2.0, True, "1"])
def test_invalid_model_class_is_rejected_before_return(value):
    message = "Unexpected class id" if type(value) is int else "integer class ids"
    with pytest.raises(ev.DataMismatchError, match=message):
        ev.run_model(ConstantModel(value), ["a", "b"], batch_size=2, load_seconds=0.0)
    for truth, predicted in (([value], [1]), ([1], [value])):
        with pytest.raises(ev.DataMismatchError, match=message):
            ev.compute_metrics(truth, predicted)


@pytest.mark.parametrize("value", [0, 1, 2])
def test_valid_model_class_is_preserved(value):
    result = ev.run_model(ConstantModel(value), ["a", "b"], batch_size=2, load_seconds=0.0)
    assert result["predictions"] == [value, value]
    assert ev.compute_metrics([value, value], result["predictions"])["confusion_matrix"][value][value] == 2


@pytest.mark.parametrize("link", ["file", "directory"])
def test_symlinked_prepared_test_is_rejected_in_check_and_run(repo, monkeypatch, link):
    """Ссылка внутри репозитория с тем же содержимым: отказ в обоих режимах до декодирования."""
    root, _, version = repo
    processed = root / "data/processed"
    if link == "file":
        (processed / "test.parquet").rename(root / "data/real_test.parquet")
        (processed / "test.parquet").symlink_to(root / "data/real_test.parquet")
    else:
        processed.rename(root / "data/real_processed")
        processed.symlink_to(root / "data/real_processed", target_is_directory=True)
    forbid = lambda *a, **k: pytest.fail("A symlinked test must not be decoded")  # noqa: E731
    monkeypatch.setattr(tb, "load_prepared_split", forbid)
    monkeypatch.setattr(pd, "read_parquet", forbid)
    result = ev.check_only(root / "reports/freeze.json", root, factory=fake_factory(version))
    test_file = next(c for c in result["checks"] if c["check"] == "test_file")
    assert result["status"] == "not_ready" and "symbolic links" in test_file["reason"]
    with pytest.raises(ev.DataMismatchError, match="symbolic links"):
        run(root, version)
    assert not (root / "out").exists()
