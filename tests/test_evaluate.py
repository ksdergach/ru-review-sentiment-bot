"""T13: финальное сравнение на искусственном репозитории; настоящий test не читается.

Ожидаемые метрики посчитаны вручную, независимо от sklearn и src.train_baseline.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import evaluate as ev, train_baseline as tb  # noqa: E402
from test_classifier import make_real_bundle, report_of  # noqa: E402

NAMES = {0: "negative", 1: "neutral", 2: "positive"}
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
    write_bytes(root / "src/evaluate.py", b"evaluator added after the freeze\n")
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
        "confirmed_decisions": {"bot_model": "baseline"}, "frozen_commit": "a" * 40,
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
    monkeypatch.setattr(ev, "git_state", lambda root, frozen: {
        "commit": "b" * 40, "frozen_commit_is_ancestor": True, "tracked_changes": False})
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


@pytest.mark.parametrize("truth,predicted", [([0, 1], [0]), ([], []), ([0, 3], [0, 1]), ([0, 1], [0, -1])])
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
    assert report["run"]["command"] == "python -m src.evaluate --run-final --freeze reports/freeze.json"
    assert report["run"]["protocol_version"] == "t12/1" and report["run"]["frozen_commit"] == "a" * 40
    # RuBERT is better on this synthetic test, but the bot model chosen on validation stays.
    assert rubert_better(report) and report["bot_model"] == "baseline"
    assert report["bot_model_changed_by_test"] is False
    assert baseline["errors"] == {"total": 4, "by_true_class": {"negative": 1, "neutral": 1, "positive": 2},
                                  "by_predicted_class": {"negative": 3, "neutral": 0, "positive": 1}}
    assert report["run"]["git_commit"] == "b" * 40
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
    freeze = ev.read_freeze(ev.DEFAULT_FREEZE_PATH)
    assert tb.sha256_file(ev.DEFAULT_FREEZE_PATH) == ev.ACCEPTED_T12_SHA256["reports/freeze.json"]
    assert freeze["frozen_commit"] == "5f22d93a8cff63b30fc129bb68a1cfbd4230995d"
    assert freeze["inference"] == {"batch_size": 64, "device": "cpu", "num_threads": 2}
    assert freeze["bot_model"] == "baseline" and freeze["test"]["rows"] == 9324
    for check in (ev._check_schema, ev._check_accepted_t12, ev._check_frozen_files, ev._check_data_prep,
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
    ({"commit": None, "frozen_commit_is_ancestor": True, "tracked_changes": False}, "commit"),
    ({"commit": "b" * 40, "frozen_commit_is_ancestor": False, "tracked_changes": False}, "frozen commit"),
    ({"commit": "b" * 40, "frozen_commit_is_ancestor": True, "tracked_changes": True}, "uncommitted"),
])
def test_git_state_must_match_freeze(repo, monkeypatch, state, message):
    root, _, _ = repo
    monkeypatch.setattr(ev, "git_state", lambda root, frozen: state)
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
    assert [c["check"] for c in result["checks"]] == ["freeze_protocol", "frozen_files_and_reports", "git",
                                                      "environment", "models", "test_file"]


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
                      "environment": "failed", "models": "failed", "test_file": "failed"}


def test_check_only_stops_without_approved_protocol(repo, monkeypatch):
    root, _, _ = repo
    forbid_test_access(monkeypatch)
    edit_freeze(root, lambda f: f.update(status="draft"))
    result = ev.check_only(root / "reports/freeze.json", root, factory=forbid_models())
    assert result["status"] == "not_ready" and [c["check"] for c in result["checks"]] == ["freeze_protocol"]
