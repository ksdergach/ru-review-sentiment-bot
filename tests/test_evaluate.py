"""T13: финальное сравнение на искусственном репозитории; настоящий test не читается.

Ожидаемые метрики посчитаны вручную, независимо от sklearn и src.train_baseline.
"""
import csv
import json
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


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """Искусственный репозиторий с данными T03, отчётами моделей и утверждённой фиксацией."""
    pytest.importorskip("pyarrow")
    root = tmp_path / "repo"
    (root / "configs").mkdir(parents=True)
    tb.write_json(root / "configs/nlp_baseline.json", {"data_prep_audit_path": "reports/data_prep/audit.json",
                                                       "data_prep_split_ids_path": "reports/data_prep/split_ids.csv"})
    tb.write_json(root / "configs/data_prep.json", {"report_dir": "reports/data_prep", "near_duplicates": {
        "decision_file": "configs/near_duplicate_decisions.csv"}})
    prep = root / "reports/data_prep"
    tb.write_json(prep / "audit.json", {"status": "completed", "final_summary": {"test": {"rows": len(TEST_ROWS)}},
                                        "exact_dedup": {"test_exact_duplicate_groups": 1},
                                        "near_duplicates": {"near_pairs_within_test": 1}})
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
    for backend in ev.BACKENDS:
        tb.write_json(root / f"reports/{backend}/run_metadata.json", {"backend": backend})
    freeze = {
        "schema": ev.FREEZE_SCHEMA, "status": "approved", "approved_by": ["team"], "approved_at": "2026-12-01",
        "frozen_commit": "a" * 40, "final_test_used_before_freeze": False,
        "bot_model": "baseline", "bot_model_selected_by": "validation_and_resources",
        "data_prep": {"audit_sha256": tb.sha256_file(prep / "audit.json"),
                      "split_ids_sha256": tb.sha256_file(prep / "split_ids.csv")},
        "test": {"path": "data/processed/test.parquet", "parquet_sha256": tb.sha256_file(test_path),
                 "rows": len(TEST_ROWS), "ids_sha256": tb.ids_sha256(frame, id_column="record_id", label_column="label_id")},
        "models": {b: {"model_path": f"models/{b}", "report_path": f"reports/{b}/run_metadata.json",
                       "report_sha256": tb.sha256_file(root / f"reports/{b}/run_metadata.json")} for b in ev.BACKENDS},
        "inference": {"batch_size": 3, "device": "cpu", "num_threads": None},
    }
    tb.write_json(root / "reports/freeze.json", freeze)
    monkeypatch.setattr(ev, "git_state", lambda root, frozen: {
        "commit": "b" * 40, "frozen_commit_is_ancestor": True, "tracked_changes": False})
    version = {**VERSION, "t03_split_ids_sha256": freeze["data_prep"]["split_ids_sha256"],
               "t03_audit_sha256": freeze["data_prep"]["audit_sha256"]}
    return root, freeze, version


def edit_freeze(root, update):
    path = root / "reports/freeze.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    update(data)
    tb.write_json(path, data)


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
        "confusion_matrix_baseline.csv", "confusion_matrix_rubert.csv"])
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
    assert report["run"]["command"] == "python -m src.evaluate --freeze reports/freeze.json"
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
    assert json.loads((out / "final_metrics.json").read_text(encoding="utf-8"))["models"]["rubert"]["metrics"] == rubert


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
    edit_freeze(root, lambda f: f["data_prep"].update(audit_sha256=tb.sha256_file(audit)))
    version["t03_audit_sha256"] = tb.sha256_file(audit)
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
    assert {"python", "platform", "logical_cpu_count"} <= set(report["environment"])


def test_shared_classifier_interface_with_real_baseline_bundle(repo):
    """Baseline проходит через настоящий src.classifier.BaselineClassifier."""
    root, freeze, version = repo
    (root / "models").mkdir()
    model_path = make_real_bundle(root / "models", training_update=lambda t: t.update(
        {k: v for k, v in version.items() if k != "final_test_used"}))
    rel = lambda p: p.relative_to(root).as_posix()  # noqa: E731
    edit_freeze(root, lambda f: f["models"]["baseline"].update(
        model_path=rel(model_path), report_path=rel(report_of(model_path)),
        report_sha256=tb.sha256_file(report_of(model_path))))
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


# --------------------------------------------------------------------------- #
# Защита финального test
# --------------------------------------------------------------------------- #
def forbid_test_access(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("The final test must not be read before all preconditions pass")
    monkeypatch.setattr(tb, "load_prepared_split", forbidden)
    monkeypatch.setattr(ev, "load_frozen_test", forbidden)


def forbid_models():
    def factory(*args, **kwargs):
        pytest.fail("Models must not be loaded without an approved T12 freeze")
    return factory


def test_missing_freeze_refuses_final_run(repo, monkeypatch):
    root, _, _ = repo
    (root / "reports/freeze.json").unlink()
    forbid_test_access(monkeypatch)
    with pytest.raises(ev.FreezeError, match="T12 freeze not found"):
        run(root, None, factory=forbid_models())
    with pytest.raises(ev.FreezeError):
        ev.check_only(root / "reports/freeze.json", root)
    assert not (root / "out").exists()


@pytest.mark.parametrize("update,message", [
    (lambda f: f.update(status="draft"), "not approved"),
    (lambda f: f.update(schema="t12-freeze/v1"), "Unsupported freeze schema"),
    (lambda f: f.update(approved_by=[]), "no approvers"),
    (lambda f: f.pop("approved_at"), "approval date"),
    (lambda f: f.update(frozen_commit="abc"), "frozen_commit"),
    (lambda f: f.update(final_test_used_before_freeze=True), "final_test_used_before_freeze"),
    (lambda f: f.update(bot_model_selected_by="test"), "validation"),
    (lambda f: f["models"].pop("rubert"), "exactly the models"),
    (lambda f: f["models"]["rubert"].update(report_sha256="0" * 64), "rubert run report differs"),
    (lambda f: f["models"]["baseline"].update(model_path="../outside"), "leaves the repository"),
    (lambda f: f["data_prep"].update(split_ids_sha256="0" * 64), "split_ids_path differs"),
    (lambda f: f["test"].update(rows=0), "rows must be positive"),
    (lambda f: f["inference"].update(batch_size=0), "batch_size"),
    (lambda f: f["inference"].update(device="cuda"), "CPU"),
])
def test_invalid_freeze_refuses_before_models_and_test(repo, monkeypatch, update, message):
    root, _, _ = repo
    edit_freeze(root, update)
    forbid_test_access(monkeypatch)
    with pytest.raises(ev.FreezeError, match=message):
        run(root, None, factory=forbid_models())
    assert not (root / "out").exists()


@pytest.mark.parametrize("state,message", [
    ({"commit": None, "frozen_commit_is_ancestor": True, "tracked_changes": False}, "commit"),
    ({"commit": "b" * 40, "frozen_commit_is_ancestor": False, "tracked_changes": False}, "frozen commit"),
    ({"commit": "b" * 40, "frozen_commit_is_ancestor": True, "tracked_changes": True}, "uncommitted"),
])
def test_git_state_must_match_freeze(repo, monkeypatch, state, message):
    root, _, _ = repo
    monkeypatch.setattr(ev, "git_state", lambda root, frozen: state)
    forbid_test_access(monkeypatch)
    with pytest.raises(ev.FreezeError, match=message):
        run(root, None, factory=forbid_models())


def test_incompatible_model_versions_refuse_before_test(repo, monkeypatch):
    root, _, version = repo
    forbid_test_access(monkeypatch)
    other = {**version, "validation_ids_sha256": "9" * 64}

    def factory(backend, model_path, *, report_path):
        return FakeModel(backend, version if backend == "baseline" else rubert_version(other))

    with pytest.raises(ev.DataMismatchError, match="validation_ids_sha256"):
        run(root, version, factory=factory)


@pytest.mark.parametrize("update,message", [
    (lambda f: f["test"].update(parquet_sha256="0" * 64), "Test Parquet differs"),
    (lambda f: f["test"].update(ids_sha256="0" * 64), "IDs/labels differ"),
    (lambda f: f["test"].update(rows=7), "T12 freeze records 7"),
])
def test_test_data_must_match_freeze(repo, update, message):
    root, _, version = repo
    edit_freeze(root, update)
    with pytest.raises(ev.DataMismatchError, match=message):
        run(root, version)
    assert not (root / "out").exists()


def test_changed_test_file_rejected_before_decoding(repo, monkeypatch):
    root, _, version = repo
    path = root / "data/processed/test.parquet"
    path.write_bytes(path.read_bytes() + b"\0")
    forbid = lambda *a, **k: pytest.fail("A changed test file must not be decoded")  # noqa: E731
    monkeypatch.setattr(tb, "load_prepared_split", forbid)
    with pytest.raises(ev.DataMismatchError, match="Test Parquet differs"):
        run(root, version)


@pytest.mark.parametrize("change", ["order", "label"])
def test_test_ids_must_match_t03_split_ids(repo, change):
    root, _, version = repo
    split_ids = root / "reports/data_prep/split_ids.csv"
    rows = list(csv.reader(split_ids.open(encoding="utf-8")))
    if change == "order":
        rows[2], rows[3] = rows[3], rows[2]
    else:
        rows[2][3] = "1"
    write_csv(split_ids, rows[0], rows[1:])
    edit_freeze(root, lambda f: f["data_prep"].update(split_ids_sha256=tb.sha256_file(split_ids)))
    version["t03_split_ids_sha256"] = tb.sha256_file(split_ids)
    with pytest.raises(ev.DataMismatchError, match="split_ids.csv"):
        run(root, version)


def test_cli_refuses_without_freeze_and_writes_nothing(tmp_path, capsys):
    out = tmp_path / "out"
    assert ev.main(["--freeze", str(tmp_path / "freeze.json"), "--out-dir", str(out)]) == 2
    assert ev.main(["--check", "--freeze", str(tmp_path / "freeze.json")]) == 2
    assert "T12 freeze not found" in capsys.readouterr().err
    assert not out.exists()


def test_check_only_reports_readiness_without_reading_test(repo, monkeypatch):
    root, _, _ = repo
    forbid_test_access(monkeypatch)
    result = ev.check_only(root / "reports/freeze.json", root)
    assert result["status"] == "ready" and result["final_test_used"] is False and result["test_rows"] == 8

