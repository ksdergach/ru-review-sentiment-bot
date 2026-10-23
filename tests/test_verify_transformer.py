"""Offline tests of the independent T06 verifier and bundle transfer tool.

Evidence is produced by a real ``run_training`` call on a tiny random BERT and
synthetic texts (only ``load_pretrained`` is replaced to avoid the network);
these are not RuBERT results.
"""
from pathlib import Path
import json
import shutil
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from src import train_baseline as base
from src import train_transformer as tt
import transformer_bundle
import verify_transformer as vt

from test_train_transformer import config, project, tiny_model  # noqa: F401  (fixtures)


@pytest.fixture
def completed_run(project, config, tiny_model, monkeypatch):  # noqa: F811
    root, _ = project
    model, tokenizer = tiny_model
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("models/\n", encoding="utf-8")
    (root / "src").mkdir()
    shutil.copy(ROOT / "src/train_transformer.py", root / "src/train_transformer.py")
    model.bert.save_pretrained(root / "pretrained")  # state before training
    config.update(epochs=2, batch_size=3, max_length=16, num_threads=1, eval_every_steps=2, log_every_steps=1)
    base.write_json(root / "configs/transformer.json", config)
    monkeypatch.setattr(tt, "load_pretrained", lambda c: (tokenizer, model, {"fixture_only": True}))
    tt.run_training(root / "configs/transformer.json", root)
    return root, config


def run(root, *flags):
    return vt.verify(root, args=vt.parse_args(["--root", str(root), *flags]))


def test_verifier_does_not_import_training_code():
    source = (ROOT / "tools/verify_transformer.py").read_text(encoding="utf-8")
    assert "from src" not in source and "import src" not in source and "train_transformer as" not in source


def test_verifier_passes_on_real_run_artifacts(completed_run):
    root, config = completed_run
    before = sorted(p.name for p in (root / config["report_dir"]).iterdir())
    result = run(root, "--full-validation", "--pretrained-check", "--pretrained", str(root / "pretrained"))
    assert result["status"] == "passed" and result["not_run"] == {}
    for name in ("parquet_checksum:train", "train_ordered_ids_equal_baseline", "bundle_checksum:model.safetensors",
                 "hf_config_class_order", "fixed_text_predictions_reproduced",
                 "full_validation_confusion_matrix_reproduced",
                 "winner_differs_from_pretrained_in_every_encoder_block"):
        assert name in result["checks"]
    assert result["fixed_texts"]["probability_labels"] == ["negative", "neutral", "positive"]
    # Plain verification never writes into the report directory.
    assert sorted(p.name for p in (root / config["report_dir"]).iterdir()) == before
    assert "verification.json" not in before


def test_pretrained_check_rejects_untrained_bundle(completed_run):
    root, config = completed_run
    with pytest.raises(vt.VerificationError, match="differs_from_pretrained"):
        run(root, "--pretrained-check", "--pretrained", str(root / config["model_dir"]))


@pytest.mark.parametrize("tamper", ["weights", "validation_csv", "class_order", "script", "extra_file", "parquet"])
def test_verifier_detects_tampering(completed_run, tamper):
    root, config = completed_run
    bundle, report = root / config["model_dir"], root / config["report_dir"]
    if tamper == "weights":
        with (bundle / "model.safetensors").open("ab") as fh:
            fh.write(b"\0")
    elif tamper == "validation_csv":
        text = (report / "validation_results.csv").read_text(encoding="utf-8").splitlines()
        text[1] = text[1].replace(",2,", ",3,", 1)
        (report / "validation_results.csv").write_text("\n".join(text) + "\n", encoding="utf-8")
    elif tamper == "class_order":
        hf = base.read_json(bundle / "config.json")
        hf["id2label"] = {"0": "positive", "1": "neutral", "2": "negative"}
        base.write_json(bundle / "config.json", hf)
    elif tamper == "script":
        with (root / "src/train_transformer.py").open("a", encoding="utf-8") as fh:
            fh.write("\n# changed\n")
    elif tamper == "extra_file":
        (bundle / "stray.bin").write_bytes(b"x")
    else:
        with (root / config["train_path"]).open("ab") as fh:
            fh.write(b"changed")
    with pytest.raises(vt.VerificationError):
        run(root)


def test_missing_bundle_is_an_error_not_a_pass(completed_run, capsys):
    root, config = completed_run
    shutil.rmtree(root / config["model_dir"])
    assert vt.main(["--root", str(root)]) == 2
    assert "Bundle not found" in capsys.readouterr().err
    assert run(root, "--no-bundle")["status"] == "passed_without_bundle"


def test_cli_refuses_to_overwrite_committed_report(completed_run):
    root, _ = completed_run
    assert vt.main(["--root", str(root), "--output", str(root / vt.COMMITTED_REPORT)]) == 2
    assert not (root / vt.COMMITTED_REPORT).exists()


def test_bundle_pack_restore_roundtrip(completed_run, tmp_path_factory):
    root, config = completed_run
    bundle = root / config["model_dir"]
    storage = tmp_path_factory.mktemp("storage")
    archive = storage / "t06.tar"
    assert transformer_bundle.main(["pack", "--root", str(root), "--out", str(archive)]) == 0
    assert (storage / "t06.tar.sha256").read_text().split()[0] == base.sha256_file(archive)
    original = {p.name: base.sha256_file(p) for p in bundle.iterdir()}
    shutil.rmtree(bundle)
    assert transformer_bundle.main(["restore", "--root", str(root), "--archive", str(archive)]) == 0
    assert {p.name: base.sha256_file(p) for p in bundle.iterdir()} == original
    assert run(root)["status"] == "passed"
    # Restore never overwrites an existing bundle; no staging directories are left behind.
    assert transformer_bundle.main(["restore", "--root", str(root), "--archive", str(archive)]) == 1
    assert sorted(p.name for p in bundle.parent.iterdir()) == [bundle.name]


def test_bundle_restore_rejects_corrupt_archive(completed_run, tmp_path_factory):
    root, config = completed_run
    bundle = root / config["model_dir"]
    archive = tmp_path_factory.mktemp("storage") / "t06.tar"
    assert transformer_bundle.main(["pack", "--root", str(root), "--out", str(archive)]) == 0
    shutil.rmtree(bundle)
    with archive.open("r+b") as fh:
        fh.seek(2048)
        fh.write(b"corrupt")
    assert transformer_bundle.main(["restore", "--root", str(root), "--archive", str(archive)]) == 1
    assert not bundle.exists() or not any(bundle.iterdir())


def test_bundle_pack_refuses_bundle_not_matching_run(completed_run, tmp_path_factory):
    root, config = completed_run
    meta = base.read_json(root / config["model_dir"] / "label_mapping.json")
    base.write_json(root / config["model_dir"] / "label_mapping.json", {**meta, "x": 3})
    archive = tmp_path_factory.mktemp("storage") / "t06.tar"
    assert transformer_bundle.main(["pack", "--root", str(root), "--out", str(archive)]) == 1
    assert not archive.exists()


def _rewrite_archive(archive, extra):
    """Copy a packed archive, append members via ``extra(tar)`` and refresh the sidecar digest."""
    import tarfile
    source = archive.read_bytes()
    with tarfile.open(archive, "w", format=tarfile.PAX_FORMAT) as out, \
            tarfile.open(fileobj=__import__("io").BytesIO(source)) as src:
        for member in src.getmembers():
            out.addfile(member, src.extractfile(member))
        extra(out)
    archive.with_name(archive.name + ".sha256").write_text(
        f"{base.sha256_file(archive)}  {archive.name}\n", encoding="utf-8")


@pytest.mark.parametrize("attack", ["duplicate", "symlink", "traversal", "unexpected"])
def test_bundle_restore_rejects_malicious_members(completed_run, tmp_path_factory, attack):
    import io
    import tarfile
    root, config = completed_run
    bundle = root / config["model_dir"]
    archive = tmp_path_factory.mktemp("storage") / "t06.tar"
    assert transformer_bundle.main(["pack", "--root", str(root), "--out", str(archive)]) == 0
    shutil.rmtree(bundle)

    def extra(tar):
        if attack == "symlink":
            info = tarfile.TarInfo("transformer/link")
            info.type, info.linkname = tarfile.SYMTYPE, "/etc/passwd"
            tar.addfile(info)
            return
        name = {"duplicate": "transformer/label_mapping.json", "traversal": "transformer/../escape",
                "unexpected": "transformer/stray.bin"}[attack]
        info = tarfile.TarInfo(name)
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))

    _rewrite_archive(archive, extra)
    assert transformer_bundle.main(["restore", "--root", str(root), "--archive", str(archive)]) == 1
    assert not bundle.exists()
    assert not (root / "models" / "escape").exists() and not (root / "escape").exists()
    # The staging directory is removed even when restore fails part-way.
    assert not list((root / "models").glob(".restore-*"))


def test_bundle_pack_rejects_symlinked_file(completed_run, tmp_path_factory):
    root, config = completed_run
    bundle = root / config["model_dir"]
    real = tmp_path_factory.mktemp("elsewhere") / "label_mapping.json"
    shutil.move(bundle / "label_mapping.json", real)
    (bundle / "label_mapping.json").symlink_to(real)
    archive = tmp_path_factory.mktemp("storage") / "t06.tar"
    assert transformer_bundle.main(["pack", "--root", str(root), "--out", str(archive)]) == 1
    assert not archive.exists()
    with pytest.raises(vt.VerificationError, match="bundle_file_set"):
        run(root)


def test_verifier_rejects_stray_subdirectory_in_bundle(completed_run):
    root, config = completed_run
    (root / config["model_dir"] / "checkpoint-old").mkdir()
    with pytest.raises(vt.VerificationError, match="bundle_file_set"):
        run(root)


@pytest.mark.parametrize("tamper", [None, "weights"])
def test_bundle_reconstruct_requires_bitwise_identical_repeat(completed_run, tmp_path_factory, tamper):
    root, config = completed_run
    bundle = root / config["model_dir"]
    # A repeat run differs only in its own training_config.json and metadata.json.
    source = tmp_path_factory.mktemp("repeat") / "transformer_repeat"
    shutil.copytree(bundle, source)
    base.write_json(source / "training_config.json", {**config, "model_dir": "models/transformer_repeat"})
    base.write_json(source / "metadata.json", {"elapsed_seconds": -1})
    if tamper == "weights":
        with (source / "model.safetensors").open("ab") as fh:
            fh.write(b"\0")
    original = {p.name: base.sha256_file(p) for p in bundle.iterdir()}
    shutil.rmtree(bundle)
    code = transformer_bundle.main(["reconstruct", "--root", str(root), "--source", str(source)])
    if tamper:
        assert code == 1 and not bundle.exists()
    else:
        assert code == 0
        assert {p.name: base.sha256_file(p) for p in bundle.iterdir()} == original
        assert run(root)["status"] == "passed"
    assert not list((root / "models").glob(".reconstruct-*"))
