"""Portable tests of the T06 model download/installation: stdlib + pytest, no torch, no Internet.

A tiny bundle is packed with tools/transformer_bundle.py and served by a local HTTP
server that imitates Google Drive (confirmation page, quota page, broken connections,
wrong content). These tests run on Linux, Windows and macOS in CI.
"""
from __future__ import annotations

import hashlib
import http.server
import io
import json
import shutil
import sys
import tarfile
import threading
import urllib.parse
from collections import namedtuple
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import fetch_transformer as ft  # noqa: E402
import transformer_bundle as tb  # noqa: E402

BUNDLE_FILES = {"config.json": b'{"a": 1}\n', "model.safetensors": bytes(range(256)) * 300,
                "tokenizer.json": "{\"токен\": 1}\n".encode("utf-8")}
CONFIRM_PAGE = """<!DOCTYPE html><html><body><p>Google Drive can't scan this file for viruses.</p>
<form id="download-form" action="/download" method="get"><input type="submit" value="Download anyway"/>
<input type="hidden" name="id" value="X"><input type="hidden" name="export" value="download">
<input type="hidden" name="confirm" value="t"><input type="hidden" name="uuid" value="u-1"></form></body></html>"""
QUOTA_PAGE = "<html><body>Too many users have viewed or downloaded this file recently.</body></html>"

Project = namedtuple("Project", "root bundle archive_bytes sha256 cache")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def _make_project(root: Path, archive_bytes: bytes | None = None) -> Project:
    """Minimal project layout with the same relative paths as the repository."""
    expected = {name: _sha(data) for name, data in BUNDLE_FILES.items()}
    _write_json(root / "reports/transformer/run_metadata.json", {"bundle_files_sha256": expected})
    _write_json(root / "configs/transformer.json", {"model_dir": "models/transformer",
                                                    "report_dir": "reports/transformer"})
    if archive_bytes is None:
        source = root / "source_bundle"
        source.mkdir()
        for name, data in BUNDLE_FILES.items():
            (source / name).write_bytes(data)
        packed = root / "packed" / "bundle.tar"
        tb.pack(source, packed, expected)
        archive_bytes = packed.read_bytes()
        shutil.rmtree(root / "packed")
        shutil.rmtree(source)
    digest = _sha(archive_bytes)
    _write_json(root / ft.SOURCE_CONFIG, {
        "archive_name": "t06_rubert_bundle.tar", "sha256": digest, "size_bytes": len(archive_bytes),
        "google_drive_file_id": "X", "google_drive_view_url": "https://drive.google.com/file/d/X/view",
        "download_dir": "artifacts/downloads"})
    return Project(root, root / "models/transformer", archive_bytes, digest,
                   root / "artifacts/downloads/t06_rubert_bundle.tar")


@pytest.fixture
def project(tmp_path):
    # Spaces and Cyrillic in the path: typical for Windows/macOS home folders.
    return _make_project(tmp_path / "мой проект")


class FakeDrive:
    """Local HTTP server; ``script`` is a list of behaviours, one per request (last one repeats)."""

    def __init__(self, payload: bytes, script):
        self.payload, self.script, self.requests = payload, list(script), []
        drive = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                drive.requests.append({"path": self.path, "range": self.headers.get("Range")})
                action = drive.script[min(len(drive.requests), len(drive.script)) - 1]
                getattr(drive, f"_{action}")(self)

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/download?id=X&export=download"

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    @staticmethod
    def _send(handler, status, body: bytes, ctype="application/octet-stream", headers=(), length=None):
        handler.send_response(status)
        handler.send_header("Content-Type", ctype)
        handler.send_header("Content-Length", str(len(body) if length is None else length))
        for key, value in headers:
            handler.send_header(key, value)
        handler.end_headers()
        handler.wfile.write(body)

    def _file(self, handler):
        start = int(handler.headers["Range"][6:-1]) if handler.headers.get("Range") else 0
        if start:
            self._send(handler, 206, self.payload[start:], headers=[
                ("Content-Range", f"bytes {start}-{len(self.payload) - 1}/{len(self.payload)}")])
        else:
            self._send(handler, 200, self.payload)

    def _no_range(self, handler):
        self._send(handler, 200, self.payload)

    def _confirm(self, handler):
        query = urllib.parse.parse_qs(urllib.parse.urlparse(handler.path).query)
        if query.get("confirm") == ["t"]:
            return self._file(handler)
        self._send(handler, 200, CONFIRM_PAGE.encode(), "text/html; charset=utf-8")

    def _quota(self, handler):
        self._send(handler, 200, QUOTA_PAGE.encode(), "text/html; charset=utf-8")

    def _truncated(self, handler):
        # Promise the whole file, send a third and drop the connection.
        self._send(handler, 200, self.payload[: len(self.payload) // 3], length=len(self.payload))
        handler.close_connection = True

    def _corrupt(self, handler):
        self._send(handler, 200, bytes(len(self.payload)))

    def _oversized(self, handler):
        self._send(handler, 200, self.payload + b"x")

    def _unavailable(self, handler):
        self._send(handler, 503, b"busy", "text/plain")

    def _missing(self, handler):
        self._send(handler, 404, b"not found", "text/plain")


@pytest.fixture
def drive_factory():
    servers = []

    def make(payload, *script):
        servers.append(FakeDrive(payload, script))
        return servers[-1]

    yield make
    for server in servers:
        server.close()


def _install(project, logs=None, sleeps=None, **kwargs):
    return ft.install(project.root, sleep=(sleeps.append if sleeps is not None else lambda s: None),
                      log=(logs.append if logs is not None else lambda m: None), timeout=10, **kwargs)


def _assert_installed(project):
    assert sorted(p.name for p in project.bundle.iterdir()) == sorted(BUNDLE_FILES)
    for name, data in BUNDLE_FILES.items():
        assert (project.bundle / name).read_bytes() == data


def _assert_not_installed(project):
    assert not project.bundle.exists()
    leftovers = [p for p in (project.root / "models").glob("*")] if (project.root / "models").exists() else []
    assert leftovers == []  # no staging directories are left behind


def _no_network(*args, **kwargs):
    raise AssertionError("network must not be used")


# --------------------------------------------------------------------------- #
def test_download_verify_restore_and_idempotent_rerun(project, drive_factory):
    drive = drive_factory(project.archive_bytes, "file")
    logs = []
    assert _install(project, logs, url=drive.url) == "installed"
    _assert_installed(project)
    assert project.cache.read_bytes() == project.archive_bytes
    assert not project.cache.with_name(project.cache.name + ".partial").exists()
    assert "Модель готова" in logs[-1]

    mtimes = {p.name: p.stat().st_mtime_ns for p in project.bundle.iterdir()}
    assert _install(project, url=drive.url, opener=_no_network) == "present"
    assert len(drive.requests) == 1
    assert {p.name: p.stat().st_mtime_ns for p in project.bundle.iterdir()} == mtimes


def test_cached_archive_is_reused_without_network(project, drive_factory):
    drive = drive_factory(project.archive_bytes, "file")
    _install(project, url=drive.url)
    shutil.rmtree(project.bundle)
    assert _install(project, url=drive.url, opener=_no_network) == "installed"
    _assert_installed(project)


def test_corrupted_cached_archive_is_downloaded_again(project, drive_factory):
    project.cache.parent.mkdir(parents=True)
    project.cache.write_bytes(b"garbage")
    drive = drive_factory(project.archive_bytes, "file")
    assert _install(project, url=drive.url) == "installed"
    assert project.cache.read_bytes() == project.archive_bytes


def test_google_drive_confirmation_page_is_followed(project, drive_factory):
    drive = drive_factory(project.archive_bytes, "confirm")
    assert _install(project, url=drive.url) == "installed"
    _assert_installed(project)
    assert len(drive.requests) == 2 and "confirm=t" in drive.requests[1]["path"]
    assert "uuid=u-1" in drive.requests[1]["path"]


def test_confirmation_url_parser():
    url = ft.confirmation_url(CONFIRM_PAGE, "https://drive.usercontent.google.com/download?id=X")
    parsed = urllib.parse.urlparse(url)
    assert (parsed.scheme, parsed.netloc, parsed.path) == ("https", "drive.usercontent.google.com", "/download")
    assert urllib.parse.parse_qs(parsed.query) == {"id": ["X"], "export": ["download"], "confirm": ["t"],
                                                    "uuid": ["u-1"]}
    assert ft.confirmation_url(QUOTA_PAGE, "https://example.org/") is None
    assert ft.drive_url("ABC").startswith("https://drive.usercontent.google.com/download?id=ABC&")


def test_html_instead_of_archive_is_reported_and_nothing_installed(project, drive_factory):
    drive = drive_factory(project.archive_bytes, "quota")
    with pytest.raises(ft.FetchError, match="HTML"):
        _install(project, url=drive.url)
    _assert_not_installed(project)
    assert not project.cache.exists()
    assert len(drive.requests) == 1  # a quota page is not hammered with retries


def test_interrupted_download_resumes_with_range(project, drive_factory):
    drive = drive_factory(project.archive_bytes, "truncated", "file")
    sleeps = []
    assert _install(project, sleeps=sleeps, url=drive.url) == "installed"
    _assert_installed(project)
    assert drive.requests[1]["range"] == f"bytes={len(project.archive_bytes) // 3}-"
    assert sleeps == [2]


def test_restart_when_server_ignores_range(project, drive_factory):
    drive = drive_factory(project.archive_bytes, "truncated", "no_range")
    assert _install(project, url=drive.url) == "installed"
    assert project.cache.read_bytes() == project.archive_bytes


def test_partial_download_survives_failed_run_and_resumes_next_time(project, drive_factory):
    broken = drive_factory(project.archive_bytes, "truncated")
    with pytest.raises(ft.RetryableError, match="Повторите команду"):
        _install(project, url=broken.url, attempts=2)
    partial = project.cache.with_name(project.cache.name + ".partial")
    assert partial.stat().st_size == len(project.archive_bytes) // 3
    _assert_not_installed(project)

    good = drive_factory(project.archive_bytes, "file")
    assert _install(project, url=good.url) == "installed"
    assert good.requests[0]["range"] == f"bytes={len(project.archive_bytes) // 3}-"
    assert not partial.exists()


def test_checksum_mismatch_deletes_download_and_installs_nothing(project, drive_factory):
    drive = drive_factory(project.archive_bytes, "corrupt")
    with pytest.raises(ft.FetchError, match="SHA-256"):
        _install(project, url=drive.url)
    _assert_not_installed(project)
    assert not project.cache.exists()
    assert not project.cache.with_name(project.cache.name + ".partial").exists()


def test_wrong_size_is_rejected(project, drive_factory):
    drive = drive_factory(project.archive_bytes, "oversized")
    with pytest.raises(ft.FetchError, match="другой файл"):
        _install(project, url=drive.url)
    _assert_not_installed(project)


def test_transient_http_error_is_retried(project, drive_factory):
    drive = drive_factory(project.archive_bytes, "unavailable", "file")
    sleeps = []
    assert _install(project, sleeps=sleeps, url=drive.url) == "installed"
    assert len(drive.requests) == 2 and sleeps == [2]


def test_permanent_http_error_is_not_retried(project, drive_factory):
    drive = drive_factory(project.archive_bytes, "missing")
    with pytest.raises(ft.FetchError, match="HTTP 404"):
        _install(project, url=drive.url)
    assert len(drive.requests) == 1
    _assert_not_installed(project)


def test_unreachable_server_gives_retry_instructions(project):
    with pytest.raises(ft.RetryableError, match="Повторите команду"):
        _install(project, url="http://127.0.0.1:9/download", attempts=2)
    _assert_not_installed(project)


def test_manual_hint_is_given_for_google_drive(project):
    def offline(url, offset, timeout):
        assert url.startswith("https://drive.usercontent.google.com/download?id=X&")
        raise OSError("network is unreachable")

    with pytest.raises(ft.RetryableError, match="--archive"):
        _install(project, opener=offline, attempts=1)


def test_insufficient_disk_space_is_detected_before_download(project, monkeypatch):
    monkeypatch.setattr(ft.shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(10, 10, 1))
    with pytest.raises(ft.FetchError, match="недостаточно места"):
        _install(project, opener=_no_network)
    _assert_not_installed(project)


def test_local_archive_installs_without_network(project, tmp_path):
    archive = tmp_path / "Загрузки" / "t06_rubert_bundle.tar"
    archive.parent.mkdir()
    archive.write_bytes(project.archive_bytes)
    assert _install(project, archive=archive, opener=_no_network) == "installed"
    _assert_installed(project)
    assert archive.read_bytes() == project.archive_bytes  # the user's file is never moved or deleted


def test_local_archive_with_wrong_checksum_is_rejected(project, tmp_path):
    archive = tmp_path / "t06_rubert_bundle.tar"
    middle = len(project.archive_bytes) // 2
    archive.write_bytes(project.archive_bytes[:middle] + bytes([project.archive_bytes[middle] ^ 1])
                        + project.archive_bytes[middle + 1:])
    with pytest.raises(ft.FetchError, match="SHA-256"):
        _install(project, archive=archive, opener=_no_network)
    _assert_not_installed(project)
    assert archive.exists()


def test_existing_different_bundle_is_never_overwritten(project, drive_factory):
    project.bundle.mkdir(parents=True)
    (project.bundle / "model.safetensors").write_bytes(b"my own experiment")
    drive = drive_factory(project.archive_bytes, "file")
    with pytest.raises(ft.LocalInputError, match="не изменён"):
        _install(project, url=drive.url)
    assert [p.name for p in project.bundle.iterdir()] == ["model.safetensors"]
    assert (project.bundle / "model.safetensors").read_bytes() == b"my own experiment"
    assert drive.requests == []


def test_empty_bundle_directory_is_filled(project, drive_factory):
    project.bundle.mkdir(parents=True)
    drive = drive_factory(project.archive_bytes, "file")
    assert _install(project, url=drive.url) == "installed"
    _assert_installed(project)


def _tar(members) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for name, data, kind in members:
            info = tarfile.TarInfo(name)
            if kind == "file":
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
            else:
                info.type, info.linkname = (tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE), data
                tar.addfile(info)
    return buffer.getvalue()


@pytest.mark.parametrize("bad", [
    ("transformer/../../escaped.txt", b"x", "file"),
    ("/tmp/absolute.txt", b"x", "file"),
    ("transformer\\..\\escaped.txt", b"x", "file"),
    ("transformer/tokenizer.json", "../../../escaped.txt", "symlink"),
    ("transformer/tokenizer.json", "transformer/config.json", "hardlink"),
])
def test_unsafe_archive_members_are_rejected(tmp_path, bad):
    good = [(f"transformer/{n}", d, "file") for n, d in BUNDLE_FILES.items() if n != "tokenizer.json"]
    project = _make_project(tmp_path / "p", _tar(good + [bad]))
    archive = tmp_path / "evil.tar"
    archive.write_bytes(project.archive_bytes)  # its SHA-256 is "expected": only member checks remain
    with pytest.raises(ft.FetchError, match="Unexpected archive member"):
        _install(project, archive=archive)
    _assert_not_installed(project)
    assert not list(tmp_path.rglob("escaped.txt"))


def test_main_exit_codes(project, tmp_path, capsys):
    assert ft.main(["--root", str(project.root), "--archive", str(tmp_path / "missing.tar")]) == 2
    assert "архив не найден" in capsys.readouterr().err
    archive = tmp_path / "ok.tar"
    archive.write_bytes(project.archive_bytes)
    assert ft.main(["--root", str(project.root), "--archive", str(archive)]) == 0
    assert ft.main(["--root", str(project.root)]) == 0  # already installed: no network
    assert "Ничего не скачивалось" in capsys.readouterr().out


def test_repository_bundle_source_matches_accepted_run():
    source = json.loads((ROOT / ft.SOURCE_CONFIG).read_text(encoding="utf-8"))
    assert source["sha256"] == "68970b2e3422a873d14da75a4910f6f813ade73c3ca99c103068f6f2d9e4c703"
    assert source["size_bytes"] == 119_224_320
    assert not Path(source["download_dir"]).is_absolute()
    expected = tb.expected_files(ROOT, "reports/transformer")
    assert len(expected) == 7 and "model.safetensors" in expected


@pytest.mark.parametrize("path, record, key", [
    ("src/train_transformer.py", "reports/transformer/run_metadata.json", "script_sha256"),
    ("reports/baseline/run_metadata.json", "reports/transformer/run_metadata.json",
     "data_version.baseline_metadata_sha256"),
    ("reports/baseline/train_ids.csv", "reports/transformer/run_metadata.json",
     "data_version.baseline_train_ids_file_sha256"),
    ("reports/data_prep/audit.json", "reports/baseline/run_metadata.json", "t03_audit_sha256"),
    ("reports/data_prep/split_ids.csv", "reports/baseline/run_metadata.json", "t03_split_ids_sha256"),
])
def test_checked_out_files_keep_recorded_bytes(path, record, key):
    """Guards against CRLF conversion on Windows checkouts (see .gitattributes)."""
    value = json.loads((ROOT / record).read_text(encoding="utf-8"))
    for part in key.split("."):
        value = value[part]
    assert tb.sha256_file(ROOT / path) == value, f"{path} bytes changed on checkout (line endings?)"
