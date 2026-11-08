"""Download, verify and install the accepted T06 RuBERT bundle: one command on any OS.

    python tools/fetch_transformer.py                      # Google Drive -> models/transformer
    python tools/fetch_transformer.py --archive FILE.tar   # pre-downloaded archive, no network
    python tools/fetch_transformer.py --url URL            # any mirror of the same archive

Standard library only: no curl, wget, tar, gdown or Google account. The model counts
as installed only after the archive SHA-256 (``configs/transformer_bundle.json``) and
all bundle files (``reports/transformer/run_metadata.json::bundle_files_sha256``) match.
Installation is ``tools/transformer_bundle.py restore``: only expected regular files are
extracted into a temporary sibling directory and installed with a single rename.

Re-running is safe: a verified bundle is left untouched and nothing is downloaded; a
bundle that differs is never overwritten. The verified archive stays in
``artifacts/downloads/`` (outside Git) for offline reinstalls, and an interrupted
download resumes from ``<archive>.partial``.

Exit code: 0 — bundle installed or already present and verified; 1 — download,
verification or installation failed (existing files unchanged); 2 — local input problem.
"""
from __future__ import annotations

import argparse
import html.parser
import http.client
import json
import os
import shutil
import ssl
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import transformer_bundle as tb  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE_CONFIG = "configs/transformer_bundle.json"
TRAINING_CONFIG = "configs/transformer.json"
CHUNK = 1 << 20
SPACE_MARGIN = 64 << 20
MAX_HTML = 1 << 20
MAX_HOPS = 3
RETRYABLE_HTTP = {408, 425, 429, 500, 502, 503, 504}
RETRY_HINT = "Повторите команду позже; скачанная часть сохранится и загрузка продолжится."
MANUAL_HINT = ("Скачайте архив вручную по ссылке {view} (кнопка «Скачать») и выполните:\n"
               "  python tools/fetch_transformer.py --archive <путь к t06_rubert_bundle.tar>")


class FetchError(Exception):
    """Download or verification failed; the message says what to do next."""


class RetryableError(FetchError):
    """Transient network failure: the same request may succeed later."""


class LocalInputError(Exception):
    """Problem with local files or arguments, not with the network."""


def drive_url(file_id: str) -> str:
    query = urllib.parse.urlencode({"id": file_id, "export": "download", "confirm": "t"})
    return f"https://drive.usercontent.google.com/download?{query}"


class _DownloadForm(html.parser.HTMLParser):
    """Google Drive "can't scan for viruses" page: a GET form with hidden fields."""

    def __init__(self):
        super().__init__()
        self.action: str | None = None
        self.fields: dict[str, str] = {}
        self._inside = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "form" and attrs.get("id") == "download-form":
            self._inside, self.action = True, attrs.get("action")
        elif tag == "input" and self._inside and attrs.get("type") == "hidden" and attrs.get("name"):
            self.fields[attrs["name"]] = attrs.get("value") or ""

    def handle_endtag(self, tag):
        if tag == "form":
            self._inside = False


def confirmation_url(page: str, base_url: str) -> str | None:
    form = _DownloadForm()
    form.feed(page)
    if not form.action or not form.fields:
        return None
    return f"{urllib.parse.urljoin(base_url, form.action)}?{urllib.parse.urlencode(form.fields)}"


def _ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    try:  # python.org builds on macOS may lack system CAs; certifi ships with requests/httpx
        import certifi
        context.load_verify_locations(certifi.where())
    except (ImportError, OSError):
        pass
    return context


def open_url(url: str, offset: int, timeout: float):
    headers = {"User-Agent": "ru-review-sentiment-bot/fetch_transformer"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                  timeout=timeout, context=_ssl_context())


def _mib(n: int) -> str:
    return f"{n / (1 << 20):.1f} МиБ"


def _connection_error(exc: Exception) -> FetchError:
    reason = getattr(exc, "reason", exc)
    if isinstance(reason, ssl.SSLCertVerificationError):
        return FetchError(f"не удалось проверить HTTPS-сертификат ({reason}). На macOS с Python с python.org "
                          "запустите «Install Certificates.command» из папки Python или выполните "
                          "python -m pip install certifi; за корпоративным прокси настройте его сертификат.")
    return RetryableError(f"сетевая ошибка: {reason}")


def _download_once(url: str, partial: Path, size: int, timeout: float, opener, log) -> None:
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        partial.unlink()
        offset = 0
    try:
        for _ in range(MAX_HOPS):
            response = opener(url, offset, timeout)
            if not response.headers.get("Content-Type", "").lower().startswith("text/html"):
                break
            with response:
                page = response.read(MAX_HTML).decode("utf-8", "replace")
                next_url = confirmation_url(page, response.geturl())
            if next_url is None:
                raise FetchError("вместо архива получена HTML-страница: Google Диск временно ограничил "
                                 "скачивание файла, закрыл доступ или ссылка устарела.")
            url = next_url
        else:
            raise FetchError("Google Диск несколько раз подряд вернул страницу подтверждения вместо архива.")
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and offset:
            partial.unlink()
            raise RetryableError("сервер отклонил докачку, загрузка начнётся заново") from exc
        if exc.code in RETRYABLE_HTTP:
            raise RetryableError(f"сервер ответил HTTP {exc.code} {exc.reason}") from exc
        raise FetchError(f"сервер ответил HTTP {exc.code} {exc.reason}: файл недоступен по ссылке "
                         "(удалён, закрыт доступ или неверный адрес).") from exc
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        raise _connection_error(exc) from exc

    with response:
        resumed = (offset and response.status == 206
                   and response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"))
        if offset and not resumed:
            log("Сервер не поддержал докачку, загрузка начинается заново.")
            offset = 0
        length = response.headers.get("Content-Length")
        if length is not None and length.isdigit() and offset + int(length) != size:
            raise FetchError(f"сервер отдаёт {offset + int(length)} байт, ожидалось {size}: "
                             "по ссылке находится другой файл.")
        written, reported = offset, offset * 10 // size
        try:
            fh = partial.open("ab" if resumed else "wb")
        except OSError as exc:
            raise FetchError(f"не удалось создать {partial}: {exc}") from exc
        with fh:
            while True:
                try:
                    chunk = response.read(CHUNK)
                except (OSError, http.client.HTTPException) as exc:
                    raise RetryableError(f"соединение прервано на {_mib(written)} из {_mib(size)}: {exc}") from exc
                if not chunk:
                    break
                written += len(chunk)
                if written > size:
                    fh.close()
                    partial.unlink()
                    raise FetchError(f"сервер прислал больше {size} байт: по ссылке находится другой файл.")
                try:
                    fh.write(chunk)
                except OSError as exc:
                    raise FetchError(f"ошибка записи {partial}: {exc}. Проверьте свободное место.") from exc
                if written * 10 // size > reported:
                    reported = written * 10 // size
                    log(f"  скачано {_mib(written)} из {_mib(size)}")
    if written < size:
        raise RetryableError(f"загрузка оборвалась: получено {_mib(written)} из {_mib(size)}")


def fetch_archive(url: str, dest: Path, sha256: str, size: int, *, attempts: int = 3, timeout: float = 60,
                  opener=open_url, sleep=time.sleep, log=print) -> None:
    """Download to ``<dest>.partial`` (resuming it), check size and SHA-256, then rename to ``dest``."""
    partial = dest.with_name(dest.name + ".partial")
    for attempt in range(1, attempts + 1):
        try:
            if not (partial.exists() and partial.stat().st_size == size):
                _download_once(url, partial, size, timeout, opener, log)
            break
        except RetryableError as exc:
            if attempt == attempts:
                raise RetryableError(f"{exc}. Все попытки ({attempts}) неудачны. {RETRY_HINT}") from exc
            delay = 2 ** attempt
            log(f"Попытка {attempt}/{attempts} не удалась: {exc}. Повтор через {delay} с.")
            sleep(delay)
    log("Проверяю SHA-256 скачанного архива…")
    digest = tb.sha256_file(partial)
    if digest != sha256:
        partial.unlink()
        raise FetchError(f"SHA-256 скачанного архива {digest} не совпадает с ожидаемой {sha256}. "
                         "Скачанный файл удалён, установка не выполнялась.")
    os.replace(partial, dest)


def check_free_space(requirements) -> None:
    """``requirements``: (directory, bytes) pairs; needs on the same volume are added up."""
    volumes: dict[int, list] = {}
    for path, need in requirements:
        probe = Path(path).absolute()
        while not probe.exists():
            probe = probe.parent
        volumes.setdefault(os.stat(probe).st_dev, [probe, 0])[1] += need
    for probe, need in volumes.values():
        free = shutil.disk_usage(probe).free
        if free < need + SPACE_MARGIN:
            raise FetchError(f"недостаточно места на диске с {probe}: свободно {_mib(free)}, "
                             f"нужно около {_mib(need + SPACE_MARGIN)}. Освободите место или укажите "
                             "--download-dir на другом диске.")


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise LocalInputError(f"не удалось прочитать {path}: {exc}") from exc


def _display(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def install(root: Path = ROOT, *, archive: Path | None = None, url: str | None = None,
            download_dir: Path | None = None, attempts: int = 3, timeout: float = 60,
            opener=open_url, sleep=time.sleep, log=print) -> str:
    """Return ``"present"`` if a verified bundle was already installed, else ``"installed"``."""
    source = _read_json(root / SOURCE_CONFIG)
    training = _read_json(root / TRAINING_CONFIG)
    bundle = root / training["model_dir"]
    try:
        expected = tb.expected_files(root, training["report_dir"])
    except (OSError, ValueError, KeyError) as exc:
        raise LocalInputError(f"нет списка файлов комплекта в {training['report_dir']}/run_metadata.json: {exc}")
    size, sha256 = source["size_bytes"], source["sha256"]

    if bundle.exists() and (not bundle.is_dir() or any(bundle.iterdir())):
        try:
            tb.check_directory(bundle, expected)
        except (OSError, ValueError) as exc:
            raise LocalInputError(f"{_display(bundle, root)} уже существует, но не совпадает с принятым "
                                  f"комплектом ({exc}). Каталог не изменён. Переименуйте или удалите его "
                                  "и повторите команду.") from exc
        log(f"Модель уже установлена и проверена: {_display(bundle, root)} ({len(expected)} файлов). "
            "Ничего не скачивалось.")
        return "present"

    if archive is not None:
        if not archive.is_file():
            raise LocalInputError(f"архив не найден: {archive}")
        log(f"Проверяю SHA-256 архива {archive}…")
        if tb.sha256_file(archive) != sha256:
            raise FetchError(f"SHA-256 архива {archive} не совпадает с ожидаемой {sha256}: файл повреждён, "
                             "скачан не полностью или это другой архив. Скачайте его заново.")
        check_free_space([(bundle.parent, size)])
    else:
        archive = (download_dir or root / source["download_dir"]) / source["archive_name"]
        if archive.exists():
            if tb.sha256_file(archive) == sha256:
                log(f"Использую ранее скачанный проверенный архив {_display(archive, root)}.")
            else:
                log(f"Ранее скачанный {_display(archive, root)} повреждён; удаляю и скачиваю заново.")
                archive.unlink()
        if not archive.exists():
            partial = archive.with_name(archive.name + ".partial")
            have = partial.stat().st_size if partial.exists() else 0
            check_free_space([(archive.parent, max(size - have, 0)), (bundle.parent, size)])
            try:
                archive.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise LocalInputError(f"не удалось создать каталог {archive.parent}: {exc}") from exc
            source_url = url or drive_url(source["google_drive_file_id"])
            log(f"Скачиваю {source['archive_name']} ({_mib(size)}) в {_display(archive, root)}…")
            try:
                fetch_archive(source_url, archive, sha256, size, attempts=attempts, timeout=timeout,
                              opener=opener, sleep=sleep, log=log)
            except FetchError as exc:
                hint = "" if url else "\n" + MANUAL_HINT.format(view=source["google_drive_view_url"])
                raise type(exc)(f"{exc}{hint}") from exc
            log("Архив скачан, размер и SHA-256 совпали.")

    log(f"Восстанавливаю комплект в {_display(bundle, root)}…")
    try:
        tb.restore(archive, bundle, expected, sha256)
        tb.check_directory(bundle, expected)
    except (OSError, ValueError, tarfile.TarError) as exc:
        raise FetchError(f"восстановление не выполнено: {type(exc).__name__}: {exc}. Временные файлы удалены, "
                         "существующие файлы не изменены. Повторите команду.") from exc
    log(f"Модель готова: {_display(bundle, root)}, все {len(expected)} файлов совпали с "
        f"{training['report_dir']}/run_metadata.json.")
    return "installed"


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:  # Russian messages on consoles/redirects with a non-UTF-8 code page
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    parser.add_argument("--archive", type=Path, help="заранее скачанный t06_rubert_bundle.tar (без сети)")
    parser.add_argument("--url", help="другой адрес того же архива (зеркало); SHA-256 проверяется так же")
    parser.add_argument("--download-dir", type=Path, help="куда скачивать архив (по умолчанию artifacts/downloads)")
    parser.add_argument("--attempts", type=int, default=3, help="число попыток скачивания (по умолчанию 3)")
    parser.add_argument("--timeout", type=float, default=60, help="тайм-аут сети в секундах (по умолчанию 60)")
    args = parser.parse_args(argv)
    if args.archive and (args.url or args.download_dir):
        parser.error("--archive нельзя сочетать с --url и --download-dir")
    if args.attempts < 1:
        parser.error("--attempts должно быть не меньше 1")
    try:
        install(args.root, archive=args.archive, url=args.url, download_dir=args.download_dir,
                attempts=args.attempts, timeout=args.timeout, log=lambda m: print(m, flush=True))
    except LocalInputError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 2
    except FetchError as exc:
        print(f"Модель не установлена: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nПрервано. Существующие файлы не изменены; повторный запуск продолжит загрузку.", file=sys.stderr)
        return 1
    print("Проверка модели:\n  python tools/restore_nlp_splits.py\n  python tools/verify_transformer.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
