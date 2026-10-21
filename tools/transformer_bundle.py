"""Pack and restore the T06 bundle outside Git, verified against run_metadata.json.

    python tools/transformer_bundle.py pack    --out /storage/t06_bundle.tar
    python tools/transformer_bundle.py restore --archive /storage/t06_bundle.tar
    python tools/transformer_bundle.py reconstruct --source models/transformer_repeat

``pack`` refuses to archive a bundle whose files differ from
``reports/transformer/run_metadata.json::bundle_files_sha256`` and writes
``<archive>.sha256`` next to the archive. ``restore`` checks the archive digest
and every member, extracts into a temporary sibling directory and installs it
with a single ``rename``; an interrupted restore never leaves a mixed bundle.
Existing non-empty bundle directories are never overwritten.

``reconstruct`` rebuilds the recorded bundle from a repeat run of the same
configuration (``configs/transformer_repeat.json``): weights, tokenizer, HF config
and mapping are copied, while ``training_config.json`` and ``metadata.json`` are
re-serialized from the committed ``config.json`` and ``run_metadata.json``. It
succeeds only if all files equal ``bundle_files_sha256``, so a repeat run that is
not bitwise identical can never pass for the original checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "transformer"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_files(root: Path, report_dir: str) -> dict[str, str]:
    metadata = json.loads((root / report_dir / "run_metadata.json").read_text(encoding="utf-8"))
    return metadata["bundle_files_sha256"]


def check_directory(bundle: Path, expected: dict[str, str]) -> None:
    present = sorted(p.name for p in bundle.iterdir())
    # Symlinks are rejected: tarfile would archive the link itself, not the weights.
    if present != sorted(expected) or not all(
            (bundle / n).is_file() and not (bundle / n).is_symlink() for n in present):
        raise ValueError(f"Bundle files {present} differ from run metadata {sorted(expected)}")
    for name, digest in expected.items():
        if sha256_file(bundle / name) != digest:
            raise ValueError(f"Checksum mismatch: {name}")


def pack(bundle: Path, out: Path, expected: dict[str, str]) -> str:
    check_directory(bundle, expected)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    # Deterministic, uncompressed: safetensors do not compress well, and order/mtime are fixed.
    with tarfile.open(partial, "w", format=tarfile.PAX_FORMAT) as tar:
        manifest = json.dumps({"bundle_files_sha256": expected}, indent=2, sort_keys=True).encode()
        for name in sorted(expected):
            info = tar.gettarinfo(bundle / name, arcname=f"{PREFIX}/{name}")
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.mtime, info.mode = 0, 0o644
            with (bundle / name).open("rb") as fh:
                tar.addfile(info, fh)
        info = tarfile.TarInfo(f"{PREFIX}/BUNDLE_SHA256SUMS.json")
        info.size, info.mode = len(manifest), 0o644
        tar.addfile(info, io.BytesIO(manifest))
    digest = sha256_file(partial)
    sidecar = out.with_name(out.name + ".sha256")
    sidecar_partial = sidecar.with_name(sidecar.name + ".partial")
    sidecar_partial.write_text(f"{digest}  {out.name}\n", encoding="utf-8")
    partial.rename(out)
    sidecar_partial.rename(sidecar)
    return digest


def restore(archive: Path, bundle: Path, expected: dict[str, str], archive_sha256: str | None) -> None:
    sidecar = archive.with_name(archive.name + ".sha256")
    if archive_sha256 is None and sidecar.exists():
        archive_sha256 = sidecar.read_text(encoding="utf-8").split()[0]
    if archive_sha256 is None:
        raise ValueError("Archive digest unknown: pass --sha256 or keep <archive>.sha256 next to it")
    if sha256_file(archive) != archive_sha256:
        raise ValueError("Archive SHA-256 mismatch")
    if bundle.exists() and any(bundle.iterdir()):
        raise FileExistsError(f"Bundle directory is not empty: {bundle}")
    bundle.parent.mkdir(parents=True, exist_ok=True)
    allowed = {f"{PREFIX}/{n}" for n in expected} | {f"{PREFIX}/BUNDLE_SHA256SUMS.json"}
    with tempfile.TemporaryDirectory(prefix=".restore-", dir=bundle.parent) as tmp:
        staging = Path(tmp) / PREFIX
        staging.mkdir()
        with tarfile.open(archive, "r") as tar:
            seen = set()
            for member in tar.getmembers():
                # Only expected regular files, each once; no links, absolute paths or traversal.
                if not member.isfile() or member.name not in allowed:
                    raise ValueError(f"Unexpected archive member: {member.name}")
                if member.name in seen:
                    raise ValueError(f"Duplicate archive member: {member.name}")
                seen.add(member.name)
                name = Path(member.name).name
                source = tar.extractfile(member)
                digest = hashlib.sha256()
                with (staging / name).open("wb") as fh:
                    for chunk in iter(lambda: source.read(1 << 20), b""):
                        digest.update(chunk)
                        fh.write(chunk)
                # Fail on the first bad member instead of after unpacking the whole archive.
                if name in expected and digest.hexdigest() != expected[name]:
                    raise ValueError(f"Checksum mismatch: {name}")
        (staging / "BUNDLE_SHA256SUMS.json").unlink(missing_ok=True)
        check_directory(staging, expected)
        if bundle.exists():
            bundle.rmdir()  # empty, checked above
        staging.rename(bundle)


def _json_bytes(data) -> bytes:
    # Same serialization as src.train_baseline.write_json used by the training run.
    return (json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def reconstruct(source: Path, bundle: Path, root: Path, report_dir: str, expected: dict[str, str]) -> None:
    if bundle.exists() and any(bundle.iterdir()):
        raise FileExistsError(f"Bundle directory is not empty: {bundle}")
    report = root / report_dir
    metadata = json.loads((report / "run_metadata.json").read_text(encoding="utf-8"))
    metadata.pop("bundle_files_sha256")
    generated = {"training_config.json": _json_bytes(json.loads((report / "config.json").read_text(encoding="utf-8"))),
                 "metadata.json": _json_bytes(metadata)}
    bundle.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".reconstruct-", dir=bundle.parent) as tmp:
        staging = Path(tmp) / PREFIX
        staging.mkdir()
        for name in expected:
            if name in generated:
                # write_json uses text mode: LF on Linux/macOS, CRLF on Windows. Rebuild the
                # bytes of the recorded run, whichever platform produced it.
                data = generated[name]
                crlf = data.replace(b"\n", b"\r\n")
                (staging / name).write_bytes(crlf if hashlib.sha256(crlf).hexdigest() == expected[name] else data)
            else:
                path = source / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError(f"Source bundle lacks regular file: {name}")
                (staging / name).write_bytes(path.read_bytes())
        check_directory(staging, expected)
        if bundle.exists():
            bundle.rmdir()  # empty, checked above
        staging.rename(bundle)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("pack", "restore", "check", "reconstruct"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report-dir", default="reports/transformer")
    parser.add_argument("--bundle", type=Path, help="default: <root>/models/transformer")
    parser.add_argument("--out", type=Path, help="pack: archive path outside Git")
    parser.add_argument("--archive", type=Path, help="restore: archive path")
    parser.add_argument("--sha256", help="restore: expected archive SHA-256 (else <archive>.sha256)")
    parser.add_argument("--source", type=Path, help="reconstruct: bundle of a repeat run with the same config")
    args = parser.parse_args(argv)
    bundle = args.bundle or args.root / "models/transformer"
    try:
        expected = expected_files(args.root, args.report_dir)
        if args.command == "pack":
            if not args.out:
                parser.error("pack requires --out")
            print(f"{pack(bundle, args.out, expected)}  {args.out}")
        elif args.command == "restore":
            if not args.archive:
                parser.error("restore requires --archive")
            restore(args.archive, bundle, expected, args.sha256)
            print(f"Restored and verified {len(expected)} files into {bundle}")
        elif args.command == "reconstruct":
            if not args.source:
                parser.error("reconstruct requires --source")
            reconstruct(args.source, bundle, args.root, args.report_dir, expected)
            print(f"Reconstructed and verified {len(expected)} files into {bundle}")
        else:
            check_directory(bundle, expected)
            print(f"Bundle matches run metadata: {len(expected)} files")
    except (ValueError, OSError, KeyError, tarfile.TarError) as exc:
        print(f"T06 bundle: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
