"""Pack and restore the T06 bundle outside Git, verified against run_metadata.json.

    python tools/transformer_bundle.py pack    --out /storage/t06_bundle.tar
    python tools/transformer_bundle.py restore --archive /storage/t06_bundle.tar

``pack`` refuses to archive a bundle whose files differ from
``reports/transformer/run_metadata.json::bundle_files_sha256`` and writes
``<archive>.sha256`` next to the archive. ``restore`` checks the archive digest
and every member, extracts into a temporary sibling directory and installs it
with a single ``rename``; an interrupted restore never leaves a mixed bundle.
Existing non-empty bundle directories are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
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
    if present != sorted(expected) or not all((bundle / n).is_file() for n in present):
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
    partial.rename(out)
    out.with_name(out.name + ".sha256").write_text(f"{digest}  {out.name}\n", encoding="utf-8")
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
            for member in tar.getmembers():
                # Only expected regular files; no links, absolute paths or traversal.
                if not member.isfile() or member.name not in allowed:
                    raise ValueError(f"Unexpected archive member: {member.name}")
                source = tar.extractfile(member)
                with (staging / Path(member.name).name).open("wb") as fh:
                    shutil.copyfileobj(source, fh)
        (staging / "BUNDLE_SHA256SUMS.json").unlink(missing_ok=True)
        check_directory(staging, expected)
        if bundle.exists():
            bundle.rmdir()  # empty, checked above
        staging.rename(bundle)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("pack", "restore", "check"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--report-dir", default="reports/transformer")
    parser.add_argument("--bundle", type=Path, help="default: <root>/models/transformer")
    parser.add_argument("--out", type=Path, help="pack: archive path outside Git")
    parser.add_argument("--archive", type=Path, help="restore: archive path")
    parser.add_argument("--sha256", help="restore: expected archive SHA-256 (else <archive>.sha256)")
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
        else:
            check_directory(bundle, expected)
            print(f"Bundle matches run metadata: {len(expected)} files")
    except (ValueError, OSError, KeyError, tarfile.TarError) as exc:
        print(f"T06 bundle: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
