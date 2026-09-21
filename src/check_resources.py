"""T05: real resource check for RuBERT (training step) + faster-whisper on the defence machine.

What a PASS means (methodology ``t05.2``):

* a real forward/backward/optimizer step of ``cointegrated/rubert-tiny2`` on a small batch taken
  from the common T04 train, padded to the profile ``max_length``; transformer layers AND the
  classifier head must change after the step;
* a real ``faster-whisper`` recognition of a separate debug Russian recording, with every segment
  consumed inside the timed section;
* RuBERT and Whisper alive in one process at the same time, each performing a useful operation
  while the other one stays loaded;
* measured time and memory (process RSS, system RAM headroom, CUDA memory) and explicit warnings.

The final 36 voice recordings and the final text test are never read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import wave
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

METHODOLOGY = "t05.2"
MIB = 1024**2

LABELS = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {value: key for key, value in LABELS.items()}
REQUIRED_TRAIN_COLUMNS = {"review_text", "label_id", "label_name", "record_id"}
FORBIDDEN_ID_EXPORT_COLUMNS = {"review_text", "text", "transcript", "segments"}
FORBIDDEN_REPORT_KEYS = {"review_text", "text", "transcript", "segments"}
FORBIDDEN_DATA_KEYS = {
    "test",
    "test_path",
    "test_file",
    "final_test",
    "final_test_path",
    "final_audio_test",
}

_TEST_TOKENS = {"test", "tests", "testing"}
_KEY_DATA_TOKENS = {"final", "path", "file", "parquet", "csv", "data", "split", "dir"}
_VALIDATION_TOKENS = {"validation", "valid", "val"}
_PATH_VALUE_SUFFIXES = (".parquet", ".csv", ".tsv", ".json", ".jsonl", ".txt")

HEAD_PARAMETER_PREFIX = "classifier."
EXPECTED_NEW_HEAD_KEYS = {"classifier.weight", "classifier.bias"}
# Tensors above this size (word embeddings: ~26M values) are not snapshotted: the snapshot would
# distort the very RAM/VRAM numbers this check exists to measure.
SNAPSHOT_MAX_ELEMENTS = 2_000_000
_ENCODER_LAYER_RE = re.compile(r"(?:^|\.)encoder\.layer\.(\d+)\.")

MIN_RAM_HEADROOM_MIB = 1024.0
MIN_VRAM_HEADROOM_MIB = 1024.0
MIN_TEXT_LEAK_CHARS = 12
LEAK_WINDOW_CHARS = 20
LEAK_WINDOW_STEP = 5
FINAL_AUDIO_NAME_MARKERS = ("final", "финал")

CPU_COMPUTE_TYPES = {"int8", "int8_float32", "int16", "float32"}
ALL_COMPUTE_TYPES = CPU_COMPUTE_TYPES | {"float16", "bfloat16", "int8_float16", "int8_bfloat16"}
WHISPER_SIZE_RANK = {
    "tiny": 0,
    "tiny.en": 0,
    "base": 1,
    "base.en": 1,
    "small": 2,
    "small.en": 2,
    "medium": 3,
    "medium.en": 3,
    "large-v1": 4,
    "large-v2": 4,
    "large-v3": 4,
    "large": 4,
}


class T05Error(RuntimeError):
    """Expected, user-actionable failure in the T05 resource check."""


@dataclass
class RuntimeObjects:
    torch: Any
    tokenizer: Any
    model: Any
    whisper_model: Any | None = None


# --------------------------------------------------------------------------- #
# Memory monitoring
# --------------------------------------------------------------------------- #
def process_rss_mib(psutil_module: Any) -> float:
    return psutil_module.Process(os.getpid()).memory_info().rss / MIB


def process_peak_rss_mib(psutil_module: Any) -> float | None:
    """Peak resident memory of this process since it started, measured by the OS.

    Windows: ``peak_wset`` (peak working set) from psutil. Linux/macOS: ``ru_maxrss``.
    Unlike sampling, this cannot miss a short spike (for example while a model is loading).
    """
    try:
        info = psutil_module.Process(os.getpid()).memory_info()
        peak = getattr(info, "peak_wset", None)
        if peak:
            return float(peak) / MIB
    except Exception:
        pass
    try:
        import resource  # POSIX only

        maxrss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    return maxrss / 1024.0 if sys.platform.startswith("linux") else maxrss / MIB


class PeakRSSMonitor:
    """Sample process RSS and system-available RAM in the background."""

    def __init__(self, psutil_module: Any, interval_seconds: float = 0.05):
        self.psutil_module = psutil_module
        self.interval_seconds = max(0.005, float(interval_seconds))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak_mib = process_rss_mib(psutil_module)
        self.min_available_mib = self._available_mib()

    def _available_mib(self) -> float | None:
        try:
            return float(self.psutil_module.virtual_memory().available) / MIB
        except Exception:
            return None

    def _sample_once(self, process: Any) -> None:
        try:
            self.peak_mib = max(self.peak_mib, process.memory_info().rss / MIB)
        except Exception:
            pass
        available = self._available_mib()
        if available is not None:
            if self.min_available_mib is None:
                self.min_available_mib = available
            else:
                self.min_available_mib = min(self.min_available_mib, available)

    def _sample_loop(self) -> None:
        process = self.psutil_module.Process(os.getpid())
        while not self._stop.is_set():
            self._sample_once(process)
            self._stop.wait(self.interval_seconds)

    def start(self) -> "PeakRSSMonitor":
        if self._thread is None:
            self._stop.clear()
            self._thread = threading.Thread(target=self._sample_loop, daemon=True)
            self._thread.start()
        return self

    def stop(self) -> float:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(0.5, self.interval_seconds * 4))
            self._thread = None
        self._sample_once(self.psutil_module.Process(os.getpid()))
        return self.peak_mib

    def __enter__(self) -> "PeakRSSMonitor":
        return self.start()

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop()


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def package_version(distribution_name: str) -> str | None:
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def repo_root_from_script() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_repo_path(root: Path, value: str) -> Path:
    """Resolve a CLI path (absolute paths allowed, e.g. the debug audio outside the repository)."""
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def resolve_config_path(root: Path, value: str) -> Path:
    """Resolve a path taken from the config: must stay inside the repository."""
    path = (root / value).resolve()
    if not path.is_relative_to(root.resolve()):
        raise T05Error(f"Config path leaves the repository root: {value!r}")
    return path


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def dumps_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False)


def write_text_lf(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def _synchronize(torch_module: Any, device: str) -> None:
    """Wait for queued GPU work; without this CUDA timings measure only kernel launches."""
    if device == "cuda":
        torch_module.cuda.synchronize()


# --------------------------------------------------------------------------- #
# Config validation
# --------------------------------------------------------------------------- #
def _walk_config(obj: Any, path: tuple[str, ...] = ()) -> Iterable[tuple[tuple[str, ...], Any]]:
    if isinstance(obj, dict):
        for key, value in obj.items():
            next_path = path + (str(key),)
            yield next_path, value
            yield from _walk_config(value, next_path)
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            next_path = path + (str(idx),)
            yield next_path, value
            yield from _walk_config(value, next_path)


def _name_tokens(name: str) -> set[str]:
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return {t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t}


def _path_segments(value: str) -> list[str]:
    return [part for part in re.split(r"[\\/]", value) if part not in ("", ".")]


def _looks_like_test_path(value: str) -> bool:
    return any(_name_tokens(part) & _TEST_TOKENS for part in _path_segments(value))


def _looks_like_validation_path(value: str) -> bool:
    return any(_name_tokens(part) & _VALIDATION_TOKENS for part in _path_segments(value))


def _looks_like_path_value(value: str) -> bool:
    return "/" in value or "\\" in value or value.lower().endswith(_PATH_VALUE_SUFFIXES)


def _is_forbidden_key(key: str) -> bool:
    if key.lower().strip() in FORBIDDEN_DATA_KEYS:
        return True
    tokens = _name_tokens(key)
    return bool(tokens & _TEST_TOKENS) and bool(tokens & _KEY_DATA_TOKENS)


def reject_final_test_references(config: dict[str, Any]) -> None:
    """Reject data hooks that could reach the protected final test.

    Keys: exact forbidden names and ``test`` combined with a data-like word (``testPath``,
    ``final_test_file``). A harmless key such as ``smoke_test_enabled`` is not rejected here (the
    strict schema rejects unknown keys anyway). Path-like values are compared token by token, so the
    real source name ``test-00000-of-00001.parquet`` is caught as well as ``test.parquet``.
    """
    for key_path, value in _walk_config(config):
        if not key_path:
            continue
        if _is_forbidden_key(key_path[-1]):
            raise T05Error(f"T05 config must not contain final-test data key: {'.'.join(key_path)}")
        if isinstance(value, str) and (key_path[0] == "paths" or _looks_like_path_value(value)):
            if _looks_like_test_path(value):
                raise T05Error(
                    f"T05 config must not reference final test data: {'.'.join(key_path)}={value!r}"
                )


def load_config(path: Path) -> dict[str, Any]:
    try:
        config = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError as exc:
        raise T05Error(f"Config not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise T05Error(f"Invalid JSON in config {path}: {exc}") from exc
    if not isinstance(config, dict):
        raise T05Error("Config root must be a JSON object.")
    reject_final_test_references(config)
    validate_config(config)
    return config


def _reject_unknown_keys(obj: dict[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(obj) - allowed
    if unknown:
        raise T05Error(f"Unknown config keys in {where}: {sorted(unknown)}")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_relative_path(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise T05Error(f"{name} must be a non-empty string.")
    win = PureWindowsPath(value)
    if win.drive or win.is_absolute() or PurePosixPath(value).is_absolute():
        raise T05Error(f"{name} must be a path relative to the repository root, got {value!r}.")
    if ".." in re.split(r"[\\/]", value):
        raise T05Error(f"{name} must not leave the repository root: {value!r}.")


def validate_config(config: dict[str, Any]) -> None:
    reject_final_test_references(config)
    required_top = {
        "seed", "machine_label", "paths", "labels", "rubert", "profiles",
        "debug_audio", "remote_fallback",
    }
    missing = required_top - set(config)
    if missing:
        raise T05Error(f"Missing config sections: {sorted(missing)}")
    _reject_unknown_keys(config, required_top, "root")

    if not _is_int(config["seed"]):
        raise T05Error("seed must be an integer.")
    if not isinstance(config["machine_label"], str) or not config["machine_label"].strip():
        raise T05Error("machine_label must be a non-empty, non-sensitive machine label.")
    if config["labels"] != LABELS:
        raise T05Error(f"labels must be exactly {LABELS}.")

    paths = config["paths"]
    if not isinstance(paths, dict):
        raise T05Error("paths must be an object.")
    allowed_paths = {
        "train_parquet", "train_ids_csv", "baseline_run_metadata", "split_ids_csv",
        "reports_dir", "local_artifacts_dir",
    }
    _reject_unknown_keys(paths, allowed_paths, "paths")
    for key in sorted(allowed_paths):
        _check_relative_path(f"paths.{key}", paths.get(key))
    for key in ("train_parquet", "train_ids_csv"):
        if _looks_like_validation_path(paths[key]):
            raise T05Error(f"paths.{key} must point to train data, not validation: {paths[key]!r}.")
    reports_parts = _path_segments(paths["reports_dir"].lower())
    artifacts_parts = _path_segments(paths["local_artifacts_dir"].lower())
    if artifacts_parts[: len(reports_parts)] == reports_parts or artifacts_parts[:1] == ["reports"]:
        raise T05Error(
            "paths.local_artifacts_dir (holds the full transcript) must not be inside the "
            "Git-tracked reports directory."
        )

    rubert = config["rubert"]
    if not isinstance(rubert, dict):
        raise T05Error("rubert must be an object.")
    _reject_unknown_keys(rubert, {"model_name", "learning_rate", "weight_decay", "training_steps"}, "rubert")
    if rubert.get("model_name") != "cointegrated/rubert-tiny2":
        raise T05Error("T05 requires rubert.model_name=cointegrated/rubert-tiny2.")
    lr = rubert.get("learning_rate")
    if not _is_number(lr) or not math.isfinite(float(lr)) or lr <= 0:
        raise T05Error("rubert.learning_rate must be a finite number > 0.")
    wd = rubert.get("weight_decay")
    if not _is_number(wd) or not math.isfinite(float(wd)) or wd < 0:
        raise T05Error("rubert.weight_decay must be a finite number >= 0.")
    steps = rubert.get("training_steps")
    if not _is_int(steps) or not 1 <= steps <= 3:
        raise T05Error("rubert.training_steps must be an integer from 1 to 3 for this smoke check.")

    audio = config["debug_audio"]
    if not isinstance(audio, dict):
        raise T05Error("debug_audio must be an object.")
    _reject_unknown_keys(audio, {"language", "max_seconds", "max_bytes"}, "debug_audio")
    if audio.get("language") != "ru":
        raise T05Error("debug_audio.language must be 'ru'.")
    max_seconds = audio.get("max_seconds")
    if not _is_number(max_seconds) or not math.isfinite(float(max_seconds)) or max_seconds <= 0:
        raise T05Error("debug_audio.max_seconds must be > 0.")
    max_bytes = audio.get("max_bytes")
    if not _is_int(max_bytes) or max_bytes <= 0:
        raise T05Error("debug_audio.max_bytes must be a positive integer.")

    profiles = config["profiles"]
    if not isinstance(profiles, dict) or set(profiles) != {"primary", "reserve"}:
        raise T05Error("profiles must contain exactly 'primary' and 'reserve'.")
    for name, profile in profiles.items():
        validate_profile(name, profile)
    validate_reserve_is_lighter(profiles["primary"], profiles["reserve"])

    remote = config["remote_fallback"]
    if not isinstance(remote, dict):
        raise T05Error("remote_fallback must be an object.")
    _reject_unknown_keys(
        remote, {"required_only_if_local_profiles_fail", "confirmed_by_human", "provider", "note"}, "remote_fallback"
    )
    if not isinstance(remote.get("required_only_if_local_profiles_fail"), bool):
        raise T05Error("remote_fallback.required_only_if_local_profiles_fail must be boolean.")
    if not isinstance(remote.get("confirmed_by_human"), bool):
        raise T05Error("remote_fallback.confirmed_by_human must be boolean.")
    provider = remote.get("provider")
    if remote["confirmed_by_human"] and (not isinstance(provider, str) or not provider.strip()):
        raise T05Error("remote_fallback.provider must be set when confirmed_by_human=true.")
    if provider is not None and not isinstance(provider, str):
        raise T05Error("remote_fallback.provider must be null or string.")
    if not isinstance(remote.get("note"), str):
        raise T05Error("remote_fallback.note must be a string.")


def validate_profile(name: str, profile: dict[str, Any]) -> None:
    if not isinstance(profile, dict):
        raise T05Error(f"profiles.{name} must be an object.")
    _reject_unknown_keys(profile, {"rubert", "whisper"}, f"profiles.{name}")
    rubert = profile.get("rubert")
    whisper = profile.get("whisper")
    if not isinstance(rubert, dict) or not isinstance(whisper, dict):
        raise T05Error(f"profiles.{name} must contain rubert and whisper objects.")

    _reject_unknown_keys(rubert, {"device", "batch_size", "max_length", "mixed_precision"}, f"profiles.{name}.rubert")
    device = rubert.get("device")
    if device not in {"cuda", "cpu"}:
        raise T05Error(f"profiles.{name}.rubert.device must be 'cuda' or 'cpu'.")
    batch_size = rubert.get("batch_size")
    if not _is_int(batch_size) or batch_size < 3:
        raise T05Error(f"profiles.{name}.rubert.batch_size must be an integer >= 3.")
    max_length = rubert.get("max_length")
    if not _is_int(max_length) or not (32 <= max_length <= 512):
        raise T05Error(f"profiles.{name}.rubert.max_length must be in [32, 512] for this smoke check.")
    mp = rubert.get("mixed_precision")
    if mp not in {"none", "fp16", "bf16"}:
        raise T05Error(f"profiles.{name}.rubert.mixed_precision must be none/fp16/bf16.")
    if device == "cpu" and mp == "fp16":
        raise T05Error(f"profiles.{name}: fp16 CPU training is not supported by this check.")

    allowed_whisper = {"model_size", "device", "compute_type", "beam_size", "vad_filter", "cpu_threads"}
    _reject_unknown_keys(whisper, allowed_whisper, f"profiles.{name}.whisper")
    w_device = whisper.get("device")
    if w_device not in {"cuda", "cpu"}:
        raise T05Error(f"profiles.{name}.whisper.device must be 'cuda' or 'cpu'.")
    if not isinstance(whisper.get("model_size"), str) or not whisper["model_size"].strip():
        raise T05Error(f"profiles.{name}.whisper.model_size must be non-empty.")
    compute_type = whisper.get("compute_type")
    if compute_type not in ALL_COMPUTE_TYPES:
        raise T05Error(f"profiles.{name}.whisper.compute_type must be one of {sorted(ALL_COMPUTE_TYPES)}.")
    if w_device == "cpu" and compute_type not in CPU_COMPUTE_TYPES:
        raise T05Error(
            f"profiles.{name}.whisper.compute_type={compute_type!r} is not supported on CPU; "
            f"use one of {sorted(CPU_COMPUTE_TYPES)}."
        )
    beam_size = whisper.get("beam_size")
    if not _is_int(beam_size) or beam_size < 1:
        raise T05Error(f"profiles.{name}.whisper.beam_size must be >= 1.")
    cpu_threads = whisper.get("cpu_threads")
    if not _is_int(cpu_threads) or cpu_threads < 1:
        raise T05Error(f"profiles.{name}.whisper.cpu_threads must be >= 1.")
    if not isinstance(whisper.get("vad_filter"), bool):
        raise T05Error(f"profiles.{name}.whisper.vad_filter must be boolean.")


def validate_reserve_is_lighter(primary: dict[str, Any], reserve: dict[str, Any]) -> None:
    """The reserve profile only makes sense if it asks for strictly less than the primary one."""
    p, r = primary["rubert"], reserve["rubert"]
    if r["batch_size"] > p["batch_size"] or r["max_length"] > p["max_length"]:
        raise T05Error("reserve profile must not exceed the primary batch_size/max_length.")
    p_rank = WHISPER_SIZE_RANK.get(primary["whisper"]["model_size"])
    r_rank = WHISPER_SIZE_RANK.get(reserve["whisper"]["model_size"])
    if p_rank is not None and r_rank is not None and r_rank > p_rank:
        raise T05Error("reserve Whisper model must not be larger than the primary one.")
    lighter = (
        r["batch_size"] < p["batch_size"]
        or r["max_length"] < p["max_length"]
        or (p_rank is not None and r_rank is not None and r_rank < p_rank)
    )
    if not lighter:
        raise T05Error("reserve profile must be strictly lighter than primary in batch, length or Whisper size.")


def seed_everything(seed: int, torch_module: Any | None = None) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch_module is None:
        return
    torch_module.manual_seed(seed)
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    if hasattr(torch_module.backends, "cudnn"):
        torch_module.backends.cudnn.deterministic = True
        torch_module.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Train data: contract, provenance, deterministic smoke batch
# --------------------------------------------------------------------------- #
def _read_train_and_ids(train_path: Path, train_ids_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not train_path.exists():
        raise T05Error(f"Prepared train parquet not found: {train_path}")
    if not train_ids_path.exists():
        raise T05Error(f"T04 train_ids.csv not found: {train_ids_path}")
    train = pd.read_parquet(train_path)
    if "record_id" in train.columns:
        train["record_id"] = train["record_id"].astype(str)
    train_ids = pd.read_csv(train_ids_path, dtype={"record_id": str})
    validate_train_contract(train, train_ids)
    return train, train_ids


def validate_train_contract(train: pd.DataFrame, train_ids: pd.DataFrame) -> None:
    missing = REQUIRED_TRAIN_COLUMNS - set(train.columns)
    if missing:
        raise T05Error(f"Prepared train is missing columns: {sorted(missing)}")
    if train.empty:
        raise T05Error("Prepared train is empty.")
    if train[list(REQUIRED_TRAIN_COLUMNS)].isnull().any().any():
        raise T05Error("Prepared train has nulls in required columns.")
    if not train["review_text"].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise T05Error("Prepared train contains non-string or empty review_text.")
    if set(train["label_id"].unique()) != {0, 1, 2}:
        raise T05Error(f"Prepared train labels must be exactly {{0,1,2}}, got {sorted(train['label_id'].unique())}.")
    expected_names = train["label_id"].map(ID2LABEL).to_numpy(dtype=object)
    actual_names = train["label_name"].to_numpy(dtype=object)
    wrong = expected_names != actual_names
    if wrong.any():
        bad = train.loc[wrong, ["record_id", "label_id", "label_name"]].head(5)
        raise T05Error(f"label_id/label_name mismatch in prepared train, examples: {bad.to_dict('records')}")
    if train["record_id"].duplicated().any():
        raise T05Error("Prepared train record_id values are not unique.")

    missing_ids = {"record_id", "label_id"} - set(train_ids.columns)
    if missing_ids:
        raise T05Error(f"train_ids.csv is missing columns: {sorted(missing_ids)}")
    forbidden_exported = FORBIDDEN_ID_EXPORT_COLUMNS & {str(c).lower() for c in train_ids.columns}
    if forbidden_exported:
        raise T05Error(f"train_ids.csv must not contain text-bearing columns: {sorted(forbidden_exported)}")
    if train_ids.empty:
        raise T05Error("train_ids.csv is empty.")
    if train_ids[["record_id", "label_id"]].isnull().any().any():
        raise T05Error("train_ids.csv has nulls in record_id/label_id.")
    if train_ids["record_id"].duplicated().any():
        raise T05Error("train_ids.csv record_id values are not unique.")
    if not train_ids["record_id"].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
        raise T05Error("train_ids.csv contains empty/non-string record_id.")
    if not set(train_ids["label_id"].unique()).issubset({0, 1, 2}):
        raise T05Error("train_ids.csv contains invalid label_id.")
    if "train_order" in train_ids.columns:
        order = pd.to_numeric(train_ids["train_order"], errors="coerce")
        if order.isnull().any() or not (order % 1 == 0).all() or order.duplicated().any():
            raise T05Error("train_ids.csv train_order must contain unique integers.")

    train_lookup = train.set_index("record_id")["label_id"]
    missing_from_train = [rid for rid in train_ids["record_id"] if rid not in train_lookup.index]
    if missing_from_train:
        raise T05Error(f"train_ids.csv contains IDs absent from prepared train, first: {missing_from_train[:5]}")
    expected = train_ids["record_id"].map(train_lookup).to_numpy(dtype=int)
    if not np.array_equal(expected, train_ids["label_id"].to_numpy(dtype=int)):
        raise T05Error("train_ids.csv label_id does not match prepared train.")
    if set(train_ids["label_id"].unique()) != {0, 1, 2}:
        raise T05Error("Common T04 train IDs must contain all three classes.")


def ids_sha256(train_ids: pd.DataFrame) -> str:
    """Same digest as T04 ``ids_sha256``: ``record_id<TAB>label_id\\n`` in train order."""
    ordered = train_ids
    if "train_order" in train_ids.columns:
        ordered = train_ids.sort_values("train_order", kind="stable")
    digest = hashlib.sha256()
    for rid, label in ordered[["record_id", "label_id"]].itertuples(index=False, name=None):
        digest.update(f"{rid}\t{int(label)}\n".encode("utf-8"))
    return digest.hexdigest()


def verify_train_provenance(
    *, train_path: Path, train_ids: pd.DataFrame, metadata_path: Path, split_ids_path: Path
) -> dict[str, Any]:
    """Prove that the smoke batch comes from the exact T04 train and from no other split.

    * ``train.parquet`` and the ID list must match the SHA-256 values T04 wrote to run_metadata.json;
    * every ID must belong to split ``train`` in the T03 ``split_ids.csv`` (IDs only, no texts;
      rows of other splits are discarded immediately, like in T04).
    """
    if not metadata_path.exists():
        raise T05Error(f"T04 run_metadata.json not found: {metadata_path}")
    if not split_ids_path.exists():
        raise T05Error(f"T03 split_ids.csv not found: {split_ids_path}")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise T05Error(f"Invalid JSON in {metadata_path}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise T05Error("T04 run_metadata.json must be a JSON object.")
    if metadata.get("final_test_used") is not False:
        raise T05Error("T04 run_metadata.json does not state final_test_used=false.")

    expected_parquet = metadata.get("train_parquet_sha256")
    actual_parquet = sha256_file(train_path)
    if not isinstance(expected_parquet, str) or expected_parquet != actual_parquet:
        raise T05Error(
            "train.parquet does not match the file T04 trained on "
            f"(run_metadata train_parquet_sha256={expected_parquet}, actual={actual_parquet}). "
            "Re-run T03/T04 or use the matching files."
        )
    expected_ids = metadata.get("train_ids_sha256")
    actual_ids = ids_sha256(train_ids)
    if not isinstance(expected_ids, str) or expected_ids != actual_ids:
        raise T05Error(
            "train_ids.csv does not match the ID list T04 trained on "
            f"(run_metadata train_ids_sha256={expected_ids}, actual={actual_ids})."
        )

    split_ids = pd.read_csv(
        split_ids_path, dtype=str, keep_default_na=False, usecols=lambda c: c in {"split", "record_id"}
    )
    missing_columns = {"split", "record_id"} - set(split_ids.columns)
    if missing_columns:
        raise T05Error(f"T03 split_ids.csv is missing columns: {sorted(missing_columns)}")
    train_split_ids = set(split_ids.loc[split_ids["split"].str.lower() == "train", "record_id"])
    del split_ids
    foreign = [rid for rid in train_ids["record_id"].astype(str) if rid not in train_split_ids]
    if foreign:
        raise T05Error(
            f"{len(foreign)} train_ids.csv ID(s) are not in the T03 'train' split "
            f"(validation/test/unknown), first: {foreign[:5]}"
        )
    return {
        "train_parquet_sha256_matches_t04": True,
        "train_ids_sha256_matches_t04": True,
        "train_ids_sha256": actual_ids,
        "all_ids_in_t03_train_split": True,
        "t03_train_split_rows": len(train_split_ids),
        "train_ids_rows": int(len(train_ids)),
        "t04_final_test_used": False,
    }


def select_fixed_batch(train: pd.DataFrame, train_ids: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    """Select a deterministic smoke batch from the exact T04 common train IDs.

    The first three rows guarantee one example from each class; the rest follow the persisted T04
    train order. No randomness, no text export.
    """
    if batch_size < 3:
        raise T05Error("batch_size must be >= 3 so the smoke batch can contain all classes.")

    ids = train_ids.copy()
    if "train_order" not in ids.columns:
        ids = ids.reset_index(drop=True)
        ids["train_order"] = range(len(ids))
    ids = ids.sort_values("train_order", kind="stable").reset_index(drop=True)

    selected_ids: list[str] = []
    for label_id in (0, 1, 2):
        candidates = ids.loc[ids["label_id"] == label_id, "record_id"]
        if candidates.empty:
            raise T05Error(f"train_ids.csv has no rows for label {label_id}.")
        selected_ids.append(str(candidates.iloc[0]))

    for rid in ids["record_id"].astype(str):
        if len(selected_ids) >= batch_size:
            break
        if rid not in selected_ids:
            selected_ids.append(rid)

    if len(selected_ids) < batch_size:
        raise T05Error(f"Not enough T04 train IDs for batch_size={batch_size}.")

    selected_ids = selected_ids[:batch_size]
    lookup = train.assign(record_id=train["record_id"].astype(str)).set_index("record_id", drop=False)
    batch = lookup.loc[selected_ids].reset_index(drop=True).copy()
    if len(batch) != batch_size:
        raise T05Error("Internal error: selected batch size mismatch.")
    if not {0, 1, 2}.issubset(set(batch["label_id"].astype(int))):
        raise T05Error("Smoke batch unexpectedly lost one of the three classes.")
    return batch


def _sample_ids_hash(batch_df: pd.DataFrame) -> str:
    payload = "\n".join(f"{rid},{int(label)}" for rid, label in zip(batch_df["record_id"], batch_df["label_id"]))
    return sha256_text(payload)


def write_sample_ids(path: Path, batch_df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "batch_order": range(len(batch_df)),
            "record_id": batch_df["record_id"].astype(str),
            "label_id": batch_df["label_id"].astype(int),
        }
    )
    out.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
def _run_command(args: list[str], timeout: int = 15) -> str | None:
    try:
        result = subprocess.run(
            args, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    text = (result.stdout or result.stderr or "").strip()
    return text or None


def _windows_cpu_name() -> str | None:
    if platform.system() != "Windows":
        return None
    out = _run_command(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
        ]
    )
    return out.strip() if out else None


def _to_number(value: str) -> int | float | str:
    try:
        number = float(value)
    except ValueError:
        return value
    return int(number) if number.is_integer() else number


def _nvidia_smi_snapshot() -> dict[str, Any] | None:
    query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    if not query:
        return None
    parts = [part.strip() for part in query.splitlines()[0].split(",")]
    if len(parts) < 6:
        return {"raw": query}
    return {
        "name": parts[0],
        "driver_version": parts[1],
        "memory_total_mib": _to_number(parts[2]),
        "memory_used_mib": _to_number(parts[3]),
        "memory_free_mib": _to_number(parts[4]),
        "utilization_percent": _to_number(parts[5]),
    }


def _os_edition() -> str | None:
    if platform.system() != "Windows":
        return None
    try:
        return platform.win32_edition()
    except Exception:
        return None


def collect_environment(torch_module: Any, psutil_module: Any, root: Path) -> dict[str, Any]:
    vm = psutil_module.virtual_memory()
    disk = shutil.disk_usage(root)
    cuda_available = bool(torch_module.cuda.is_available())
    gpu = None
    if cuda_available:
        props = torch_module.cuda.get_device_properties(0)
        free_bytes, total_bytes = torch_module.cuda.mem_get_info()
        gpu = {
            "name": torch_module.cuda.get_device_name(0),
            "total_vram_mib": props.total_memory / MIB,
            "device_free_vram_mib_at_start": free_bytes / MIB,
            "device_total_vram_mib_reported_by_driver": total_bytes / MIB,
            "compute_capability": f"{props.major}.{props.minor}",
        }
    return {
        "timestamp_utc": _utc_now(),
        "os": platform.platform(),
        "os_edition": _os_edition(),
        "system": platform.system(),
        "release": platform.release(),
        "python": sys.version.split()[0],
        "python_executable_name": Path(sys.executable).name,
        "cpu": _windows_cpu_name() or platform.processor() or "unknown",
        "logical_cpu_count": psutil_module.cpu_count(logical=True),
        "physical_cpu_count": psutil_module.cpu_count(logical=False),
        "ram_total_mib": vm.total / MIB,
        "ram_note": "RAM as visible to the OS; installed RAM can be larger (firmware/iGPU reservations).",
        "ram_available_mib_at_start": vm.available / MIB,
        "process_rss_mib_at_start": process_rss_mib(psutil_module),
        "repo_disk_free_gib": disk.free / (1024**3),
        "torch": getattr(torch_module, "__version__", None),
        "torch_cuda_runtime": getattr(getattr(torch_module, "version", None), "cuda", None),
        "cuda_available": cuda_available,
        "gpu": gpu,
        "nvidia_smi": _nvidia_smi_snapshot(),
        "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE") == "1",
        "hf_token_present": bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")),
        "versions": {
            "pandas": package_version("pandas"),
            "pyarrow": package_version("pyarrow"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "tokenizers": package_version("tokenizers"),
            "faster-whisper": package_version("faster-whisper"),
            "ctranslate2": package_version("ctranslate2"),
            "av": package_version("av"),
            "psutil": package_version("psutil"),
        },
    }


def cuda_memory_snapshot(torch_module: Any) -> dict[str, Any] | None:
    if not torch_module.cuda.is_available():
        return None
    free_bytes, total_bytes = torch_module.cuda.mem_get_info()
    return {
        "torch_allocated_mib": torch_module.cuda.memory_allocated() / MIB,
        "torch_reserved_mib": torch_module.cuda.memory_reserved() / MIB,
        "torch_peak_allocated_mib": torch_module.cuda.max_memory_allocated() / MIB,
        "torch_peak_reserved_mib": torch_module.cuda.max_memory_reserved() / MIB,
        "device_used_mib": (total_bytes - free_bytes) / MIB,
        "device_total_mib": total_bytes / MIB,
        "nvidia_smi": _nvidia_smi_snapshot(),
    }


# --------------------------------------------------------------------------- #
# Debug audio guards
# --------------------------------------------------------------------------- #
def _probe_wav_duration_seconds(audio_path: Path) -> float | None:
    try:
        with wave.open(str(audio_path), "rb") as handle:
            frames, rate = handle.getnframes(), handle.getframerate()
    except (wave.Error, EOFError, OSError):
        return None
    return frames / rate if rate > 0 else None


def probe_audio_duration_seconds(audio_path: Path) -> float | None:
    """Duration without decoding the whole audio. ``None`` means it could not be determined."""
    if audio_path.suffix.lower() == ".wav":
        duration = _probe_wav_duration_seconds(audio_path)
        if duration is not None:
            return duration
    try:
        import av
    except ImportError:
        return None
    try:
        with av.open(str(audio_path)) as container:
            if container.duration is not None:
                # PyAV container duration is expressed in AV_TIME_BASE units.
                return float(container.duration) / 1_000_000.0
            for stream in container.streams.audio:
                if stream.duration is not None and stream.time_base is not None:
                    return float(stream.duration * stream.time_base)
    except Exception:
        return None
    return None


def looks_like_final_recording(audio_path: Path) -> bool:
    """Heuristic guard (defence in depth): file name or its folder mentions the final set."""
    names = (audio_path.stem.lower(), audio_path.parent.name.lower())
    return any(marker in name for name in names for marker in FINAL_AUDIO_NAME_MARKERS)


def validate_debug_audio(audio_path: Path, config: dict[str, Any]) -> dict[str, Any]:
    if not audio_path.exists() or not audio_path.is_file():
        raise T05Error(f"Debug audio not found: {audio_path}")
    if looks_like_final_recording(audio_path):
        raise T05Error(
            "The audio file or its folder looks like the final recording set ('final'/'финал' in the name). "
            "T05 must use a separate debug recording: rename or move the file if it really is one."
        )
    limits = config["debug_audio"]
    size = int(audio_path.stat().st_size)
    if size <= 0:
        raise T05Error("Debug audio file is empty.")
    if size > int(limits["max_bytes"]):
        raise T05Error(f"Debug audio is too large: {size} bytes > configured {int(limits['max_bytes'])} bytes.")
    duration = probe_audio_duration_seconds(audio_path)
    if duration is None:
        raise T05Error(
            "Cannot determine the debug audio duration (PyAV missing or file unreadable); refusing to run "
            f"without verifying the {float(limits['max_seconds']):g}s limit."
        )
    if duration <= 0:
        raise T05Error("Debug audio duration is zero or invalid.")
    if duration > float(limits["max_seconds"]):
        raise T05Error(f"Debug audio is too long: {duration:.2f}s > configured {float(limits['max_seconds']):.2f}s.")
    return {
        "audio_bytes": size,
        "audio_duration_seconds": duration,
        "audio_sha256": sha256_file(audio_path),
    }


# --------------------------------------------------------------------------- #
# Git safety
# --------------------------------------------------------------------------- #
def _run_git(root: Path, *args: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=root, capture_output=True, text=True, check=False, timeout=15
        )
    except (OSError, subprocess.TimeoutExpired):
        return None


def git_path_ignored(root: Path, path: Path) -> bool | None:
    """True: cannot be committed (ignored or outside the repository); False: could be committed;
    None: Git unavailable / not a repository."""
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return True
    proc = _run_git(root, "check-ignore", "-q", rel.as_posix())
    if proc is None:
        return None
    if proc.returncode == 0:
        return True
    if proc.returncode == 1:
        return False
    return None


def git_safety_status(root: Path, path: Path, label: str) -> str:
    """Refuse to continue if a file with private content could be committed."""
    ignored = git_path_ignored(root, path)
    if ignored is False:
        raise T05Error(
            f"{label} is not ignored by Git ({path.name}). Add it to .gitignore first "
            "(for example 'artifacts/' and audio extensions) and check with: git check-ignore -v <path>."
        )
    if ignored is None:
        return "git_unavailable_not_checked"
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return "outside_repository"
    return "git_ignored"


# --------------------------------------------------------------------------- #
# RuBERT: parameter-update evidence (pure numpy, independent of torch)
# --------------------------------------------------------------------------- #
def is_head_parameter(name: str) -> bool:
    return name.startswith(HEAD_PARAMETER_PREFIX) or ".classifier." in name


def select_sentinel_parameters(ndims: Mapping[str, int]) -> dict[str, list[str]]:
    """Pick parameters whose change proves that training reached the transformer AND the head.

    Encoder sentinels: a 2-D weight in the FIRST and in the LAST ``encoder.layer.N``. Embeddings alone
    are not enough: they change even if every transformer layer is frozen. Head sentinels: all
    ``classifier.*`` parameters.
    """
    head = sorted(name for name in ndims if is_head_parameter(name))
    layers: dict[int, list[str]] = {}
    for name, ndim in ndims.items():
        match = _ENCODER_LAYER_RE.search(name)
        if match and ndim >= 2 and not is_head_parameter(name):
            layers.setdefault(int(match.group(1)), []).append(name)
    encoder: list[str] = []
    if layers:
        first, last = min(layers), max(layers)
        encoder.append(sorted(layers[first])[0])
        if last != first:
            encoder.append(sorted(layers[last])[0])
    if not encoder:
        raise T05Error("Could not locate 2-D transformer-layer parameters (encoder.layer.N.*) in the RuBERT model.")
    if not head:
        raise T05Error("Could not locate classifier-head parameters (classifier.*) in the RuBERT model.")
    return {"encoder": encoder, "head": head}


def take_parameter_snapshot(named_params: Sequence[tuple[str, Any]]) -> dict[str, np.ndarray]:
    """Copy small/medium trainable tensors to host memory; huge tensors are skipped by design."""
    snapshot: dict[str, np.ndarray] = {}
    for name, param in named_params:
        if param.numel() <= SNAPSHOT_MAX_ELEMENTS:
            snapshot[name] = np.array(param.detach().float().cpu().numpy(), copy=True)
    return snapshot


def evaluate_parameter_updates(
    before: Mapping[str, np.ndarray],
    after: Mapping[str, np.ndarray],
    grad_norms: Mapping[str, float],
    sentinels: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """Require real gradients and real parameter changes for encoder and head sentinels."""
    problems: list[str] = []
    max_update: dict[str, float] = {}
    for name in set(before) & set(after):
        max_update[name] = float(np.max(np.abs(np.asarray(after[name]) - np.asarray(before[name]))))
    for group in ("encoder", "head"):
        for name in sentinels[group]:
            norm = grad_norms.get(name)
            if norm is None or not math.isfinite(norm) or norm <= 0.0:
                problems.append(f"{group} parameter {name}: gradient is missing, zero or non-finite ({norm})")
            if name not in max_update:
                problems.append(f"{group} parameter {name}: no before/after snapshot")
            elif not max_update[name] > 0.0:
                problems.append(f"{group} parameter {name}: value did not change after optimizer.step()")
    if problems:
        raise T05Error("RuBERT training step did not update encoder and head:\n- " + "\n- ".join(problems))

    def group_stats(names: list[str]) -> tuple[int, int]:
        present = [n for n in names if n in max_update]
        return len(present), sum(1 for n in present if max_update[n] > 0.0)

    encoder_names = [n for n in before if not is_head_parameter(n)]
    head_names = [n for n in before if is_head_parameter(n)]
    enc_total, enc_changed = group_stats(encoder_names)
    head_total, head_changed = group_stats(head_names)
    return {
        "encoder_updated": True,
        "head_updated": True,
        "encoder_sentinel_parameters": list(sentinels["encoder"]),
        "head_sentinel_parameters": list(sentinels["head"]),
        "encoder_max_abs_update": max(max_update[n] for n in sentinels["encoder"]),
        "head_max_abs_update": max(max_update[n] for n in sentinels["head"]),
        "sentinel_grad_norms": {n: float(grad_norms[n]) for g in ("encoder", "head") for n in sentinels[g]},
        "encoder_tensors_snapshotted": enc_total,
        "encoder_tensors_changed": enc_changed,
        "head_tensors_snapshotted": head_total,
        "head_tensors_changed": head_changed,
        "snapshot_max_elements_per_tensor": SNAPSHOT_MAX_ELEMENTS,
    }


def verify_module_device(module: Any, expected_device: str, label: str = "model") -> None:
    found = {p.device.type for p in module.parameters()}
    if found != {expected_device}:
        raise T05Error(f"{label} parameters are on {sorted(found)}, expected only '{expected_device}'.")


def collect_grad_norms(named_params: Mapping[str, Any], names: Iterable[str]) -> dict[str, float]:
    norms: dict[str, float] = {}
    for name in names:
        grad = named_params[name].grad
        norms[name] = 0.0 if grad is None else float(grad.detach().float().norm().item())
    return norms


def summarize_loading_info(loading_info: Any) -> dict[str, Any]:
    """Explain the Hugging Face load report. A new 3-class head is expected to be MISSING from the
    checkpoint and the original MLM head (``cls.*``) to be UNEXPECTED."""

    def keys(name: str) -> list[str]:
        raw = loading_info.get(name) if isinstance(loading_info, Mapping) else getattr(loading_info, name, None)
        return sorted(str(k) for k in (raw or []))

    if loading_info is None:
        return {"available": False}
    missing, unexpected = keys("missing_keys"), keys("unexpected_keys")
    return {
        "available": True,
        "missing_keys": missing[:20],
        "missing_keys_count": len(missing),
        "unexpected_keys_count": len(unexpected),
        "unexpected_keys_sample": unexpected[:5],
        "mismatched_keys_count": len(keys("mismatched_keys")),
        "only_new_head_missing": set(missing) <= EXPECTED_NEW_HEAD_KEYS,
    }


def load_rubert_model(rubert_cfg: dict[str, Any], device: str, torch_module: Any) -> tuple[Any, Any, dict[str, Any]]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as exc:
        raise T05Error("RuBERT dependencies are missing. Install transformers first.") from exc

    name = rubert_cfg["model_name"]
    start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer_seconds = time.perf_counter() - start

    kwargs = {"num_labels": 3, "id2label": ID2LABEL, "label2id": LABELS}
    start = time.perf_counter()
    loading_info = None
    try:
        result = AutoModelForSequenceClassification.from_pretrained(name, output_loading_info=True, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            model, loading_info = result
        else:
            model = result
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(name, **kwargs)
    model.to(device)
    _synchronize(torch_module, device)
    model_seconds = time.perf_counter() - start

    if int(model.config.num_labels) != 3:
        raise T05Error(f"Loaded RuBERT classifier has num_labels={model.config.num_labels}, expected 3.")
    if {int(k): v for k, v in model.config.id2label.items()} != ID2LABEL:
        raise T05Error(f"RuBERT id2label {dict(model.config.id2label)} does not match {ID2LABEL}.")
    return tokenizer, model, {
        "tokenizer_load_seconds": tokenizer_seconds,
        "model_load_seconds": model_seconds,
        "model_commit_hash": getattr(getattr(model, "config", None), "_commit_hash", None),
        "tokenizer_commit_hash": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "loading_info": summarize_loading_info(loading_info),
        "id2label": {str(k): v for k, v in sorted(ID2LABEL.items())},
    }


def run_training_steps(
    *,
    model: Any,
    tokenizer: Any,
    batch_df: pd.DataFrame,
    profile_rubert: dict[str, Any],
    rubert_cfg: dict[str, Any],
    torch_module: Any,
    psutil_module: Any,
) -> dict[str, Any]:
    """Real forward -> loss -> backward -> optimizer.step on a batch padded to ``max_length``."""
    torch = torch_module
    device = profile_rubert["device"]
    max_length = int(profile_rubert["max_length"])
    steps = int(rubert_cfg["training_steps"])
    mixed_precision = profile_rubert["mixed_precision"]

    verify_module_device(model, device, "RuBERT model")
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not trainable:
        raise T05Error("RuBERT model has no trainable parameters.")
    param_by_name = dict(trainable)
    sentinels = select_sentinel_parameters({n: p.ndim for n, p in trainable})
    sentinel_names = [n for group in ("encoder", "head") for n in sentinels[group]]

    texts = batch_df["review_text"].tolist()
    labels = batch_df["label_id"].astype(int).tolist()
    start = time.perf_counter()
    encoded = tokenizer(
        texts, padding="max_length", truncation=True, max_length=max_length, return_tensors="pt"
    )
    encoded = {key: value for key, value in encoded.items()}
    if tuple(encoded["input_ids"].shape) != (len(texts), max_length):
        raise T05Error(
            f"Resource smoke batch has shape {tuple(encoded['input_ids'].shape)}, expected "
            f"({len(texts)}, {max_length}): padding to max_length did not happen."
        )
    attention = encoded.get("attention_mask")
    non_pad_tokens = attention.sum(dim=1).tolist() if attention is not None else []
    encoded["labels"] = torch.tensor(labels, dtype=torch.long)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    _synchronize(torch, device)
    tokenization_seconds = time.perf_counter() - start
    for key, value in encoded.items():
        if value.device.type != device:
            raise T05Error(f"Input tensor {key} is on {value.device.type}, expected {device}.")

    optimizer = torch.optim.AdamW(
        [p for _, p in trainable], lr=float(rubert_cfg["learning_rate"]), weight_decay=float(rubert_cfg["weight_decay"])
    )
    optimizer_param_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    not_in_optimizer = [n for n in sentinel_names if id(param_by_name[n]) not in optimizer_param_ids]
    if not_in_optimizer:
        raise T05Error(f"Optimizer does not contain parameters: {not_in_optimizer}")

    use_amp = device == "cuda" and mixed_precision in {"fp16", "bf16"}
    amp_dtype = torch.float16 if mixed_precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and mixed_precision == "fp16"))

    before = take_parameter_snapshot(trainable)
    model.train()
    losses: list[float] = []
    step_seconds: list[float] = []
    skipped_by_scaler = 0
    last_valid_grad_norms: dict[str, float] | None = None
    cuda_peak_after_backward_mib: float | None = None

    for step_index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        scale_before = scaler.get_scale() if scaler.is_enabled() else None
        _synchronize(torch, device)  # nothing queued before the timer starts
        start = time.perf_counter()
        with torch.amp.autocast(device_type=device, dtype=amp_dtype if use_amp else None, enabled=use_amp):
            outputs = model(**encoded)
            loss = outputs.loss
        if step_index == 0 and loss.device.type != device:
            raise T05Error(f"Loss was computed on {loss.device.type}, expected {device}.")
        if scaler.is_enabled():
            scaler.scale(loss).backward()
        else:
            loss.backward()
        if step_index == 0 and device == "cuda":
            # Before optimizer state exists: the peak here is driven by activations, i.e. by max_length.
            cuda_peak_after_backward_mib = torch.cuda.max_memory_allocated() / MIB
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        _synchronize(torch, device)
        step_seconds.append(time.perf_counter() - start)

        loss_value = float(loss.detach().float().cpu().item())
        if not math.isfinite(loss_value):
            raise T05Error(f"Non-finite RuBERT loss: {loss_value}")
        losses.append(loss_value)
        skipped = scaler.is_enabled() and scaler.get_scale() < scale_before
        if skipped:  # fp16 overflow: GradScaler legitimately skips the step and lowers its scale
            skipped_by_scaler += 1
        else:
            # Gradients are still attached after step() and are already unscaled.
            last_valid_grad_norms = collect_grad_norms(param_by_name, sentinel_names)

    if last_valid_grad_norms is None:
        raise T05Error("Every optimizer step was skipped by GradScaler (inf/NaN gradients); no update happened.")
    after = take_parameter_snapshot(trainable)
    evidence = evaluate_parameter_updates(before, after, last_valid_grad_norms, sentinels)

    cuda_at_peak = cuda_memory_snapshot(torch) if device == "cuda" else None
    model.eval()
    model.zero_grad(set_to_none=True)  # gradients would otherwise stay in VRAM for the joint stage
    outputs = loss = optimizer = scaler = encoded = None
    if device == "cuda":
        torch.cuda.empty_cache()
    cuda_after_release = cuda_memory_snapshot(torch) if device == "cuda" else None

    return {
        "batch_size": int(profile_rubert["batch_size"]),
        "max_length": max_length,
        "mixed_precision": mixed_precision,
        "training_steps": steps,
        "tokenization_seconds": tokenization_seconds,
        "encoded_sequence_length": max_length,
        "non_pad_tokens_min": int(min(non_pad_tokens)) if non_pad_tokens else None,
        "non_pad_tokens_max": int(max(non_pad_tokens)) if non_pad_tokens else None,
        "non_pad_tokens_mean": (sum(non_pad_tokens) / len(non_pad_tokens)) if non_pad_tokens else None,
        "resource_stress_padding": "max_length",
        "model_parameters_on_device": device,
        "inputs_on_device": device,
        "optimizer_contains_sentinel_parameters": True,
        "step_seconds": step_seconds,
        "step_seconds_first_cold": step_seconds[0],
        "step_seconds_steady_mean": (sum(step_seconds[1:]) / len(step_seconds[1:])) if len(step_seconds) > 1 else None,
        "optimizer_steps_skipped_by_grad_scaler": skipped_by_scaler,
        "losses": losses,
        **evidence,
        "cuda_peak_allocated_after_backward_mib": cuda_peak_after_backward_mib,
        "cuda_memory_at_peak_window": cuda_at_peak,
        "cuda_memory_after_release": cuda_after_release,
    }


def run_rubert_training_step(
    *, config: dict[str, Any], profile_name: str, batch_df: pd.DataFrame, psutil_module: Any
) -> tuple[RuntimeObjects, dict[str, Any]]:
    try:
        import torch
    except ImportError as exc:
        raise T05Error("PyTorch is missing. Install a working (CUDA) PyTorch build first.") from exc

    seed_everything(int(config["seed"]), torch)
    profile_rubert = config["profiles"][profile_name]["rubert"]
    rubert_cfg = config["rubert"]
    device = profile_rubert["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise T05Error(
            "RuBERT profile requires CUDA, but torch.cuda.is_available() is False. "
            "Install/use CUDA-enabled PyTorch or run a CPU profile."
        )
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    before_rss = process_rss_mib(psutil_module)
    monitor = PeakRSSMonitor(psutil_module).start()
    try:
        tokenizer, model, load_metrics = load_rubert_model(rubert_cfg, device, torch)
        step_metrics = run_training_steps(
            model=model,
            tokenizer=tokenizer,
            batch_df=batch_df,
            profile_rubert=profile_rubert,
            rubert_cfg=rubert_cfg,
            torch_module=torch,
            psutil_module=psutil_module,
        )
    finally:
        rss_peak_mib = monitor.stop()

    metrics = {
        "status": "PASS",
        "model_name": rubert_cfg["model_name"],
        "device": device,
        "learning_rate": float(rubert_cfg["learning_rate"]),
        "weight_decay": float(rubert_cfg["weight_decay"]),
        **load_metrics,
        **step_metrics,
        "process_rss_before_mib": before_rss,
        "process_rss_after_mib": process_rss_mib(psutil_module),
        "process_rss_peak_sampled_mib": rss_peak_mib,
        "cuda_device_used_mib_after_step": (step_metrics["cuda_memory_at_peak_window"] or {}).get("device_used_mib"),
    }
    return RuntimeObjects(torch=torch, tokenizer=tokenizer, model=model), metrics


# --------------------------------------------------------------------------- #
# Whisper
# --------------------------------------------------------------------------- #
def describe_whisper_backend(model: Any, expected_device: str) -> dict[str, Any]:
    """Ask CTranslate2 where the model really lives; refuse a silent CUDA/CPU mix-up."""
    inner = getattr(model, "model", None)
    reported = getattr(inner, "device", None)
    reported_device = str(reported).lower() if reported is not None else None
    if reported_device is not None and expected_device not in reported_device:
        raise T05Error(f"Whisper backend reports device {reported_device!r}, expected {expected_device!r}.")
    compute = getattr(inner, "compute_type", None)
    return {
        "backend_reported_device": reported_device,
        "backend_reported_compute_type": str(compute) if compute is not None else None,
    }


def transcribe_fully(
    model: Any, audio_path: Path, *, language: str, beam_size: int, vad_filter: bool
) -> tuple[list[Any], Any, float]:
    """Run recognition and time it INCLUDING the iteration over segments.

    faster-whisper returns a lazy generator: the real decoding happens while it is consumed, so a
    timer that stops after ``transcribe()`` returns would measure almost nothing.
    """
    start = time.perf_counter()
    segments_iter, info = model.transcribe(
        str(audio_path), language=language, beam_size=int(beam_size), vad_filter=bool(vad_filter)
    )
    segments = list(segments_iter)
    return segments, info, time.perf_counter() - start


def join_segments_text(segments: Sequence[Any]) -> str:
    return " ".join(s.text.strip() for s in segments if s.text.strip()).strip()


def run_whisper_transcription(
    *,
    profile: dict[str, Any],
    audio_path: Path,
    local_artifact_path: Path,
    psutil_module: Any,
    language: str = "ru",
) -> tuple[Any, dict[str, Any], str]:
    """Load Whisper and recognise the debug audio. The transcript text is returned separately (for
    the leak check) and is written only to the local, Git-ignored artifact."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise T05Error("faster-whisper is not installed. Run: python -m pip install faster-whisper") from exc

    wcfg = profile["whisper"]
    before_rss = process_rss_mib(psutil_module)
    monitor = PeakRSSMonitor(psutil_module).start()
    before_nvidia = _nvidia_smi_snapshot()
    try:
        load_start = time.perf_counter()
        try:
            model = WhisperModel(
                wcfg["model_size"],
                device=wcfg["device"],
                compute_type=wcfg["compute_type"],
                cpu_threads=int(wcfg["cpu_threads"]),
            )
        except Exception as exc:  # CTranslate2 can raise RuntimeError/OSError depending on backend
            raise T05Error(
                "Whisper model failed to load. If you intentionally use device='cuda' on Windows, "
                "faster-whisper/CTranslate2 requires its own CUDA 12 cuBLAS + cuDNN 9 runtime; "
                "PyTorch CUDA alone is not sufficient. The default T05 profiles use CPU int8. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        load_seconds = time.perf_counter() - load_start
        backend = describe_whisper_backend(model, wcfg["device"])
        try:
            segments, info, transcribe_seconds = transcribe_fully(
                model, audio_path, language=language, beam_size=wcfg["beam_size"], vad_filter=wcfg["vad_filter"]
            )
        except Exception as exc:
            raise T05Error(f"Whisper transcription failed: {type(exc).__name__}: {exc}") from exc
    finally:
        rss_peak_mib = monitor.stop()

    text = join_segments_text(segments)
    if not text:
        raise T05Error("Whisper returned an empty transcription. Use a clear debug Russian speech recording.")

    audio_sha = sha256_file(audio_path)
    local_payload = {
        "task": "T05",
        "created_at_utc": _utc_now(),
        "audio_file_name": audio_path.name,
        "audio_sha256": audio_sha,
        "model_size": wcfg["model_size"],
        "device": wcfg["device"],
        "compute_type": wcfg["compute_type"],
        "requested_language": language,
        "language": getattr(info, "language", language),
        "text": text,
        "segments": [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segments],
    }
    write_text_lf(local_artifact_path, dumps_json(local_payload))

    file_duration = probe_audio_duration_seconds(audio_path)
    metrics = {
        "status": "PASS",
        "requested_language": language,
        "language_forced": True,
        "model_size": wcfg["model_size"],
        "device": wcfg["device"],
        "compute_type": wcfg["compute_type"],
        "beam_size": int(wcfg["beam_size"]),
        "vad_filter": bool(wcfg["vad_filter"]),
        "cpu_threads": int(wcfg["cpu_threads"]),
        **backend,
        "model_load_seconds": load_seconds,
        "model_load_note": "includes a one-off download if the model was not cached yet",
        "transcription_seconds": transcribe_seconds,
        "transcription_measured_with_all_segments_consumed": True,
        "audio_sha256": audio_sha,
        "audio_bytes": audio_path.stat().st_size,
        "audio_duration_seconds": file_duration,
        "recognized_duration_seconds": max((float(s.end) for s in segments), default=0.0),
        "real_time_factor": (transcribe_seconds / file_duration) if file_duration and file_duration > 0 else None,
        "segment_count": len(segments),
        "transcript_chars": len(text),
        "transcript_sha256": sha256_text(text),
        "language": getattr(info, "language", language),
        "process_rss_before_mib": before_rss,
        "process_rss_after_mib": process_rss_mib(psutil_module),
        "process_rss_peak_sampled_mib": rss_peak_mib,
        "nvidia_smi_before": before_nvidia,
        "nvidia_smi_after": _nvidia_smi_snapshot(),
        "transcript_in_git_report": False,
    }
    return model, metrics, text


# --------------------------------------------------------------------------- #
# Joint load
# --------------------------------------------------------------------------- #
def run_joint_core(
    *,
    rubert_forward: Callable[[], Any],
    whisper_pass: Callable[[], dict[str, Any]],
    psutil_module: Any,
    sync: Callable[[], None] = lambda: None,
) -> dict[str, Any]:
    """RuBERT forward -> Whisper recognition -> RuBERT forward, while both models stay loaded.

    The second RuBERT output must equal the first: it proves RuBERT is still intact and usable
    after Whisper has run next to it.
    """
    order: list[str] = []
    monitor = PeakRSSMonitor(psutil_module).start()
    try:
        start = time.perf_counter()
        logits_first = np.asarray(rubert_forward(), dtype=float)
        sync()
        rubert_first_seconds = time.perf_counter() - start
        order.append("rubert_forward")

        whisper = whisper_pass()
        order.append("whisper_transcribe")

        start = time.perf_counter()
        logits_second = np.asarray(rubert_forward(), dtype=float)
        sync()
        rubert_second_seconds = time.perf_counter() - start
        order.append("rubert_forward")
    finally:
        peak = monitor.stop()

    if not (np.isfinite(logits_first).all() and np.isfinite(logits_second).all()):
        raise T05Error("Joint-load RuBERT inference produced non-finite logits.")
    if logits_first.shape[-1] != 3:
        raise T05Error(f"Joint-load RuBERT logits have {logits_first.shape[-1]} classes, expected 3.")
    if not np.allclose(logits_first, logits_second, rtol=1e-3, atol=1e-3):
        raise T05Error("RuBERT output changed after Whisper ran next to it; the loaded models interfere.")
    pred = int(np.argmax(logits_first.reshape(-1, logits_first.shape[-1])[0]))
    return {
        "status": "PASS",
        "operation_order": order,
        "rubert_forward_seconds_before_whisper": rubert_first_seconds,
        "rubert_forward_seconds_after_whisper": rubert_second_seconds,
        "rubert_logits_identical_before_after_whisper": True,
        "predicted_label_id": pred,
        "predicted_label": ID2LABEL[pred],
        "whisper_while_rubert_loaded": whisper,
        "process_rss_peak_sampled_mib": peak,
    }


def verify_joint_operation(
    *,
    runtime: RuntimeObjects,
    profile: dict[str, Any],
    batch_df: pd.DataFrame,
    audio_path: Path,
    language: str,
    cold_transcript_sha256: str,
    psutil_module: Any,
) -> dict[str, Any]:
    torch = runtime.torch
    device = profile["rubert"]["device"]
    wcfg = profile["whisper"]
    if runtime.whisper_model is None or runtime.model is None:
        raise T05Error("Joint-load check requires RuBERT and Whisper to be loaded at the same time.")
    verify_module_device(runtime.model, device, "RuBERT model (joint stage)")
    backend = describe_whisper_backend(runtime.whisper_model, wcfg["device"])
    max_length = int(profile["rubert"]["max_length"])
    text = str(batch_df.iloc[0]["review_text"])
    cuda_before = cuda_memory_snapshot(torch) if device == "cuda" else None
    if device == "cuda" and not (cuda_before and cuda_before["torch_allocated_mib"] > 0):
        raise T05Error("RuBERT weights are not resident in CUDA memory at the joint-load stage.")

    def rubert_forward() -> Any:
        runtime.model.eval()
        encoded = runtime.tokenizer([text], padding="max_length", truncation=True, max_length=max_length, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}
        if int(encoded["input_ids"].shape[1]) != max_length:
            raise T05Error("Joint-load input was not padded to max_length.")
        with torch.no_grad():
            logits = runtime.model(**encoded).logits
        return logits.detach().float().cpu().numpy()

    def whisper_pass() -> dict[str, Any]:
        segments, _info, seconds = transcribe_fully(
            runtime.whisper_model, audio_path, language=language, beam_size=wcfg["beam_size"], vad_filter=wcfg["vad_filter"]
        )
        transcript = join_segments_text(segments)
        return {
            "transcription_seconds_warm": seconds,
            "segments_consumed": len(segments),
            "transcript_chars": len(transcript),
            "transcript_sha256": sha256_text(transcript),
            "transcript_matches_cold_run": sha256_text(transcript) == cold_transcript_sha256,
        }

    core = run_joint_core(
        rubert_forward=rubert_forward,
        whisper_pass=whisper_pass,
        psutil_module=psutil_module,
        sync=lambda: _synchronize(torch, device),
    )
    return {
        **core,
        "rubert_and_whisper_loaded_together": True,
        "rubert_parameters_on_device": device,
        "rubert_parameter_count": int(sum(p.numel() for p in runtime.model.parameters())),
        "whisper_backend": backend,
        "encoded_sequence_length": max_length,
        "process_rss_mib": process_rss_mib(psutil_module),
        "cuda_memory": cuda_memory_snapshot(torch) if device == "cuda" else None,
    }


# --------------------------------------------------------------------------- #
# Findings, memory summary, reports
# --------------------------------------------------------------------------- #
def build_memory_summary(*, environment: dict[str, Any], tracker: PeakRSSMonitor, psutil_module: Any) -> dict[str, Any]:
    return {
        "process_peak_rss_mib_os": process_peak_rss_mib(psutil_module),
        "process_peak_rss_mib_sampled": tracker.peak_mib,
        "process_rss_mib_at_start": environment.get("process_rss_mib_at_start"),
        "process_rss_mib_at_end": process_rss_mib(psutil_module),
        "system_total_mib": environment.get("ram_total_mib"),
        "system_available_mib_at_start": environment.get("ram_available_mib_at_start"),
        "system_available_mib_min_sampled": tracker.min_available_mib,
        "definition": (
            "process_peak_rss_mib_os is the OS-reported peak working set of this Python process for the whole "
            "run (imports, both models, training step); *_sampled values come from a 50 ms sampler and can "
            "miss short spikes."
        ),
    }


def external_backup_status(config: dict[str, Any]) -> dict[str, Any]:
    remote = config["remote_fallback"]
    confirmed = bool(remote["confirmed_by_human"])
    return {
        "status": "CONFIRMED_BY_HUMAN_NOT_CHECKED_BY_SCRIPT" if confirmed else "NOT_VERIFIED_NOT_CLAIMED",
        "provider": remote.get("provider"),
        "note": remote.get("note"),
        "explanation": (
            "reserve profile = a lighter LOCAL configuration; external/remote compute is a different thing "
            "and is not tested by this script."
        ),
    }


def build_resource_findings(
    *,
    environment: dict[str, Any],
    memory: dict[str, Any],
    rubert: dict[str, Any],
    whisper: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []

    def add(level: str, code: str, message: str) -> None:
        findings.append({"level": level, "code": code, "message": message})

    min_available = memory.get("system_available_mib_min_sampled")
    if min_available is not None:
        level = "WARNING" if min_available < MIN_RAM_HEADROOM_MIB else "OK"
        add(
            level,
            "ram_headroom",
            f"Минимум свободной системной RAM за прогон: {min_available:.0f} MiB (порог {MIN_RAM_HEADROOM_MIB:.0f} MiB); "
            f"свободно на старте: {memory.get('system_available_mib_at_start') or 0:.0f} MiB из "
            f"{memory.get('system_total_mib') or 0:.0f} MiB, видимых ОС. Другие открытые программы уменьшают запас; "
            "перед демонстрацией нужно закрыть лишнее.",
        )
    total_vram = (environment.get("gpu") or {}).get("total_vram_mib")
    used_vram = rubert.get("cuda_device_used_mib_after_step")
    if total_vram and used_vram is not None:
        headroom = total_vram - used_vram
        add(
            "WARNING" if headroom < MIN_VRAM_HEADROOM_MIB else "OK",
            "vram_headroom",
            f"Свободно VRAM на устройстве сразу после шага обучения: {headroom:.0f} MiB из {total_vram:.0f} MiB "
            f"(занято всеми процессами {used_vram:.0f} MiB; порог {MIN_VRAM_HEADROOM_MIB:.0f} MiB).",
        )
    rtf = whisper.get("real_time_factor")
    if rtf is not None and rtf > 1.0:
        add("WARNING", "whisper_slower_than_realtime", f"Whisper медленнее реального времени: RTF={rtf:.2f} (первый вызов).")
    add(
        "INFO",
        "first_call_timings",
        "Время первого шага RuBERT и первого распознавания включает прогрев (CUDA/ядра/кэши); "
        "устоявшийся шаг и повторное распознавание указаны отдельно. Время загрузки Whisper при первом "
        "запуске может включать скачивание модели.",
    )
    info = rubert.get("loading_info") or {}
    if info.get("available") and not info.get("only_new_head_missing", True):
        add(
            "WARNING",
            "rubert_missing_weights",
            f"Из чекпойнта не загружены не только classifier.*: {info.get('missing_keys')}. Проверьте загрузку модели.",
        )
    elif info.get("available"):
        add(
            "INFO",
            "rubert_head_initialised_from_scratch",
            "Ожидаемо: classifier.weight/bias созданы заново (в чекпойнте rubert-tiny2 нет 3-классовой головы), "
            "cls.* исходного чекпойнта не используются. Это не ошибка.",
        )
    if rubert.get("mixed_precision") != "none":
        add(
            "INFO",
            "mixed_precision",
            f"Замеры сделаны с mixed_precision={rubert.get('mixed_precision')}; при обучении в другой точности "
            "расход VRAM нужно измерить заново.",
        )
    backup = external_backup_status(config)
    add("INFO", "external_backup_compute", f"Внешний резервный доступ к вычислениям: {backup['status']} (скриптом не проверяется).")
    return findings


def _walk_report_for_forbidden_keys(obj: Any, path: tuple[str, ...] = ()) -> None:
    if isinstance(obj, dict):
        for key, value in obj.items():
            if str(key).lower() in FORBIDDEN_REPORT_KEYS:
                raise T05Error(f"Git-safe report unexpectedly contains text field: {'.'.join(path + (str(key),))}")
            _walk_report_for_forbidden_keys(value, path + (str(key),))
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            _walk_report_for_forbidden_keys(value, path + (str(i),))


def sanitize_report_for_git(report: dict[str, Any]) -> dict[str, Any]:
    """Defensive check by key name (values are checked by ``assert_no_text_leak``)."""
    _walk_report_for_forbidden_keys(report)
    return report


def _leak_probes(text: str) -> list[str]:
    """Fragments of ``text`` whose presence in a report means the text (or a part of it) leaked.

    Short texts are checked whole; long ones through overlapping windows, so a leak of a single
    segment or of a quoted fragment (>= ``LEAK_WINDOW_CHARS + LEAK_WINDOW_STEP`` characters) is caught too.
    """
    needle = text.strip()
    if len(needle) < MIN_TEXT_LEAK_CHARS:
        return []
    if len(needle) <= LEAK_WINDOW_CHARS:
        return [needle]
    starts = list(range(0, len(needle) - LEAK_WINDOW_CHARS + 1, LEAK_WINDOW_STEP))
    if starts[-1] != len(needle) - LEAK_WINDOW_CHARS:
        starts.append(len(needle) - LEAK_WINDOW_CHARS)
    return [needle[start : start + LEAK_WINDOW_CHARS] for start in starts]


def assert_no_text_leak(outputs: Mapping[str, str], forbidden_texts: Iterable[str]) -> None:
    """Fail if a transcript or a review text (even a fragment of it) appears in any Git-safe output,
    under any key name and in raw or JSON-escaped form."""
    for text in forbidden_texts:
        for probe in _leak_probes(text):
            variants = {probe, json.dumps(probe, ensure_ascii=True)[1:-1], json.dumps(probe, ensure_ascii=False)[1:-1]}
            for name, content in outputs.items():
                if any(variant in content for variant in variants):
                    raise T05Error(f"Text content leaked into Git-safe output {name}; nothing was written.")


def build_report(
    *,
    config: dict[str, Any],
    profile_name: str,
    environment: dict[str, Any],
    train_path: Path,
    train_ids_path: Path,
    provenance: dict[str, Any],
    batch_df: pd.DataFrame,
    audio_meta: dict[str, Any],
    audio_location: str,
    artifacts_location: str,
    rubert_metrics: dict[str, Any],
    whisper_metrics: dict[str, Any],
    joint_metrics: dict[str, Any],
    memory: dict[str, Any],
    findings: list[dict[str, str]],
) -> dict[str, Any]:
    # review_text and transcript text are deliberately absent from this Git-safe report.
    status = "PASS" if all(m.get("status") == "PASS" for m in (rubert_metrics, whisper_metrics, joint_metrics)) else "FAIL"
    return {
        "task": "T05",
        "methodology": METHODOLOGY,
        "status": status,
        "has_resource_warnings": any(f["level"] == "WARNING" for f in findings),
        "profile": profile_name,
        "machine_label": config["machine_label"],
        "created_at_utc": _utc_now(),
        "constraints": {
            "final_36_audio_used": False,
            "audio_declared_debug_only": True,
            "audio_final_name_heuristic_passed": True,
            "final_text_test_used": False,
        },
        "environment": environment,
        "inputs": {
            "train_parquet": config["paths"]["train_parquet"],
            "train_parquet_sha256": sha256_file(train_path),
            "train_ids_csv": config["paths"]["train_ids_csv"],
            "train_ids_file_sha256": sha256_file(train_ids_path),
            "rubert_smoke_batch_size": len(batch_df),
            "rubert_smoke_batch_label_counts": {str(k): int(v) for k, v in Counter(batch_df["label_id"].astype(int)).items()},
            "rubert_smoke_batch_ids_sha256": _sample_ids_hash(batch_df),
            "debug_audio": {**audio_meta, "limits": config["debug_audio"], "location": audio_location},
        },
        "train_provenance": provenance,
        "profile_config": config["profiles"][profile_name],
        "rubert": rubert_metrics,
        "whisper": {k: v for k, v in whisper_metrics.items() if k != "local_transcript_artifact"},
        "joint_load": joint_metrics,
        "memory": memory,
        "resource_findings": findings,
        "resource_plan": {
            "this_report_covers_profile_only": profile_name,
            "primary_profile": "primary",
            "reserve_profile": "reserve",
            "external_backup_compute": external_backup_status(config),
        },
        "local_only_artifacts": {
            "whisper_result": (Path(config["paths"]["local_artifacts_dir"]) / profile_name / "whisper_result.json").as_posix(),
            "contains_transcript": True,
            "must_not_commit": True,
            "git_safety": artifacts_location,
        },
    }


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_summary(report: dict[str, Any]) -> str:
    env, rub, wh, joint, mem = report["environment"], report["rubert"], report["whisper"], report["joint_load"], report["memory"]
    gpu = env.get("gpu") or {}
    ver = env.get("versions") or {}
    cuda = rub.get("cuda_memory_at_peak_window") or {}
    prov = report["train_provenance"]
    warm = (joint.get("whisper_while_rubert_loaded") or {}).get("transcription_seconds_warm")
    lines = [
        "# T05 — проверка вычислительных ресурсов",
        "",
        f"**Статус:** {report['status']}" + (" (есть предупреждения по ресурсам)" if report["has_resource_warnings"] else ""),
        f"**Профиль:** `{report['profile']}`",
        f"**Машина:** `{report.get('machine_label', 'n/a')}`",
        f"**Методика:** `{report['methodology']}`",
        "",
        "## Машина",
        "",
        f"- ОС: {env.get('os')} ({env.get('os_edition') or 'редакция не определена'})",
        f"- Python: {env.get('python')}",
        f"- CPU: {env.get('cpu')} (физических ядер {env.get('physical_cpu_count')}, логических {env.get('logical_cpu_count')})",
        f"- RAM, видимая ОС: {_fmt((env.get('ram_total_mib') or 0) / 1024)} GiB; свободно на старте: {_fmt(env.get('ram_available_mib_at_start'), 0)} MiB",
        f"- GPU: {gpu.get('name', 'n/a')}; VRAM: {_fmt((gpu.get('total_vram_mib') or 0) / 1024)} GiB",
        f"- PyTorch: {env.get('torch')} (CUDA runtime {env.get('torch_cuda_runtime')}), CUDA доступна: {env.get('cuda_available')}",
        f"- Библиотеки: transformers {ver.get('transformers')}, faster-whisper {ver.get('faster-whisper')}, "
        f"ctranslate2 {ver.get('ctranslate2')}, av {ver.get('av')}, psutil {ver.get('psutil')}",
        "",
        "## Реальный шаг RuBERT",
        "",
        f"- model: `{rub['model_name']}`, device: `{rub['device']}` (параметры модели и входы проверены на этом устройстве)",
        f"- batch_size: {rub['batch_size']}, max_length: {rub['max_length']}, mixed_precision: `{rub['mixed_precision']}`, шагов: {rub['training_steps']}",
        f"- фактическая длина тензора: {rub.get('encoded_sequence_length')} токенов (padding до max_length); "
        f"реальных токенов min/mean/max: {_fmt(rub.get('non_pad_tokens_min'))}/{_fmt(rub.get('non_pad_tokens_mean'))}/{_fmt(rub.get('non_pad_tokens_max'))}",
        f"- loss: {_fmt(rub['losses'][0], 6)} → {_fmt(rub['losses'][-1], 6)}",
        f"- первый шаг (холодный): {_fmt(rub.get('step_seconds_first_cold'), 3)} s; устоявшийся шаг: {_fmt(rub.get('step_seconds_steady_mean'), 3)} s",
        f"- encoder updated: {rub['encoder_updated']} (слои: {', '.join(rub['encoder_sentinel_parameters'])}; max|Δ|={rub['encoder_max_abs_update']:.3g}; "
        f"изменено тензоров {rub['encoder_tensors_changed']}/{rub['encoder_tensors_snapshotted']})",
        f"- head updated: {rub['head_updated']} (max|Δ|={rub['head_max_abs_update']:.3g}; изменено тензоров {rub['head_tensors_changed']}/{rub['head_tensors_snapshotted']})",
        f"- шагов пропущено GradScaler: {rub.get('optimizer_steps_skipped_by_grad_scaler')}",
        f"- CUDA peak allocated после backward (активации, зависят от max_length): {_fmt(rub.get('cuda_peak_allocated_after_backward_mib'))} MiB",
        f"- CUDA peak allocated за шаг: {_fmt(cuda.get('torch_peak_allocated_mib'))} MiB; peak reserved: {_fmt(cuda.get('torch_peak_reserved_mib'))} MiB",
        f"- занято на устройстве после шага (все процессы): {_fmt(cuda.get('device_used_mib'), 0)} MiB",
        f"- peak RSS процесса на этапе RuBERT (выборочно): {_fmt(rub.get('process_rss_peak_sampled_mib'))} MiB",
        f"- загрузка: токенизатор {_fmt(rub.get('tokenizer_load_seconds'), 3)} s, модель {_fmt(rub.get('model_load_seconds'), 3)} s",
        "",
        "## Реальное распознавание faster-whisper",
        "",
        f"- model: `{wh['model_size']}`, device: `{wh['device']}` (backend сообщает: {wh.get('backend_reported_device')}), compute_type: `{wh['compute_type']}`",
        f"- время загрузки: {_fmt(wh['model_load_seconds'], 3)} s (при первом запуске может включать скачивание)",
        f"- время распознавания (первый вызов, все segments прочитаны): {_fmt(wh['transcription_seconds'], 3)} s",
        f"- повторное распознавание при загруженном RuBERT: {_fmt(warm, 3)} s",
        f"- длительность аудио: {_fmt(wh.get('audio_duration_seconds'), 3)} s; real-time factor: {_fmt(wh.get('real_time_factor'), 3)}",
        f"- peak RSS процесса на этапе Whisper (RuBERT уже загружен, выборочно): {_fmt(wh.get('process_rss_peak_sampled_mib'))} MiB",
        f"- сегментов: {wh['segment_count']}; символов в транскрипции: {wh['transcript_chars']}; SHA-256: `{wh['transcript_sha256']}`",
        "- язык распознавания задан явно (`ru`), поэтому language_probability не является доказательством определения языка.",
        "- текст транскрипции хранится только в локальном `artifacts/resource_check/` и не коммитится.",
        "",
        "## Совместная загрузка",
        "",
        f"- обе модели в памяти одновременно: {joint['rubert_and_whisper_loaded_together']}; порядок операций: {' → '.join(joint['operation_order'])}",
        f"- RuBERT после Whisper выдаёт те же логиты: {joint['rubert_logits_identical_before_after_whisper']}; forward {_fmt(joint['rubert_forward_seconds_before_whisper'], 3)} s / {_fmt(joint['rubert_forward_seconds_after_whisper'], 3)} s",
        f"- текущий RSS процесса: {_fmt(joint['process_rss_mib'])} MiB",
        "",
        "## Память за весь прогон",
        "",
        f"- пиковый RSS процесса (по данным ОС): {_fmt(mem.get('process_peak_rss_mib_os'))} MiB; по выборке: {_fmt(mem.get('process_peak_rss_mib_sampled'))} MiB",
        f"- минимум свободной системной RAM: {_fmt(mem.get('system_available_mib_min_sampled'), 0)} MiB (на старте {_fmt(mem.get('system_available_mib_at_start'), 0)} MiB)",
        "",
        "## Предупреждения и проблемы ресурсов",
        "",
    ]
    lines += [f"- [{f['level']}] {f['message']}" for f in report["resource_findings"]]
    lines += [
        "",
        "## Происхождение обучающей выборки",
        "",
        f"- train.parquet и список ID совпадают с SHA-256 из run_metadata.json T04: {prov['train_parquet_sha256_matches_t04'] and prov['train_ids_sha256_matches_t04']}",
        f"- все ID входят в сплит `train` T03 (validation/test не использовались): {prov['all_ids_in_t03_train_split']}",
        f"- состав smoke-батча по классам: {report['inputs']['rubert_smoke_batch_label_counts']}; SHA-256 списка ID: `{report['inputs']['rubert_smoke_batch_ids_sha256']}`",
        "",
        "## Ограничения",
        "",
        "- Финальные 36 голосовых записей не использовались; отладочное аудио объявлено отдельным (SHA-256 записан в отчёте).",
        "- Финальный текстовый test не использовался.",
        f"- Отчёт подтверждает только профиль `{report['profile']}` на машине `{report.get('machine_label')}` в момент запуска.",
        f"- Внешний резервный доступ к вычислениям: {report['resource_plan']['external_backup_compute']['status']} — скриптом не проверялся и не заявляется.",
        "",
    ]
    return "\n".join(lines)


def write_summary(path: Path, report: dict[str, Any]) -> None:
    write_text_lf(path, render_summary(report))


def resource_table_rows(report: dict[str, Any]) -> list[tuple[str, str, Any, str]]:
    env, rub, wh, joint, mem = report["environment"], report["rubert"], report["whisper"], report["joint_load"], report["memory"]
    gpu = env.get("gpu") or {}
    ver = env.get("versions") or {}
    cuda = rub.get("cuda_memory_at_peak_window") or {}
    warm = (joint.get("whisper_while_rubert_loaded") or {}).get("transcription_seconds_warm")
    return [
        ("run", "profile", report["profile"], "name"),
        ("run", "machine_label", report["machine_label"], "name"),
        ("run", "status", report["status"], "status"),
        ("run", "methodology", report["methodology"], "version"),
        ("environment", "os", env.get("os"), "name"),
        ("environment", "os_edition", env.get("os_edition"), "name"),
        ("environment", "python", env.get("python"), "version"),
        ("environment", "cpu", env.get("cpu"), "name"),
        ("environment", "ram_total_visible_to_os", (env.get("ram_total_mib") or 0) / 1024, "GiB"),
        ("environment", "ram_available_at_start", env.get("ram_available_mib_at_start"), "MiB"),
        ("environment", "gpu", gpu.get("name"), "name"),
        ("environment", "vram_total", (gpu.get("total_vram_mib") or 0) / 1024, "GiB"),
        ("environment", "torch", env.get("torch"), "version"),
        ("environment", "transformers", ver.get("transformers"), "version"),
        ("environment", "faster_whisper", ver.get("faster-whisper"), "version"),
        ("environment", "ctranslate2", ver.get("ctranslate2"), "version"),
        ("rubert", "device", rub.get("device"), "name"),
        ("rubert", "batch_size", rub.get("batch_size"), "records"),
        ("rubert", "max_length", rub.get("max_length"), "tokens"),
        ("rubert", "encoded_sequence_length", rub.get("encoded_sequence_length"), "tokens"),
        ("rubert", "mixed_precision", rub.get("mixed_precision"), "name"),
        ("rubert", "training_steps", rub.get("training_steps"), "steps"),
        ("rubert", "model_load_seconds", rub.get("model_load_seconds"), "seconds"),
        ("rubert", "step_seconds_first_cold", rub.get("step_seconds_first_cold"), "seconds"),
        ("rubert", "step_seconds_steady_mean", rub.get("step_seconds_steady_mean"), "seconds"),
        ("rubert", "cuda_peak_allocated_after_backward", rub.get("cuda_peak_allocated_after_backward_mib"), "MiB"),
        ("rubert", "cuda_peak_allocated", cuda.get("torch_peak_allocated_mib"), "MiB"),
        ("rubert", "cuda_peak_reserved", cuda.get("torch_peak_reserved_mib"), "MiB"),
        ("rubert", "cuda_device_used_after_step", cuda.get("device_used_mib"), "MiB"),
        ("rubert", "process_rss_peak_sampled", rub.get("process_rss_peak_sampled_mib"), "MiB"),
        ("rubert", "encoder_updated", rub.get("encoder_updated"), "bool"),
        ("rubert", "head_updated", rub.get("head_updated"), "bool"),
        ("whisper", "model_size", wh.get("model_size"), "name"),
        ("whisper", "device", wh.get("device"), "name"),
        ("whisper", "compute_type", wh.get("compute_type"), "name"),
        ("whisper", "model_load_seconds", wh.get("model_load_seconds"), "seconds"),
        ("whisper", "transcription_seconds", wh.get("transcription_seconds"), "seconds"),
        ("whisper", "transcription_seconds_warm", warm, "seconds"),
        ("whisper", "audio_duration_seconds", wh.get("audio_duration_seconds"), "seconds"),
        ("whisper", "real_time_factor", wh.get("real_time_factor"), "ratio"),
        ("whisper", "process_rss_peak_sampled", wh.get("process_rss_peak_sampled_mib"), "MiB"),
        ("whisper", "transcript_chars", wh.get("transcript_chars"), "chars"),
        ("joint", "both_loaded", joint.get("rubert_and_whisper_loaded_together"), "bool"),
        ("joint", "rubert_logits_identical_before_after_whisper", joint.get("rubert_logits_identical_before_after_whisper"), "bool"),
        ("joint", "process_rss_now", joint.get("process_rss_mib"), "MiB"),
        ("memory", "process_peak_rss_whole_run_os", mem.get("process_peak_rss_mib_os"), "MiB"),
        ("memory", "process_peak_rss_whole_run_sampled", mem.get("process_peak_rss_mib_sampled"), "MiB"),
        ("memory", "system_available_min_sampled", mem.get("system_available_mib_min_sampled"), "MiB"),
        ("constraints", "final_36_audio_used", report["constraints"]["final_36_audio_used"], "bool"),
        ("constraints", "final_text_test_used", report["constraints"]["final_text_test_used"], "bool"),
        ("constraints", "external_backup_compute", report["resource_plan"]["external_backup_compute"]["status"], "status"),
    ]


def render_resource_table(report: dict[str, Any]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["section", "metric", "value", "unit"])
    for section, metric, value, unit in resource_table_rows(report):
        writer.writerow([section, metric, "" if value is None else value, unit])
    return buffer.getvalue()


def write_resource_table(path: Path, report: dict[str, Any]) -> None:
    write_text_lf(path, render_resource_table(report))


# --------------------------------------------------------------------------- #
# Pre-flight and CLI
# --------------------------------------------------------------------------- #
def preflight(config: dict[str, Any], profile_name: str, root: Path, audio_path: Path | None) -> dict[str, Any]:
    """Cheap checks only: no model download, no training, no test data, no audio needed."""
    issues: list[str] = []
    warnings: list[str] = []
    details: dict[str, Any] = {}
    try:
        import torch
    except ImportError:
        torch = None
        issues.append("PyTorch is not installed.")
    try:
        import psutil
    except ImportError:
        psutil = None
        issues.append("psutil is not installed.")

    for package in ("transformers", "faster-whisper", "pyarrow"):
        if package_version(package) is None:
            issues.append(f"{package} is not installed.")
    if package_version("av") is None:
        warnings.append("PyAV ('av') is not installed: only .wav debug audio can be validated (duration limit).")

    profile = config["profiles"][profile_name]
    if torch is not None and profile["rubert"]["device"] == "cuda":
        if not torch.cuda.is_available():
            issues.append("Selected RuBERT profile requires CUDA but torch.cuda.is_available() is False.")
        else:
            details["cuda_device"] = torch.cuda.get_device_name(0)
    if profile["whisper"]["device"] == "cuda":
        warnings.append(
            "Whisper CUDA uses CTranslate2 runtime; PyTorch CUDA availability alone does not prove that backend is ready."
        )
    if psutil is not None:
        logical = psutil.cpu_count(logical=True) or 0
        if logical and int(profile["whisper"]["cpu_threads"]) > logical:
            warnings.append(f"Whisper cpu_threads={profile['whisper']['cpu_threads']} exceeds logical CPU count={logical}.")

    paths = config["paths"]
    train_path = resolve_config_path(root, paths["train_parquet"])
    ids_path = resolve_config_path(root, paths["train_ids_csv"])
    metadata_path = resolve_config_path(root, paths["baseline_run_metadata"])
    split_ids_path = resolve_config_path(root, paths["split_ids_csv"])
    for label, path in (
        ("prepared train", train_path),
        ("T04 train IDs", ids_path),
        ("T04 run_metadata.json", metadata_path),
        ("T03 split_ids.csv", split_ids_path),
    ):
        if not path.exists():
            issues.append(f"Missing {label}: {path}")
    if all(p.exists() for p in (train_path, ids_path, metadata_path, split_ids_path)):
        try:
            train, train_ids = _read_train_and_ids(train_path, ids_path)
            details["prepared_train_rows"] = int(len(train))
            details["common_train_ids_rows"] = int(len(train_ids))
            details["common_train_class_counts"] = {str(k): int(v) for k, v in Counter(train_ids["label_id"].astype(int)).items()}
            details["train_provenance"] = verify_train_provenance(
                train_path=train_path, train_ids=train_ids, metadata_path=metadata_path, split_ids_path=split_ids_path
            )
        except Exception as exc:
            issues.append(f"Prepared train / T04 train IDs contract failed: {type(exc).__name__}: {exc}")

    artifacts_file = resolve_config_path(root, paths["local_artifacts_dir"]) / profile_name / "whisper_result.json"
    try:
        details["local_artifacts_git_safety"] = git_safety_status(root, artifacts_file, "Local transcript folder")
    except T05Error as exc:
        issues.append(str(exc))

    if audio_path is not None:
        try:
            details.update(validate_debug_audio(audio_path, config))
            details["debug_audio_git_safety"] = git_safety_status(root, audio_path, "Debug audio file")
        except T05Error as exc:
            issues.append(str(exc))

    return {
        "status": "PASS" if not issues else "FAIL",
        "profile": profile_name,
        "machine_label": config["machine_label"],
        "issues": issues,
        "warnings": warnings,
        "details": details,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T05: real resource check for RuBERT + faster-whisper.")
    parser.add_argument("--config", default="configs/resource_check.json", help="Path to T05 JSON config.")
    parser.add_argument("--profile", default="primary", help="Resource profile name (primary/reserve).")
    parser.add_argument("--audio", help="Path to a separate debug Russian speech recording.")
    parser.add_argument(
        "--confirm-debug-audio",
        action="store_true",
        help="Confirm that --audio is a separate debug recording and not one of the final 36 recordings.",
    )
    parser.add_argument("--preflight", action="store_true", help="Only validate environment and paths; do not run models.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Set HF_HUB_OFFLINE=1: prove that both models load from the local cache (no network).",
    )
    return parser.parse_args(argv)


def _remove_stale_outputs(paths: Iterable[Path]) -> None:
    """A failed re-run must not leave an old PASS report of an older methodology behind."""
    for path in paths:
        path.unlink(missing_ok=True)


def main(argv: list[str] | None = None, root: Path | None = None) -> int:
    args = parse_args(argv)
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    root = (root if root is not None else repo_root_from_script()).resolve()
    config_path = resolve_repo_path(root, args.config)
    try:
        config = load_config(config_path)
        if args.profile not in config["profiles"]:
            raise T05Error(f"Unknown profile {args.profile!r}; available: {sorted(config['profiles'])}")
        audio_path = resolve_repo_path(root, args.audio) if args.audio else None

        if not args.preflight:
            # Cheap declarations first: do not even open an audio file that was not confirmed.
            if audio_path is None:
                raise T05Error("--audio is required for the real T05 check.")
            if not args.confirm_debug_audio:
                raise T05Error(
                    "Refusing to use audio without --confirm-debug-audio. "
                    "T05 must use a separate debug recording, not the final 36."
                )

        pf = preflight(config, args.profile, root, audio_path)
        if args.preflight:
            print(dumps_json(pf))
            return 0 if pf["status"] == "PASS" else 2
        if pf["issues"]:
            raise T05Error("Preflight failed:\n- " + "\n- ".join(pf["issues"]))
        assert audio_path is not None
        audio_meta = validate_debug_audio(audio_path, config)

        try:
            import psutil
            import torch
        except ImportError as exc:
            raise T05Error("Install psutil and working PyTorch before the real T05 run.") from exc

        paths = config["paths"]
        train_path = resolve_config_path(root, paths["train_parquet"])
        train_ids_path = resolve_config_path(root, paths["train_ids_csv"])
        metadata_path = resolve_config_path(root, paths["baseline_run_metadata"])
        split_ids_path = resolve_config_path(root, paths["split_ids_csv"])
        reports_root = resolve_config_path(root, paths["reports_dir"]) / args.profile
        local_artifacts_root = resolve_config_path(root, paths["local_artifacts_dir"]) / args.profile
        report_path = reports_root / "resource_report.json"
        summary_path = reports_root / "resource_summary.md"
        table_path = reports_root / "resource_table.csv"
        sample_ids_path = reports_root / "rubert_sample_ids.csv"
        local_whisper_path = local_artifacts_root / "whisper_result.json"
        artifacts_location = git_safety_status(root, local_whisper_path, "Local transcript folder")
        audio_location = git_safety_status(root, audio_path, "Debug audio file")
        _remove_stale_outputs([report_path, summary_path, table_path, sample_ids_path])
        reports_root.mkdir(parents=True, exist_ok=True)
        local_artifacts_root.mkdir(parents=True, exist_ok=True)

        environment = collect_environment(torch, psutil, root)
        profile = config["profiles"][args.profile]
        train, train_ids = _read_train_and_ids(train_path, train_ids_path)
        provenance = verify_train_provenance(
            train_path=train_path, train_ids=train_ids, metadata_path=metadata_path, split_ids_path=split_ids_path
        )
        batch_df = select_fixed_batch(train, train_ids, int(profile["rubert"]["batch_size"]))
        write_sample_ids(sample_ids_path, batch_df)

        tracker = PeakRSSMonitor(psutil).start()
        try:
            runtime, rubert_metrics = run_rubert_training_step(
                config=config, profile_name=args.profile, batch_df=batch_df, psutil_module=psutil
            )
            whisper_model, whisper_metrics, transcript_text = run_whisper_transcription(
                profile=profile,
                audio_path=audio_path,
                local_artifact_path=local_whisper_path,
                psutil_module=psutil,
                language=config["debug_audio"]["language"],
            )
            runtime.whisper_model = whisper_model
            joint_metrics = verify_joint_operation(
                runtime=runtime,
                profile=profile,
                batch_df=batch_df,
                audio_path=audio_path,
                language=config["debug_audio"]["language"],
                cold_transcript_sha256=whisper_metrics["transcript_sha256"],
                psutil_module=psutil,
            )
        finally:
            tracker.stop()
        memory = build_memory_summary(environment=environment, tracker=tracker, psutil_module=psutil)
        findings = build_resource_findings(
            environment=environment, memory=memory, rubert=rubert_metrics, whisper=whisper_metrics, config=config
        )

        report = build_report(
            config=config,
            profile_name=args.profile,
            environment=environment,
            train_path=train_path,
            train_ids_path=train_ids_path,
            provenance=provenance,
            batch_df=batch_df,
            audio_meta=audio_meta,
            audio_location=audio_location,
            artifacts_location=artifacts_location,
            rubert_metrics=rubert_metrics,
            whisper_metrics=whisper_metrics,
            joint_metrics=joint_metrics,
            memory=memory,
            findings=findings,
        )
        sanitize_report_for_git(report)
        outputs = {
            "resource_report.json": dumps_json(report),
            "resource_summary.md": render_summary(report),
            "resource_table.csv": render_resource_table(report),
        }
        assert_no_text_leak(outputs, [transcript_text, *batch_df["review_text"].astype(str).tolist()])
        write_text_lf(report_path, outputs["resource_report.json"])
        write_text_lf(summary_path, outputs["resource_summary.md"])
        write_text_lf(table_path, outputs["resource_table.csv"])

        steady = rubert_metrics.get("step_seconds_steady_mean")
        peak_rss = memory.get("process_peak_rss_mib_os") or memory.get("process_peak_rss_mib_sampled")
        warm = joint_metrics["whisper_while_rubert_loaded"]["transcription_seconds_warm"]
        print("T05 real resource check completed.")
        print(f"Profile: {args.profile}")
        print(
            f"RuBERT: PASS | batch={rubert_metrics['batch_size']} | max_length={rubert_metrics['max_length']} "
            f"| first step={rubert_metrics['step_seconds_first_cold']:.3f}s "
            f"| steady step={'n/a' if steady is None else f'{steady:.3f}s'} "
            f"| encoder/head updated={rubert_metrics['encoder_updated']}/{rubert_metrics['head_updated']}"
        )
        print(
            f"Whisper: PASS | {whisper_metrics['model_size']} on {whisper_metrics['device']} "
            f"| transcription={whisper_metrics['transcription_seconds']:.3f}s (repeat {warm:.3f}s) "
            f"| segments={whisper_metrics['segment_count']} | chars={whisper_metrics['transcript_chars']}"
        )
        print(
            f"Joint load: PASS | {' -> '.join(joint_metrics['operation_order'])} | RSS now={joint_metrics['process_rss_mib']:.1f} MiB"
        )
        print(
            f"Whole-run peak RSS: {peak_rss:.1f} MiB | min free system RAM: "
            f"{(memory.get('system_available_mib_min_sampled') or 0):.0f} MiB"
        )
        for finding in findings:
            if finding["level"] == "WARNING":
                print(f"WARNING [{finding['code']}]: {finding['message']}")
        print(f"Git-safe report: {report_path}")
        print(f"Summary: {summary_path}")
        print(f"Resource table: {table_path}")
        print(f"Local transcript artifact (DO NOT COMMIT): {local_whisper_path}")
        print("Final 36 audio recordings were NOT used. Final text test was NOT used.")
        return 0
    except T05Error as exc:
        print(f"T05 ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"
        lowered = message.lower()
        if "out of memory" in lowered or "cuda oom" in lowered:
            print(
                "T05 RESOURCE ERROR: CUDA/host memory was insufficient for this profile. "
                "Run the reserve profile and keep this failure as evidence that the primary profile does not fit. "
                f"Original error: {message}",
                file=sys.stderr,
            )
            return 4
        print(f"T05 UNEXPECTED ERROR: {message}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
