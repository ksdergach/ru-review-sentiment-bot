"""Train-only boundary-frequency diagnostic; does not change the fitted baseline."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import train_baseline as b


def main():
    c = b.load_config(ROOT / "configs/nlp_baseline.json")
    meta = b.read_json(ROOT / c["report_dir"] / "run_metadata.json")
    train = ROOT / c["train_path"]
    if b.sha256_file(train) != meta["train_parquet_sha256"]:
        raise ValueError("Training data checksum changed")
    params = {k: v for k, v in c["tfidf"].items() if k not in ("sublinear_tf", "norm", "use_idf", "smooth_idf")}
    params.update(ngram_range=tuple(params["ngram_range"]), max_features=None)
    vectorizer = CountVectorizer(**params)
    counts = vectorizer.fit_transform(pd.read_parquet(train)[c["text_column"]])
    frequencies = np.asarray(counts.sum(axis=0)).ravel()
    limit = c["tfidf"]["max_features"]
    if limit is None or len(frequencies) <= limit:
        raise ValueError("No vocabulary truncation boundary to diagnose")
    cutoff = int(np.sort(frequencies)[-limit])
    above = int((frequencies > cutoff).sum())
    bundle_path = ROOT / c["model_dir"] / c["bundle_name"]
    if b.sha256_file(bundle_path) != meta["bundle_sha256"]:
        raise ValueError("Model checksum changed")
    selected = set(b.load_bundle(bundle_path)["vectorizer"].vocabulary_)
    result = {"train_only": True, "train_sha256": meta["train_parquet_sha256"],
              "eligible_features": len(frequencies), "max_features": limit,
              "cutoff_term_frequency": cutoff, "features_above_cutoff": above,
              "features_at_cutoff": int((frequencies == cutoff).sum()), "available_slots_at_cutoff": limit - above,
              "selected_tied_features": sum(term in selected for term, index in vectorizer.vocabulary_.items() if frequencies[index] == cutoff),
              "implementation": "CountVectorizer._limit_features uses (-tfs[mask]).argsort() without stable=True or an explicit secondary key",
              "conclusion": "Boundary ties are present; different selection across original Windows/Mac runs is a hypothesis, not proven without the original Windows bundle. IDF floating point differences also not excluded.",
              "windows_bundle_available": False}
    b.write_json(ROOT / "reports/nlp/tfidf_boundary_diagnostic.json", result)
    print(result)


if __name__ == "__main__":
    main()
