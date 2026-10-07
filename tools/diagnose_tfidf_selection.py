"""Reproduce TF-IDF feature-cap ambiguity on train only; export aggregates, no text."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train_baseline import build_vectorizer, _vectorizer_state_sha256, library_versions


def main():
    cfg = json.loads((ROOT / 'configs/baseline.json').read_text())
    # Deliberately no test/validation access: vocabulary is a train-only quantity.
    train_path = ROOT / cfg['train_path']
    texts = pd.read_parquet(train_path)[cfg['text_column']].tolist()
    params = {k: v for k, v in cfg['tfidf'].items() if k in CountVectorizer().get_params()}
    params.update(max_features=None, ngram_range=tuple(params['ngram_range']), dtype=np.int64)
    counter = CountVectorizer(**params)
    counts = counter.fit_transform(texts)
    terms = counter.get_feature_names_out()
    frequencies = np.asarray(counts.sum(axis=0)).ravel()
    limit = cfg['tfidf']['max_features']
    if limit is None or len(terms) <= limit:
        raise ValueError('This diagnostic requires an active feature cap')
    cutoff = int(np.sort(frequencies)[-limit])
    greater = int(np.count_nonzero(frequencies > cutoff))
    result = {
        'train_rows': len(texts), 'eligible_features': len(terms), 'limit': limit,
        'cutoff_term_frequency': cutoff, 'above_cutoff': greater,
        'tied_at_cutoff': int(np.count_nonzero(frequencies == cutoff)), 'tie_slots': limit - greater,
        'train_parquet_sha256': hashlib.sha256(train_path.read_bytes()).hexdigest(),
        'library_versions': library_versions(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'historical_acceptance_sha256': hashlib.sha256((ROOT / 'reports/acceptance/2026-10-03.json').read_bytes()).hexdigest(),
        'candidate_states': {}, 'final_test_used': False,
        'scope': 'Different equal-frequency tie policies on the same train; not a rerun on Windows.',
    }
    selections = {}
    for kind in ('quicksort', 'heapsort', 'stable'):
        # Float64 reproduces the dtype used by the original TfidfVectorizer.
        order = np.argsort(-frequencies.astype(np.float64), kind=kind)[:limit]
        selected = sorted(str(x) for x in terms[order])
        selections[kind] = set(selected)
        v = build_vectorizer(cfg['tfidf'])
        v.set_params(vocabulary={term: i for i, term in enumerate(selected)})
        v.fit(texts)
        result['candidate_states'][kind] = {'tfidf_state_sha256': _vectorizer_state_sha256(v)}
    for kind in selections:
        result['candidate_states'][kind]['symmetric_difference_from_quicksort'] = len(selections[kind] ^ selections['quicksort'])
    out = ROOT / 'reports/baseline/reproducibility'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'feature_selection.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'library_versions'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
