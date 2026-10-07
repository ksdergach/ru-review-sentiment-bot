"""Compare two trusted local training bundles on validation only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.train_baseline import load_bundle, _vectorizer_state_sha256, sha256_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--first-dir', type=Path, default=ROOT / 'artifacts/t04-reproducibility/stable-first')
    args = parser.parse_args()
    first_meta = json.loads((args.first_dir / 'run_metadata.json').read_text())
    second_meta = json.loads((ROOT / 'reports/baseline/run_metadata.json').read_text())
    first_path = args.first_dir / 'baseline.joblib'
    second_path = ROOT / second_meta['bundle_path']
    for path, meta in ((first_path, first_meta), (second_path, second_meta)):
        assert sha256_file(path) == meta['bundle_sha256']
        assert meta['final_test_used'] is False
        assert not meta['winner_convergence_warning']
        assert meta['script_sha256'] == sha256_file(ROOT / 'src/train_baseline.py')
    first, second = load_bundle(first_path), load_bundle(second_path)
    cfg = json.loads((ROOT / 'configs/baseline.json').read_text())
    path = ROOT / cfg['validation_path']
    assert sha256_file(path) == first_meta['validation_parquet_sha256'] == second_meta['validation_parquet_sha256']
    texts = pd.read_parquet(path)[cfg['text_column']].tolist()
    probabilities = []
    for bundle in (first, second):
        probabilities.append(bundle['classifier'].predict_proba(bundle['vectorizer'].transform(texts)))
    checks = {
        'train_ids_match': first_meta['train_ids_sha256'] == second_meta['train_ids_sha256'],
        'validation_ids_match': first_meta['validation_ids_sha256'] == second_meta['validation_ids_sha256'],
        'vocabulary_match': first['vectorizer'].vocabulary_ == second['vectorizer'].vocabulary_,
        'idf_exact_match': np.array_equal(first['vectorizer'].idf_, second['vectorizer'].idf_),
        'tfidf_state_match': _vectorizer_state_sha256(first['vectorizer']) == _vectorizer_state_sha256(second['vectorizer']),
        'coefficients_exact_match': np.array_equal(first['classifier'].coef_, second['classifier'].coef_),
        'intercepts_exact_match': np.array_equal(first['classifier'].intercept_, second['classifier'].intercept_),
        'validation_predictions_match': np.array_equal(probabilities[0].argmax(axis=1), probabilities[1].argmax(axis=1)),
        'validation_probabilities_match': np.allclose(*probabilities, atol=1e-12, rtol=0),
        'winner_matches': first_meta['winner_C'] == second_meta['winner_C'],
    }
    first_winner = json.loads((args.first_dir / 'winner_config.json').read_text())['winner']
    second_winner = json.loads((ROOT / 'reports/baseline/winner_config.json').read_text())['winner']
    checks['winner_metrics_match'] = first_winner == second_winner
    assert all(checks.values()), checks
    output = {
        'date': '2026-10-07', 'status': 'PASS', 'checks': checks,
        'validation_rows': len(texts), 'max_probability_difference': float(np.max(np.abs(probabilities[0] - probabilities[1]))),
        'first_bundle_sha256': sha256_file(first_path), 'accepted_bundle_sha256': sha256_file(second_path),
        'serialized_bytes_match': sha256_file(first_path) == sha256_file(second_path),
        'tfidf_state_sha256': _vectorizer_state_sha256(second['vectorizer']),
        'source_script_sha256': second_meta['script_sha256'],
        'verification_script_sha256': sha256_file(Path(__file__)),
        'winner': second_winner, 'final_test_used': False,
        'scope': 'Two full fresh fits and bundle loads on macOS arm64; Windows not rerun.',
        'training_environment': {'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'},
        'historical_windows_model_available': False,
    }
    out = ROOT / 'reports/baseline/reproducibility/repeat_runs.json'
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
