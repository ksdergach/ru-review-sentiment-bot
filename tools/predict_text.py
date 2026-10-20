"""Run the same single-text inference used by Telegram, without a bot token.

    CLASSIFIER_BACKEND=rubert python tools/predict_text.py 'Отличный фильм'

This is a prediction adapter, not an evaluation of the reserved final test.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.classifier import ClassifierError, InvalidTextError, predict


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("text")
    args = parser.parse_args(argv)
    try:
        result = predict(args.text)
    except (ClassifierError, InvalidTextError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
