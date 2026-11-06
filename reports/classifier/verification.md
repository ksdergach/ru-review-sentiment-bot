# T07: проверка общего интерфейса

Реализованы baseline и RuBERT через `src.classifier.predict` и фабрику
`create_classifier`. Обязательный результат: label, label_id, confidence,
probabilities в порядке negative/neutral/positive. Поле truncated сообщает об
усечении; baseline возвращает false. Пустые, нестроковые и превышающие 2000
символов сообщения отклоняются до загрузки модели.

Проверка выполнена 06.11.2026 на состоянии `master` (`2c47aa5`, T06) с
изменениями T07; Linux x86_64, CPU, Python 3.13.7, torch 2.10.0+cpu,
transformers 5.5.0. Финальная test-выборка не использовалась.

## Принятый комплект RuBERT

- `python tools/fetch_transformer.py`: архив `t06_rubert_bundle.tar` получен по
  `configs/transformer_bundle.json`, размер 119 224 320 байт, SHA-256
  `68970b2e3422a873d14da75a4910f6f813ade73c3ca99c103068f6f2d9e4c703`;
  все 7 файлов `models/transformer` совпали с
  `reports/transformer/run_metadata.json::bundle_files_sha256`.
- `HF_HUB_OFFLINE=1 python tools/verify_transformer.py`: status passed,
  53 проверки, `final_test_used: false`.
- `tests/test_classifier_accepted_rubert.py` (5 тестов, выполняются при
  установленном комплекте) и отдельная проверка с запретом сетевых соединений
  и функций обучения:
  - `RuBERTClassifier()` и `predict` при `CLASSIFIER_BACKEND=rubert` загружают
    комплект по умолчанию; для трёх текстов `train_baseline._smoke_texts()`
    метки positive / negative / positive, label_id 2 / 0 / 2, confidence
    0,974920 / 0,961005 / 0,375531;
  - результат совпадает с `train_transformer.predict_texts` (|Δp| ≤ 1e-7) и с
    `reports/transformer/bundle_roundtrip.json` (|Δp| ≤ 1e-6);
  - ответ бота равен `format_prediction` общего результата; JSON
    `tools/predict_text.py` совпадает с прямым вызовом;
  - порядок классов: `label_mapping.json`, `training_config.json`,
    `config.json` (id2label/label2id) и порядок вероятностей —
    negative=0, neutral=1, positive=2;
  - усечение на настоящем tokenizer (max_length 64 с учётом специальных
    токенов): 64 токена — truncated false, 65 и 74 — true; бот добавляет
    «Учтено только начало отзыва: текст усечён моделью.» только при true;
  - модель загружается один раз: после первой загрузки 4 одновременных и
    5 последовательных вызовов не загружают её повторно;
  - сетевых соединений и вызовов обучения нет; `data_version` содержит
    `final_test_used: false` и `train_ids_equal_baseline: true`.

## Ошибки комплекта

Отсутствующий каталог, удалённый или лишний файл, изменённые веса,
переставленный mapping, отсутствующий отчёт и несовместимая конфигурация
обработки приводят к `ModelUnavailableError` до загрузки весов. Сообщение
содержит путь и причину, например
`Could not load verified RuBERT bundle <путь>: ValueError: Transformer bundle checksum mismatch: label_mapping.json`;
для отсутствующего каталога указана команда `python tools/fetch_transformer.py`.
`tools/predict_text.py` печатает сообщение и завершается с кодом 1, бот
отвечает сообщением о недоступности классификации.

## Baseline

Принятый комплект `models/nlp_baseline/tfidf_logreg_baseline.joblib`
(SHA-256 `389a768bcd108721c79ebf3a7b9dc5212ec100f5c2e1b06cf519ed076cd9bee3`)
в проверочной среде отсутствовал. Локально обученный командой
`python src/train_baseline.py --config configs/nlp_baseline.json` комплект
имеет другой SHA-256 (`fa490d90…`) и validation macro-F1 0,719980, поэтому
используется только для проверки интерфейса, а не как принятая модель.
На нём для трёх контрольных текстов (positive / negative / neutral) общий
`predict`, прямой `train_baseline.predict_texts`, бот и `tools/predict_text.py`
дали одинаковые результаты; порядок классов в сохранённом комплекте —
negative/neutral/positive.

## Тесты

- `HF_HUB_OFFLINE=1 python -m pytest -q` на `master` + T07 с установленным
  комплектом RuBERT — 567 passed.
- Без `models/transformer` тесты T07 — 76 passed, 1 skipped
  (`tests/test_classifier_accepted_rubert.py`).

Код T06, принятые веса, метрики и данные не изменялись.
