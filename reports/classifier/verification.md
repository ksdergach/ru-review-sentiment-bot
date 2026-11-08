# T07: проверка общего интерфейса

Реализованы baseline и RuBERT через `src.classifier.predict` (бот, CLI),
`src.classifier.predict_batch` (оценка) и фабрику `create_classifier`.
Обязательный результат: label, label_id, confidence, probabilities в порядке
negative/neutral/positive и truncated; baseline возвращает truncated false.
`predict` и `predict_batch` используют одну загрузку, обработку и проверку
результата. Пустые, нестроковые и (для интерактивного ввода) превышающие
2000 символов тексты отклоняются до загрузки модели.

Проверка выполнена 07–08.11.2026 на результате слияния `master` (`2c47aa5`, T06)
с изменениями T07;
Linux x86_64, CPU, Python 3.13.7, torch 2.10.0+cpu, transformers 5.5.0,
scikit-learn 1.7.2. Финальная test-выборка не использовалась.

## Принятый комплект RuBERT (T06)

- `python tools/fetch_transformer.py`: архив `t06_rubert_bundle.tar`,
  119 224 320 байт, SHA-256
  `68970b2e3422a873d14da75a4910f6f813ade73c3ca99c103068f6f2d9e4c703`;
  все 7 файлов совпали с `reports/transformer/run_metadata.json`.
- `HF_HUB_OFFLINE=1 python tools/verify_transformer.py`: passed, 53 проверки.
- `tests/test_classifier_accepted_rubert.py` — 7 passed. Запреты сетевых
  соединений (`socket.connect`, `connect_ex`, `create_connection`,
  `getaddrinfo`) и обучения (`run_training`, `load_pretrained`, `make_loader`,
  `Tensor.backward`, `Optimizer.step`) установлены модульной фикстурой до
  создания классификатора; первая загрузка, конкурентная первая загрузка
  (4 потока, одна загрузка), инференс и повторное использование проходят под
  ними, попыток обращения нет.
  - три текста `train_baseline._smoke_texts()`: positive / negative /
    positive, label_id 2 / 0 / 2, confidence 0,974920 / 0,961005 / 0,375531;
    совпадение с `train_transformer.predict_texts` (|Δp| ≤ 1e-7) и с
    `reports/transformer/bundle_roundtrip.json` (|Δp| ≤ 1e-6);
  - бот, `tools/predict_text.py` и `predict_batch` дают те же label,
    label_id, truncated; вероятности `predict_batch` совпадают с `predict`
    в пределах 1e-6 (паддинг пакета);
  - порядок классов `label_mapping.json`, `config.json`, вероятностей —
    negative=0/neutral=1/positive=2; копия с переставленным mapping отклонена;
  - усечение (max_length 64 с учётом специальных токенов): 64 токена —
    false, 65 — true; бот предупреждает только при true.
- `CLASSIFIER_BACKEND=rubert python tools/evaluate_classifier.py` (оценка
  через `predict_batch` на полной validation, 10 396 строк): macro-F1
  0,6678240259861936, матрица ошибок совпала с
  `reports/transformer/run_metadata.json`; усечено 3 340 текстов.

## Baseline (T04/N02)

Принятый комплект `models/nlp_baseline/tfidf_logreg_baseline.joblib`
предоставлен командой проекта (в Git не хранится): 3 598 091 байт, SHA-256
`389a768bcd108721c79ebf3a7b9dc5212ec100f5c2e1b06cf519ed076cd9bee3` совпал с
`reports/nlp/training/run_metadata.json` до десериализации; SHA-256
`data/processed/train.parquet` и `validation.parquet` совпали с тем же отчётом.

- `tests/test_classifier_accepted_baseline.py` — 8 passed. Запреты сетевых
  соединений и обучения (`fit`/`fit_transform` у `CountVectorizer`,
  `TfidfVectorizer`, `LogisticRegression`; `run_training`,
  `fit_vectorizer_on_train`, `build_classifier`, `save_bundle`) установлены
  модульной фикстурой до первой загрузки, попыток обращения нет; комплект
  загружен один раз, в том числе при конкурентной первой загрузке (4 потока).
  - загруженное состояние совпадает с N02: `model_state_sha256`
    `273359cc…` = `reports/nlp/baseline_metrics.json`; версия данных
    (SHA-256 ID и Parquet, состояние TF-IDF, winner_C 2.0,
    `final_test_used: false`) совпадает с отчётом запуска;
  - три текста `train_baseline._smoke_texts()`: positive / negative /
    neutral, label_id 2 / 0 / 1 — как в `reports/nlp/training/bundle_roundtrip.json`;
    вероятности совпадают с `train_baseline.predict_texts` (|Δp| ≤ 1e-12);
  - бот, `tools/predict_text.py`, `predict` и `predict_batch` дают одинаковый
    результат;
  - полная validation (10 396 строк) через `predict_batch`: метки совпадают с
    `train_baseline.predict_texts`, |Δp| ≤ 1e-12, хеш предсказаний
    `a293a880…` = `validation_predictions_sha256` N02, матрица ошибок и
    macro-F1 совпадают;
  - порядок классов `label_mapping`, `classes_`, вероятностей —
    negative=0/neutral=1/positive=2; копии принятого комплекта с
    переставленным `label_mapping` и с обратным порядком `classes_` (со своим
    отчётом, чтобы проверка не останавливалась на SHA-256) отклонены.
- `CLASSIFIER_BACKEND=baseline python tools/evaluate_classifier.py`: passed,
  10 396 строк validation, macro-F1 0,7180146428367195, accuracy
  0,8893805309734514, матрица ошибок
  `[[2484,195,146],[117,193,126],[251,315,6569]]` совпала с отчётом N02,
  усечённых текстов нет.
- Независимо от `src.classifier`: `tools/report_nlp_baseline.py` (функция
  `evaluate`, без записи отчётов) воспроизвёл все поля
  `reports/nlp/baseline_metrics.json`, кроме `evaluation_versions.platform`
  (обучение — macOS arm64, проверка — Linux x86_64); 40 разобранных ошибок
  `baseline_errors.csv` совпали по метке и решающей оценке (|Δ| ≤ 1e-15).

Локально переобученный на Linux x86_64 комплект (SHA-256 `fa490d90…`)
не является принятым: оптимизация LR на другой платформе остановилась в
другой точке (C=2: 105 итераций вместо 146, macro-F1 0,719980). Общий
интерфейс отклоняет его до `joblib.load`
(`Baseline bundle checksum differs from run report`), тест принятого
baseline пропускается с указанием SHA-256.

Строгая загрузка baseline проверена тестами на маленьких настоящих
TF-IDF + LR комплектах с собственным отчётом запуска
(`tests/test_classifier.py`, `tests/test_evaluate_classifier.py`):
SHA-256 до десериализации, отказ при `training={}`, некорректных SHA-256/ID,
числе строк, `final_test_used`, пересечении выборок, выборе не по validation,
расхождении с параметрами модели, состоянием TF-IDF или отчётом.

## Ошибки комплекта

Отсутствующий или изменённый комплект любого backend приводит к
`ModelUnavailableError` с путём и причиной; для отсутствующего RuBERT указана
команда `python tools/fetch_transformer.py`. `tools/predict_text.py`
завершается с кодом 1, бот отвечает сообщением о недоступности классификации.

## Независимые проверки

`tools/verify_transformer.py` и `tools/report_nlp_baseline.py` не используют
`src.classifier`: они воспроизводят результаты T06 и N02 независимо от
проверяемого кода интерфейса.
