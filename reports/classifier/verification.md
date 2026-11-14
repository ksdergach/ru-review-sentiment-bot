# T07: проверка общего интерфейса

Реализованы baseline и RuBERT через `src.classifier.predict` (бот, CLI),
`src.classifier.predict_batch` (оценка) и фабрику `create_classifier`.
Обязательный результат: label, label_id, confidence, probabilities в порядке
negative/neutral/positive и truncated; baseline возвращает truncated false.
`predict` и `predict_batch` используют одну загрузку, обработку и проверку
результата. Пустые, нестроковые и (для интерактивного ввода) превышающие
2000 символов тексты отклоняются до загрузки модели.

Проверка выполнена 07–08.11.2026 на результате слияния `master` (`2c47aa5`, T06)
с изменениями T07; исправления по ревью проверены на `master` `b2423db`
(после слияния PR №35);
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
  ними, попыток обращения нет. Загрузки считаются как приращение внутри
  каждого теста, поэтому результат не зависит от порядка запуска (проверены
  исходный, обратный и порядок «конкурентная загрузка → первая загрузка»).
  - три текста `train_baseline._smoke_texts()`: positive / negative /
    positive, label_id 2 / 0 / 2, confidence 0,974920 / 0,961005 / 0,375531;
    на том же входе вероятности интерфейса точно равны
    `train_transformer.predict_texts`; с `reports/transformer/bundle_roundtrip.json`
    (запись на машине обучения) — |Δp| ≤ 1e-6;
  - один текст и тот же текст в пакете с другими дополняются до разной длины,
    и операции float32 суммируют в другом порядке. Измерено на всех 10 396
    текстах validation (один текст против пакета оценки): |Δp| до 8,9e-7 при
    1 потоке CPU и до 1,1e-6 при 4 потоках, метки и `truncated` совпадают.
    Поэтому допуск сравнения одиночного и пакетного вызова — 2e-6 (на трёх
    фиксированных текстах при 1 потоке наблюдается 1,5e-7). Тест проходит при
    1, 2 и 4 потоках;
  - бот, `tools/predict_text.py` и `predict_batch` дают те же label,
    label_id, truncated; вероятности `predict_batch` совпадают с `predict`
    в пределах 2e-6;
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

- `tests/test_classifier_accepted_baseline.py` — 10 passed. Запреты сетевых
  соединений и обучения (`fit`/`fit_transform` у `CountVectorizer`,
  `TfidfVectorizer`, `LogisticRegression`; `run_training`,
  `fit_vectorizer_on_train`, `save_bundle`) установлены модульной фикстурой до
  первой загрузки, попыток обращения нет; комплект загружен один раз, в том
  числе при конкурентной первой загрузке (4 потока), и не загружается повторно
  при переиспользовании. Результат не зависит от порядка запуска тестов.
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
    отчётом, чтобы проверка не останавливалась на SHA-256) отклонены;
  - копия с `text_processing.tfidf.lowercase=True` и `lowercase=False` у
    vectorizer (со своим отчётом) отклонена: `Baseline TF-IDF parameters
    differ from text_processing.tfidf: lowercase`; без проверки она отвечала
    бы neutral вместо positive на «ОТЛИЧНЫЙ ФИЛЬМ».
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

Параметры, которые читает инференс, сверяются с объявленными в комплекте:
все параметры `TfidfVectorizer` (в том числе `analyzer`, `lowercase`,
`ngram_range` — список в JSON и tuple в scikit-learn, `min_df`, `max_df`,
`max_features`, `sublinear_tf`, `norm`, `use_idf`, `smooth_idf`,
`token_pattern`, а также `binary`, `strip_accents` и другие значения по
умолчанию) с `text_processing.tfidf`; параметры внутреннего
`TfidfTransformer`, которым scikit-learn 1.7.2 применяет `sublinear_tf`, IDF и
`norm` при `transform`; параметры `LogisticRegression` (`C`, `solver`,
`multi_class`, `class_weight`, …) с `logistic_regression`. Ожидаемые объекты
строятся функциями обучения `build_vectorizer` и `build_classifier`, сравнение
учитывает тип значения. Отказ проверен на 23 несовместимых вариантах:
22 в `test_inference_parameters_must_match_declared_configuration` и 1 в
`test_declared_tfidf_change_is_rejected_even_if_vectorizer_matches`; у каждого
изменённого комплекта собственный отчёт с совпадающим SHA-256. Проверка
находится в `src/classifier.py`: `src/train_baseline.py` не изменялся, так
как его SHA-256 входит в принятый отчёт N02.

## Ошибки комплекта

Отсутствующий или изменённый комплект любого backend приводит к
`ModelUnavailableError` с путём и причиной; для отсутствующего RuBERT указана
команда `python tools/fetch_transformer.py`. `tools/predict_text.py`
завершается с кодом 1, бот отвечает сообщением о недоступности классификации.

## Независимые проверки

`tools/verify_transformer.py` и `tools/report_nlp_baseline.py` не используют
`src.classifier`: они воспроизводят результаты T06 и N02 независимо от
проверяемого кода интерфейса.

## Проверка после исправления порядка валидации

Оценочный адаптер сначала загружает проверенный комплект модели, затем сверяет
SHA-256 validation до декодирования Parquet. После чтения проверяются ID и число
строк, и только затем вызывается predict_batch. Три регрессионных сценария
запрещают чтение текстов и/или предсказания до соответствующих проверок.

После объединения с master:
- полный набор на macOS / Python 3.13: 601 passed, 1 skipped;
- пропущен модуль точного комплекта baseline N02: локально установлен принятый
  комплект T04 с собственным run_metadata.json, а не файл N02;
- принятый RuBERT установлен из опубликованного архива, SHA-256 архива и всех
  семи файлов совпали; тесты принятого RuBERT выполнены;
- через исправленный evaluate_classifier на всех 10 396 validation-примерах
  воспроизведены матрицы ошибок и macro-F1: baseline T04 — 0,7180146428367195,
  RuBERT T06 — 0,6678240259861936.

Обучение и финальная test-оценка не выполнялись.

## Исправления по независимому ревью

Проверено 08.11.2026 и повторно 13.11.2026 на `master` `b2423db` с
исправлениями; Linux x86_64, CPU, принятые комплекты baseline N02 (SHA-256
`389a768b…`) и RuBERT T06.

| Замечание | На `b2423db` без исправлений | С исправлениями |
|---|---|---|
| Параметры TF-IDF/LR не сверялись с объявленными | 23 варианта из `tests/test_classifier.py` приняты; те же 22 изменения на копиях N02 со своим отчётом: 19 меняют параметры N02 и приняты (`lowercase=False`: positive → neutral) | 23 варианта отклонены с ожидаемой причиной; на копиях N02 19 отклонены с именем параметра |
| Допуск 1e-7 одиночного и пакетного вызова RuBERT | при 1 потоке CPU: 6 passed, 1 failed | 7 passed при 1 (3 запуска), 2 и 4 потоках |
| Зависимость тестов от порядка запуска | «конкурентная загрузка → первая загрузка»: 1 passed, 1 error для каждого backend | 2 passed; обратный порядок — baseline 10, RuBERT 7 passed |

На копиях N02 три изменения из 22 (`ngram_range=(1, 2)`, `min_df=2`, `C=2.0`)
совпадают с параметрами N02 и комплект не меняют, поэтому принимаются обоими
загрузчиками; в синтетическом комплекте теста (`ngram_range=[1, 1]`,
`min_df=1`, `C=1.0`) они отклоняются. Проверка 22 изменений на копиях N02
выполнена разово; в наборе тестов на N02 — сценарий `lowercase`
(`test_tfidf_parameters_differing_from_declaration_are_rejected`).

- `HF_HUB_OFFLINE=1 python -m pytest -q`: 634 passed, 0 failed, 0 skipped;
  так же при 1 потоке CPU и в обратном порядке тестов. Без моделей —
  617 passed, 2 skipped (модули принятых комплектов).
- `tools/evaluate_classifier.py` (с исправленным порядком проверок): baseline —
  macro-F1 0,7180146428367195, RuBERT — 0,6678240259861936, матрицы ошибок
  совпали; `tools/verify_transformer.py` — passed, 53 проверки;
  `tools/report_nlp_baseline.py` (`evaluate`, без записи) — все поля
  `baseline_metrics.json`, кроме платформы.

Обучение и финальная test-оценка не выполнялись.
