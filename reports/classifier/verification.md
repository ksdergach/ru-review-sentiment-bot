# T07: проверка общего интерфейса

Реализованы baseline и RuBERT через `src.classifier.predict` (бот, CLI),
`src.classifier.predict_batch` (оценка) и фабрику `create_classifier`.
Обязательный результат: label, label_id, confidence, probabilities в порядке
negative/neutral/positive и truncated; baseline возвращает truncated false.
`predict` и `predict_batch` используют одну загрузку, обработку и проверку
результата. Пустые, нестроковые и (для интерактивного ввода) превышающие
2000 символов тексты отклоняются до загрузки модели.

Проверка выполнена 07.11.2026 на `master` (`2c47aa5`, T06) с изменениями T07;
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
(SHA-256 `389a768bcd108721c79ebf3a7b9dc5212ec100f5c2e1b06cf519ed076cd9bee3`)
в проверочной среде **отсутствует**, документированного источника получения
у проекта нет. Поэтому `tests/test_classifier_accepted_baseline.py`
пропущен, а согласованность путей и порядок классов на принятом baseline
**не подтверждены**.

Имеющийся локально обученный комплект (SHA-256 `fa490d90…`) не используется
как доказательство T04: общий интерфейс отклоняет его до `joblib.load`
(`Baseline bundle checksum differs from run report`), оценочный адаптер
завершается с кодом 2.

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
