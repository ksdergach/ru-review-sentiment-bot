# T13: сравнение baseline и RuBERT на одном текстовом test

Issue: #16. Зависит от T12 (#15).

**Статус: подготовительная разработка.** Инструмент и его автотесты на искусственных
данных готовы. Финальный test не открывался, модели на нём не запускались,
реальных финальных метрик нет. Финальный запуск возможен только после
утверждённой фиксации T12.

## Назначение

`src/evaluate.py` один раз оценивает обе зафиксированные модели на полном неизменном
`data/processed/test.parquet` и сохраняет агрегированные отчёты:

- accuracy, macro-F1, precision/recall/F1/support каждого класса;
- confusion matrix 3×3 для каждой модели в порядке `negative=0`, `neutral=1`, `positive=2`;
  класс без предсказаний остаётся в матрице и в macro-F1 (precision = 0);
- время загрузки, прогрева и основного прохода, batch size и характеристики среды;
- отдельный расчёт чувствительности к повторам внутри test (основной test не меняется).

Используются существующие компоненты, без второй системы классификации:

| Что | Откуда |
|---|---|
| Загрузка и проверка комплектов, предсказания | `src.classifier.create_classifier(...).predict_batch` — тот же путь, что у бота и `tools/evaluate_classifier.py` |
| Метрики и матрица ошибок | `src.train_baseline.validation_metrics`, `validation_confusion_matrix` |
| Чтение и проверка схемы test | `src.train_baseline.load_prepared_split`, `ids_sha256`, `load_t03_audit` |
| Правило точных повторов | `src.train_baseline.normalize_text_key` (правило T03) |

`tools/evaluate_classifier.py` по-прежнему работает только с validation
и отказывается от файлов с `test` в имени.

## Входные данные

| Вход | Источник |
|---|---|
| `data/processed/test.parquet` | T03 (`src/prepare_data.py`), колонки `record_id`, `review_text`, `label_id`, `label_name`, `review_language`, `movie_id` |
| `reports/data_prep/split_ids.csv`, `audit.json` | T03: ID и метки без текстов; размер test и число повторов |
| `reports/data_prep/near_duplicate_candidates.csv`, `configs/near_duplicate_decisions.csv` | T03: подтверждённые близкие пары |
| Комплекты моделей и их `run_metadata.json` | T04/N02 (baseline) и T06 (RuBERT); пути задаются фиксацией |
| Файл фиксации T12 (по умолчанию `reports/freeze.json`) | T12, ещё не создан |

## Требования к фиксации T12

Формат фиксации T12 ещё не утверждён. `src/evaluate.py` описывает **черновик** —
минимум, который нужен T13 (`FREEZE_SCHEMA = "t13-final-text-eval-input/draft-1"`).
T12 может принять его или добавить адаптер и собственные проверки в `FREEZE_CHECKS`.
Окончательная интеграция зависит от T12. Пример структуры с заглушками
(это не настоящая фиксация):

```json
{
  "schema": "t13-final-text-eval-input/draft-1",
  "status": "approved",
  "approved_by": ["<участники>"],
  "approved_at": "<дата>",
  "frozen_commit": "<40 hex>",
  "final_test_used_before_freeze": false,
  "bot_model": "baseline | rubert",
  "bot_model_selected_by": "validation_and_resources",
  "data_prep": {"audit_sha256": "<sha256>", "split_ids_sha256": "<sha256>"},
  "test": {"path": "data/processed/test.parquet", "parquet_sha256": "<sha256>",
           "rows": 9324, "ids_sha256": "<sha256 пар record_id\\tlabel_id>"},
  "models": {
    "baseline": {"model_path": "<путь>", "report_path": "<run_metadata.json>", "report_sha256": "<sha256>"},
    "rubert": {"model_path": "<путь>", "report_path": "<run_metadata.json>", "report_sha256": "<sha256>"}
  },
  "inference": {"batch_size": 64, "device": "cpu", "num_threads": null}
}
```

`test.ids_sha256` считается `src.train_baseline.ids_sha256` по `record_id` и `label_id`
и может быть получен из `split_ids.csv` без чтения текстов.

## Проверки до чтения test

Финальный запуск останавливается с кодом 2 и не пишет отчёты, если не выполнено любое условие.
Порядок важен: тексты test декодируются только после шагов 1–3.

1. **Фиксация:** файл есть, схема поддерживается, `status = approved`, указаны участники и дата,
   `final_test_used_before_freeze = false`, модель бота выбрана по validation и ресурсам;
   ровно две модели `baseline` и `rubert`; SHA-256 их отчётов, `audit.json` и `split_ids.csv`
   совпадают с записанными; batch size положителен, устройство — CPU.
   Git: код потомок `frozen_commit`, в отслеживаемых файлах нет незакоммиченных изменений.
2. **Модели:** оба комплекта загружаются и проверяются `src.classifier` (суммы, версия данных,
   `final_test_used = false`); у обеих моделей совпадают train/validation ID и размеры,
   а версия T03 совпадает с фиксацией.
3. **Файл test:** SHA-256 Parquet совпадает с фиксацией.
4. **Содержимое test:** число строк и `ids_sha256` совпадают с фиксацией, число строк —
   с `audit.json`, порядок ID и метки — с `split_ids.csv`.
5. **Сравнение:** обе модели получают одни и те же тексты в одном порядке и одинаковыми партиями;
   ID, метки, размер и версия test одинаковы для обеих строк таблицы.

## Методика измерения времени

Для каждой модели отдельно (`time.perf_counter`, CPU, `max_text_length=None` — как на validation):

- **загрузка** — проверка и загрузка комплекта через `src.classifier` до чтения test;
- **прогрев** — один вызов `predict_batch` на 3 фиксированных искусственных текстах `WARMUP_TEXTS` (не из test);
- **инференс** — весь test в исходном порядке, `predict_batch` на партиях по `inference.batch_size`;
  записываются секунды, мс на текст и тексты в секунду.

RuBERT внутри дополнительно делит партию по `batch_size` своего `training_config.json`.
`inference.num_threads` (если задан) передаётся в `torch.set_num_threads` перед загрузкой RuBERT.
В отчёт записываются Python, ОС, процессор, число ядер, объём RAM, версии библиотек
и фактическое число потоков torch. Сравнение времени корректно только для одной машины.

## Повторы внутри test

Основной расчёт выполняется на полном test. По аудиту T03 внутри test нет точных повторов
и есть 7 подтверждённых пар близких повторов (14 строк).
Правило чувствительности задано заранее: группы — точные повторы по `normalize_text_key`
плюс пары T03 с решением `duplicate`, обе записи которых в test (компоненты связности).
В каждой группе остаётся первая запись в порядке test. Число групп сверяется с `audit.json`.
Новая дедупликация и ручной разбор test не выполняются.

## Команды

Автотесты на искусственных данных (настоящий test не читается):

```bash
python -m pytest tests/test_evaluate.py -q
```

Проверка готовности без загрузки моделей и чтения test. Сейчас она завершается с кодом 2:
фиксации T12 нет.

```bash
python -m src.evaluate --check
```

Финальный запуск — **только после завершения T12**:

```bash
python tools/fetch_transformer.py                 # принятый комплект RuBERT
# получить зафиксированный комплект baseline по README и протоколу T12
# получить data/processed/test.parquet способом, записанным в T12 (T03: src/prepare_data.py),
# и убедиться, что отслеживаемые отчёты T03 не изменились: git status
python -m src.evaluate --check --freeze reports/freeze.json
python -m src.evaluate --freeze reports/freeze.json --out-dir reports/evaluation
```

Повторный запуск с той же фиксацией и тем же commit должен дать те же метрики и матрицы;
время зависит от машины.

## Формат отчётов

`reports/evaluation/` (только агрегаты и ID, без текстов отзывов):

| Файл | Содержимое |
|---|---|
| `final_metrics.json` | commit, фиксация и её SHA-256, команда, размер и суммы test, версия данных, протокол инференса, среда, метрики, матрицы, время, чувствительность к повторам |
| `final_table.csv` | строка на модель: accuracy, macro-F1, P/R/F1/support по классам, время, batch size, отметка модели бота |
| `final_report.md` | итоговая таблица и две confusion matrix в Markdown |
| `confusion_matrix_baseline.csv`, `confusion_matrix_rubert.csv` | строки — истинный класс, столбцы — предсказанный |
| `predictions.csv` | `record_id`, `label_id`, `baseline_label_id`, `rubert_label_id` |

Матрицы сохраняются как таблицы; PNG-графики не строятся (matplotlib не входит в зависимости проекта).

## Оставшиеся блокировки

- T12 (#15) не завершена: нет утверждённой фиксации, комплектов обеих моделей и протокола.
  Формат фиксации T12 может отличаться от черновика — тогда нужен адаптер.
- Принятый комплект baseline в репозитории не хранится; штатная команда его получения
  отсутствует (отмечено в T16, #14).
- Реальные финальные таблицы, матрицы и время ещё не получены; критерии #16 остаются открытыми.
- Разбор 30–50 ошибок выбранной модели (team_plan §6) в этот инструмент не входит.
