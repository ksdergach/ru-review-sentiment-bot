# T04 — TF-IDF + Logistic Regression baseline

Решение задачи **[T04 · Н1] Обучить TF-IDF + Logistic Regression и сохранить baseline**.
Результат: воспроизводимый трёхклассовый baseline (`negative=0`, `neutral=1`, `positive=2`)
с выбором параметров **только по validation**.

## Что делает решение

- Использует только подготовленные T03 `train.parquet` и `validation.parquet`. Финальный test
  не читается и не используется; его нет ни в конфигурации, ни в коде загрузки данных.
- Конфигурация проверяется по **строгой схеме**: любой неизвестный ключ на любом уровне — ошибка,
  а значит test нельзя добавить под другим именем (`final_test`, `evaluation.test`, `testPath`, …).
  Дополнительно отклоняются пути к данным, содержащие токен `test` (`data/processed/test.parquet`),
  абсолютные пути и пути с `..`.
- До обучения проверяет данные T03: схему и типы, все три класса в train и validation, соответствие
  `label_id` ↔ `label_name`, уникальность `record_id`, отсутствие пересечений `record_id`, `movie_id`
  и **нормализованных текстов** (NFC → casefold → невидимые символы → пробелы, как в T03) между train и validation.
- Сверяет подготовленные Parquet с T03 `audit.json` (статус `completed`, число строк и распределение
  классов) и с `split_ids.csv` (порядок и метки). Файлы из разных запусков T03 → ошибка, а не обучение.
- TF-IDF (word-level) обучается **только на train**; validation — только `transform`. Веса классов
  считаются **только по train**. Дообучения на train + validation нет.
- Для 2–3 заданных значений `C` считает на validation macro-F1, accuracy, precision/recall/F1/support
  каждого класса (порядок классов фиксирован `[0, 1, 2]`, редкий neutral всегда в отчёте) и матрицу ошибок победителя.
- Выбирает победителя автоматически: максимальный validation macro-F1, при точном равенстве — меньший `C`.
- Сохраняет TF-IDF, Logistic Regression, mapping и порядок классов, правила обработки, версии библиотек и
  метаданные в один `joblib`-комплект, затем загружает его и требует **те же классы** (и вероятности с допуском 1e-12)
  на фиксированных текстах **и те же предсказания на всём validation**.
- Не скрывает предупреждения: `ConvergenceWarning` фиксируется по каждому запуску; остальные предупреждения
  записываются в отчёт и выводятся повторно.
- GPU не нужен.

## Зависимости

`numpy`, `pandas`, `pyarrow` (чтение Parquet), `scikit-learn`, `scipy`, `joblib`, `pytest` (только для тестов).
Установка общего `requirements.txt` описана в [README.md](README.md); версии, с которыми запускалось обучение,
записываются в `reports/baseline/run_metadata.json`.

## Файлы

Скопируйте в корень репозитория (структура каталогов сохраняется):

```text
src/train_baseline.py
configs/baseline.json
tests/test_train_baseline.py
README_T04.md
```

К моменту запуска T03 должна создать (с одного и того же запуска T03):

```text
data/processed/train.parquet
data/processed/validation.parquet
reports/data_prep/audit.json
reports/data_prep/split_ids.csv
```

`split_ids.csv` — единый файл T03 с ID и метками всех сплитов (без текстов). T04 сразу отбрасывает строки
не из train/validation и не использует их.

## 1. `.gitignore`

Модели не коммитятся. В `.gitignore` должны быть строки:

```gitignore
models/
*.joblib
```

Проверка (команда должна вывести правило, а не пустой результат):

```powershell
git check-ignore -v models/baseline/tfidf_logreg_baseline.joblib
```

Если после обучения комплект не игнорируется Git, `train_baseline.py` напечатает предупреждение.

## 2. Быстрая проверка без корпуса (smoke-test)

```powershell
python src/train_baseline.py --smoke-test
```

Ожидается первая строка `T04 smoke test passed.` и JSON с `"passed": true`, `"bundle_loadable": true`,
`"probability_class_ids": [0, 1, 2]`. Smoke-test не читает Parquet и CSV: маленький цикл
`train → save → load → predict` на синтетических текстах. Его вывод прикладывается к Issue как результат smoke-теста.

## 3. Автоматические тесты

```powershell
python -m pytest -q tests/test_train_baseline.py
```

Тесты проверяют инварианты T04: невозможность подключить test (в том числе под другим именем и через значения путей),
строгую схему конфигурации, значения `C`, контракт данных T03, нормализованные пересечения train/validation,
соответствие `audit.json` и `split_ids.csv`, TF-IDF и веса классов только из train (независимая пересборка
и сравнение состояния/коэффициентов), выбор победителя, метрики и порядок классов, общий train/subset,
отсутствие текстов и абсолютных путей в отчётах, воспроизводимость, сохранение/загрузку (в том числе в новом
процессе и с испорченными комплектами). Тест реального Parquet пропускается, если не установлен `pyarrow`.

## 4. Полное обучение

Одна команда из корня репозитория:

```powershell
python src/train_baseline.py --config configs/baseline.json
```

По умолчанию используется весь подготовленный `data/processed/train.parquet`. По данным `audit_report.md` T03
(конвейер t03.2) это 76 971 строк train (21 482 / 3 290 / 52 199) и 10 396 строк validation (2 825 / 436 / 7 135);
при другом `audit.json` числа будут другими, но сверка с ним обязательна.

**Общий train/subset для RuBERT.** Если T03 создал `reports/data_prep/train_subsample_ids.csv`
(подвыборка включена в конфигурации T03), укажите этот файл в `configs/baseline.json`:
`"train_ids_path": "reports/data_prep/train_subsample_ids.csv"`. T04 применит **ровно эти ID в указанном
порядке** и заново проверит три класса. Если аудит T03 сообщает о включённой подвыборке, а `train_ids_path` не
задан, T04 остановится. Validation при этом не сокращается и не балансируется. В любом случае результат
фиксируется в `reports/baseline/train_ids.csv` (только `train_order,record_id,label_id`) и в метаданных
(`train_selection_source`, `train_is_subset`, `train_ids_sha256`, `train_ids_file_sha256`).
Этот же список ID использует RuBERT.

## 5. Параметры baseline

```text
word-level TF-IDF
  lowercase = true, ngram_range = (1, 2), min_df = 2, max_df = 0.98, max_features = 100000
  sublinear_tf = true, norm = l2, token_pattern = (?u)\b\w\w+\b

Logistic Regression (multinomial, L2)
  C = 0.5 / 1.0 / 2.0, solver = lbfgs, max_iter = 1000, tol = 1e-4
  class_weight = balanced, рассчитанный только по train
  seed = 42
```

Это ограниченный baseline-поиск по 3 значениям регуляризации, а не большой перебор.

**Что видит модель.** `review_text` подаётся в TF-IDF без ручной очистки (ни лемматизации, ни удаления
пунктуации). Стандартный `token_pattern` сам отбрасывает пунктуацию, эмодзи и односимвольные слова — это свойство
токенизатора word-level TF-IDF, оно зафиксировано в конфигурации и в комплекте модели.

## 6. Правило выбора модели

Единственный содержательный критерий — **максимальный validation macro-F1**. При точном равенстве — меньший `C`
(технический детерминированный tie-break, не второй критерий качества). Accuracy и F1 neutral выводятся, но выбор не определяют.
Победитель не зависит от порядка значений `C`. Test не читается и не используется.

**Сходимость.** Запуск, не достигший `max_iter`, помечается в `validation_results.csv`, `validation_summary.md`
и метаданных, а при выборе такого победителя печатается предупреждение. Такие запуски не исключаются автоматически
(это был бы скрытый второй критерий): решение — увеличить `max_iter` в конфигурации и повторить обучение.

## 7. Что создаётся после обучения

В Git можно сохранить (ID без текстов и агрегированные результаты):

```text
reports/baseline/
├── validation_results.csv     # таблица 2–3 запусков: C → все метрики
├── validation_summary.md      # таблица, победитель и его отрыв, матрица ошибок
├── winner_config.json         # конфигурация победителя (без таймингов; повторяется байт-в-байт)
├── run_metadata.json          # версии библиотек, суммы данных, seed, тайминги, sha256 комплекта
├── bundle_roundtrip.json      # проверка save → load на фиксированных текстах и на validation
└── train_ids.csv              # train_order,record_id,label_id (порядка нескольких МБ для полного train)
```

Локально (не в Git):

```text
models/baseline/tfidf_logreg_baseline.joblib
```

`run_metadata.json` фиксирует: seed; размеры и распределения классов; SHA-256 подготовленных Parquet,
`audit.json`, `split_ids.csv` и **исходных** Parquet из аудита T03; SHA-256 точного порядка train/validation ID;
размер словаря и SHA-256 состояния TF-IDF (словарь + IDF); веса классов; победителя; версии Python, NumPy,
pandas, scikit-learn, SciPy, joblib, PyArrow; SHA-256 конфигурации и `train_baseline.py`; Git commit, если доступен;
SHA-256 комплекта и признак `bundle_git_ignored`; `final_test_used: false`. Абсолютные пути в отчёты не попадают.

`bundle_roundtrip.json` — результат проверки после сохранения (это не smoke-test из раздела 2).

### Просмотр отчётов в Windows PowerShell

Отчёты сохраняются в UTF-8 без BOM — так их корректно показывают Git, GitHub и редакторы.
Windows PowerShell 5.1 по умолчанию читает файл в кодировке системы, поэтому `Get-Content`
без параметров выводит кириллицу как нечитаемый набор символов. Кодировку нужно указать явно:

```powershell
Get-Content reports\baseline\validation_summary.md -Encoding UTF8
```

Искажается только вывод в консоли; сам файл при этом корректен. В PowerShell 7 параметр не нужен.

## 8. Комплект модели

Комплект содержит: TF-IDF, Logistic Regression, mapping `negative/neutral/positive`, порядок классов `[0, 1, 2]`,
правила обработки (`manual_normalization = none`, `truncation = null`, параметры TF-IDF), параметры
классификатора, версии библиотек и метаданные обучения. Он самодостаточен: загружается обычным `joblib.load`
без импорта `src/train_baseline.py`. Функции `load_bundle` и `predict_texts` в `train_baseline.py`
дополнительно проверяют структуру комплекта (типы, обученность, совпадение размера словаря и коэффициентов, порядок классов).

**Важно.** `joblib` основан на pickle: загружайте только комплекты, созданные вашей командой. Для стабильных
результатов загружайте комплект в окружении с теми же версиями scikit-learn/NumPy (они записаны в комплекте).

## 9. Что коммитить

Можно: `src/train_baseline.py`, `configs/baseline.json`, `tests/test_train_baseline.py`, `README_T04.md`, `reports/baseline/*`.

Нельзя: `models/` (обученные модели), `data/processed/`, секреты, любые выгрузки текстов отзывов.

```powershell
git status
git diff --cached --check
```

## 10. Критерии готовности T04

T04 можно считать завершённой только после реального запуска на подготовленных данных T03:

1. smoke-test проходит;
2. автоматические тесты проходят;
3. полное обучение запускается одной командой без ошибок;
4. получена таблица 2–3 validation-запусков (`validation_results.csv`);
5. победитель выбран по validation macro-F1 и это объяснено в `validation_summary.md`;
6. final test не использовался (`final_test_used: false`);
7. все три класса сохранены, редкий neutral присутствует в метриках;
8. зафиксирован общий train/subset ID (`train_ids.csv`, хэши в метаданных);
9. комплект загружается и даёт те же ответы (`bundle_roundtrip.json`);
10. в Git уходят конфигурации, ID и агрегированные результаты; модель остаётся локально.
