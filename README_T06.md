# T06 — дообучение RuBERT

`src/train_transformer.py` дообучает **весь** `cointegrated/rubert-tiny2` и новую
трёхклассовую голову. Checkpoint выбирается только по validation macro-F1;
при точном равенстве сохраняется более ранний шаг. Финальный test не открывается.
Порядок классов: **negative=0, neutral=1, positive=2**.

## Воспроизведение

Команды выполняются из корня репозитория. Зависимости устанавливаются только
в локальное окружение. Фактические версии окружения записаны в логах и метаданных.
Для этого запуска использован Python 3.12.3 на Linux CPU; исходные отчёты baseline
получены в другом окружении. Данные, их порядок и контрольные суммы одинаковы.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install numpy==2.3.3 pandas==2.3.2 pyarrow==24.0.0 scipy==1.15.3 scikit-learn==1.7.2 joblib==1.5.2 transformers==5.5.0 tokenizers==0.22.2 psutil==7.1.0 pytest==8.4.2
.venv/bin/python tools/restore_nlp_splits.py
.venv/bin/python -m src.train_transformer --config configs/transformer.json --check-data
HF_HOME=.cache/huggingface .venv/bin/python -m src.train_transformer --config configs/transformer.json --pilot
HF_HOME=.cache/huggingface .venv/bin/python -m src.train_transformer --config configs/transformer.json
.venv/bin/python -m src.train_transformer --verify-bundle models/transformer
.venv/bin/python -m pytest tests/test_train_transformer.py tests/test_verify_transformer.py -q
HF_HUB_OFFLINE=1 .venv/bin/python tools/verify_transformer.py --full-validation --pretrained-check
```

Первое скачивание модели требует доступа к публичному Hugging Face Hub. Токен
не нужен и не передаётся. Модель и токенизатор привязаны к revision
`e8ed3b0c8bbf4fb6984c3de043bf7d2f4e5969ae` (тот же commit, что в T05).
Вызов `--verify-bundle` использует `local_files_only=True`.
Для всего набора тестов проекта дополнительно нужны пакеты из `requirements.txt`
(`aiogram`, `faster-whisper`, `ctranslate2`, `av`).

### Проверка на отдельной машине

Нужны Linux, Python 3.12 (исходный запуск и повторная проверка — 3.12.3; на 3.13.7
из `.python-version` тесты T06 не прогонялись), около 3 ГБ свободной RAM и 3 ГБ диска. `sudo` не требуется:
всё ставится в локальное `.venv` внутри клона.

```bash
git clone https://github.com/ksdergach/ru-review-sentiment-bot.git && cd ru-review-sentiment-bot
git checkout vds/issue-8
python3 -m venv .venv
.venv/bin/python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -r requirements.txt   # torch уже установлен, версия совпадает
.venv/bin/python tools/restore_nlp_splits.py           # train/validation из NLP_dataset/, test не открывается
HF_HUB_OFFLINE=1 .venv/bin/python -m pytest -q
.venv/bin/python tools/verify_transformer.py --no-bundle   # доказательства и данные без весов
# после восстановления комплекта (см. «Хранение и перенос комплекта»):
.venv/bin/python tools/transformer_bundle.py restore --archive <путь>/t06_rubert_bundle.tar
HF_HUB_OFFLINE=1 .venv/bin/python tools/verify_transformer.py --full-validation
```

`tools/verify_transformer.py` не импортирует `src/*`. Контрольные суммы, упорядоченные
ID/label, метрики из матриц ошибок и правило выбора реализованы в нём заново.
Комплект загружается напрямую через `AutoModelForSequenceClassification` и
`AutoTokenizer` с `local_files_only=True`. Результат выводится в stdout, при `--output` —
в указанный файл. Закоммиченный `reports/transformer/verification.json` скрипт не
перезаписывает: это исторический результат прежней версии проверки на исходном
комплекте, полученный 20.10.2026. Ключи:
`--full-validation` заново прогоняет все 10 396 validation-текстов и сверяет матрицу
ошибок победителя; `--pretrained-check` сравнивает энкодер победителя с закреплённой
pretrained-ревизией из локального кэша HF и требует отличий в каждом блоке;
`--no-bundle` проверяет только данные и журналы, а статус `passed_without_bundle`
явно означает, что веса не проверялись. Без комплекта обычный запуск завершается
с кодом 2, а не «успехом».

Скрипт отказывается перезаписывать непустые каталоги отчётов и весов. Для
повторения создайте копию JSON с новыми `report_dir` и `model_dir`, например
`reports/transformer_repeat` и `models/transformer_repeat`. Принимается только
относительный путь внутри проекта. Веса должны быть исключены из Git.
Копия конфига автоматически сохраняется в отчёт и комплект модели.

## Данные и обработка

Восстановление T03 воспроизводит принятые train/validation по ID и проверяет
побайтовые SHA-256 Parquet из отчёта baseline; новая разбивка не создаётся.
Исходные Parquet сохраняются. Файл test при восстановлении не открывается.

Train берётся в порядке `reports/baseline/train_ids.csv`: **76 971** строка.
Validation: **10 396** строк. До токенизации проверяются контрольные суммы T03,
Parquet, количество строк, упорядоченные пары ID/label, язык, mapping, отсутствие
пересечений ID, нормализованных текстов и movie ID между train и validation.
Общий `split_ids.csv` содержит ID всех частей; строки вне train/validation
отбрасываются, тестовые тексты не загружаются. Оптимизатор получает только train.
Веса классов вычисляются только по train: `N / (3 * count[class])`.

В обе модели передаются те же исходные train-тексты. Baseline использует
word TF-IDF с lowercase, униграммами/биграммами, без усечения; RuBERT использует
штатный WordPiece tokenizer без дополнительной ручной нормализации.
Сохраняется начало текста: `truncation=True`, `truncation_side=right`,
`max_length=64`, включая специальные токены; padding до максимальной длины
в текущем batch. Это означает, что длинные отзывы учитываются частично.
Один и тот же `encode()` применяется при обучении, validation и локальном
предсказании. Проверки нормализованных текстов служат только поиску утечек.

## Бюджет и выбор checkpoint

Выполнены два коротких ресурсных pilot-запуска и одна конфигурация с полной
эпохой. Pilot обучается только на train, не оценивает validation и отбрасывает
веса; основной запуск заново загружает pretrained модель с тем же seed.
Логи pilot сохранены, их метрики не участвуют в выборе.

| Назначение | batch | max_length | шаги | seed | precision |
|---|---:|---:|---:|---:|---|
| Первый ресурсный pilot | 16 | 128 | 3 | 42 | float32 |
| Второй ресурсный pilot | 64 | 64 | 3 | 42 | float32 |
| Основное обучение | 64 | 64 | 1203 (1 эпоха) | 42 | float32 |

Первый pilot хранится в `reports/transformer_pilot_128/pilot/` (архив перенесён
из первоначального `report_dir`; исходный конфиг сохранён). Оценка чистого
времени эпохи по нему — 8246 секунд. Второй pilot находится в
`reports/transformer/pilot/`: оценка 2683 секунды, peak RSS 1977 MiB.
Укороченный лимит выбран **по ресурсам, до просмотра validation**. Это
ограниченный CPU-эксперимент; он не устанавливает потолок качества RuBERT.

Оба pilot выполнены **более ранней версией скрипта**: в их `run_metadata.json`
`script_sha256 = 964ce3…`, а закоммиченный код и основной запуск — `c31da4…`.
Поэтому pilot-журналы нельзя побайтно воспроизвести текущим кодом. Они служат только
оценкой времени и памяти и в выборе checkpoint не участвуют. Основной запуск
соответствует закоммиченному скрипту, это проверяет `training_script_checksum`.

Основной запуск: AdamW, постоянный learning rate `2e-5`, weight decay `0.01`,
clip grad norm `1.0`, balanced train class weights, 2 CPU threads, детерминированные
операции PyTorch. Все 29 194 707 параметров обучаются. Shuffle без возвращения,
`drop_last=False`: за эпоху каждый принятый train-ID используется ровно один раз.
Запланированы оценки после шагов **602** и **1203**; каждая использует полный
validation. Иных конфигураций по качеству не искали.

Реальные результаты и победитель: [validation_summary.md](reports/transformer/validation_summary.md).
Машиночитаемая таблица: `reports/transformer/validation_results.csv`.
Отсутствие преимущества над baseline допустимо; знак разности macro-F1
сохраняется без дополнительных поисков. Это сравнение **по validation**;
финальную оценку на test задача T06 не выполняет.

## Артефакты и проверки

- `reports/transformer/training_log.jsonl`: реальные шаги, loss, время, память,
  validation и свидетельство обновления головы, первого и последнего слоёв
  энкодера (градиенты и разность весов до/после первого optimizer step).
- `reports/transformer/config.json`, `data_version.json`, `loading_info.json`,
  `run_metadata.json`, `winner_config.json`: конфиг, версия данных, полнота
  загрузки pretrained энкодера, победитель, версии библиотек и SHA-256 файлов.
- `reports/transformer/bundle_roundtrip.json`: совпадение всех validation
  предсказаний после локальной загрузки и ответы на три фиксированных текста
  из `src/train_baseline.py::_smoke_texts()`; тексты в отчёте заменены SHA-256.
- `models/transformer/`: `model.safetensors`, tokenizer, HF config,
  `training_config.json`, `label_mapping.json`, `metadata.json`. Комплект
  локальный, исключён из Git. Перенос: см. «Хранение и перенос комплекта».
- `artifacts/transformer/`: локальные консольные журналы, исключённые из Git.

`train_seconds` измеряет forward/backward/optimizer, с синхронизацией CUDA
при соответствующем устройстве; токенизация и validation в него не входят.
`elapsed_seconds` включает загрузку модели, обучение, validation, сохранение и
проверку загрузки; восстановление и предварительная сверка данных в него не входят. Память: RSS текущего процесса и Linux VmHWM (peak RSS),
CUDA peak allocated/reserved при доступной CUDA; на CPU поля CUDA равны null.

Регрессионные тесты используют небольшой случайный BERT и синтетические
тексты без сети. Они проверяют защиту данных, обновление слоёв, выбор по
macro-F1, порядок классов, усечение и сохранение/загрузку. Результат этих
тестов не является результатом обучения RuBERT. Исправление прежнего падения
`test_end_to_end_training_updates_encoder_and_head` — сравнение метрики после
CSV с `pytest.approx`, поскольку парсинг decimal менял последний бит float.

## Поведение при прерывании

До завершения обучения отсутствие `run_metadata.json` и события `complete`
в основном журнале означает незавершённый запуск. Промежуточный победитель
сохраняется после каждой оценки validation. Это комплект для предсказания,
а не checkpoint для продолжения AdamW: состояние оптимизатора не сохраняется.
При прерывании сохраняйте прежние журналы и каталог модели; повторите ту же
конфигурацию с новыми каталогами вывода. Не смешивайте результаты разных запусков.

Скрипт обучения записывает комплект поверх промежуточного победителя без
временного каталога. Если запись прервётся, комплект может оказаться смешанным.
Такой комплект не пройдёт сверку с `bundle_files_sha256`: её выполняют
`tools/transformer_bundle.py` и `tools/verify_transformer.py`. Атомарную запись и вынос
`clip_grad_norm` в конфиг в `src/train_transformer.py` намеренно **не** добавляли.
Любое изменение скрипта меняет `script_sha256`, и принятый запуск перестал бы
соответствовать коду. Атомарность обеспечена при переносе: `restore` распаковывает
архив во временный каталог, проверяет его и устанавливает одним `rename`.
`clip_grad_norm = 1.0` зафиксирован в коде и описан выше.

## Хранение и перенос комплекта

Веса не помещаются в Git. Источник истины для комплекта —
`reports/transformer/run_metadata.json::bundle_files_sha256` (7 файлов).

```bash
# на машине, где есть комплект: детерминированный tar + <архив>.sha256
.venv/bin/python tools/transformer_bundle.py check
.venv/bin/python tools/transformer_bundle.py pack --out <хранилище>/t06_rubert_bundle.tar
# на машине проверяющего:
.venv/bin/python tools/transformer_bundle.py restore --archive <хранилище>/t06_rubert_bundle.tar
HF_HUB_OFFLINE=1 .venv/bin/python tools/verify_transformer.py --full-validation
```

`pack` не архивирует комплект, если набор файлов или хотя бы одна сумма отличаются
от метаданных запуска. `restore` проверяет SHA-256 архива и каждого файла, допускает
только ожидаемые обычные файлы (без ссылок и `..`) и не перезаписывает непустой
каталог. Хранилище — GitHub Release asset этого репозитория или общий диск
команды. Публикация архива выполняется только с разрешения владельца репозитория.

**Состояние на 20.10.2026.** Исходный каталог обучения на VDS удалён. Комплект с
`model.safetensors = 57d96aea…` в доступных рабочих каталогах не найден, во внешнее
хранилище он не выгружался. Пока архив не найдётся, C4 нельзя подтвердить на
исходном checkpoint. Повторное обучение той же командой займёт около 37 минут на
2 CPU и потребует около 2 ГБ RAM. Оно даст **новый** комплект, который сохраняется в
отдельные `reports/transformer_repeat/` и `models/transformer_repeat/` и сверяется
с исходными метриками. Совпадение SHA-256 весов детерминированные алгоритмы
PyTorch не гарантируют, его нужно подтвердить фактически.

Полный локальный regression suite: **478 passed**, фактический вывод —
`reports/transformer/test_results.txt` (абсолютные пути к `.venv` в предупреждениях
заменены на относительные; результаты не менялись). После исправлений по ревью:
**495 passed**, вывод в `reports/transformer/test_results_review_fix.txt`.
Добавлены тест правого усечения на различимых токенах и реальная загрузка
локального энкодера через `load_pretrained` без подмены, в том числе отказ на
неполном энкодере. Добавлено сравнение весов победителя из `model.safetensors`
с исходными и тесты независимого верификатора и переноса комплекта. В
end-to-end тестах подменяется только источник pretrained-модели, это не
интеграция с RuBERT. Первоначальные два падения тестов речи
устранены установкой уже закреплённых в проекте пакетов faster-whisper,
ctranslate2 и av в `.venv`; исходный код речи не менялся.

## Фактический результат 20.10.2026

Основной запуск завершился с кодом 0: 1203 optimizer steps, ровно 76 971
train-пример за полную эпоху. Победил checkpoint 1203 с validation macro-F1
**0,667824**, checkpoint 602 — 0,655146. Baseline: **0,718015**; разность
**−0,050191**. В этом ограниченном CPU-эксперименте преимущества нет.
Качество на финальном test не измерялось.

Время обучения: **2036,37 s**, общее время с оценками и загрузкой комплекта:
**2211,43 s**. Peak RSS: **1890,47 MiB**. CUDA не использовалась.
Все validation-предсказания после загрузки сохранённого победителя совпали.
На трёх заранее фиксированных текстах получены top-1 ID **[2, 0, 2]**, а столбцы
вероятностей соответствуют **[negative, neutral, positive]**. Третий пример
(обычный фильм без сильных впечатлений) классифицирован как positive:
P(neutral)=0,368508, P(positive)=0,375531. Этот ответ сохранён без замены текста;
правильный порядок классов не означает безошибочную классификацию.

Прежняя версия `tools/verify_transformer.py` в среде обучения завершилась с кодом 0.
Она выполнила 33 проверки, включая загрузку комплекта; результат сохранён в
`reports/transformer/verification.json`. Эта версия использовала функции
`src/train_transformer.py`, поэтому подтверждала внутреннюю согласованность, а не
независимую проверку.

Повторная проверка 20.10.2026 выполнена новой, независимой версией в чистом клоне
и отдельном `.venv`: `verify_transformer.py --no-bundle` дал 32 успешные проверки.
Подтверждены побайтовые SHA-256 восстановленных Parquet и T03, упорядоченные ID
train и validation, совпадающие с baseline, отсутствие пересечений и test-ID,
SHA скрипта и конфига, бюджет 1203 шага, точки validation 602/1203, метрики из
матриц ошибок, правило выбора и разность с baseline.

**Открытый блокер C4:** исходного комплекта весов в доступной среде нет (см. «Хранение
и перенос комплекта»). Загрузку и предсказания на исходном checkpoint повторно
проверить нельзя, пока не найден архив или не согласовано повторное обучение.
Это локальные автоматические проверки, а не оценка качества и не финальная test-оценка.
