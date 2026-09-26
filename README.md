# ru-review-sentiment-bot

Учебный проект: голосовой отзыв Telegram → Whisper → текстовый классификатор
→ `negative=0`, `neutral=1`, `positive=2`.
[План](docs/team_plan.md) и [доска](https://github.com/users/ksdergach/projects/2).

В репозитории уже есть подготовка данных (T03), обучение baseline (T04)
и проверка ресурсов (T05). Полное обучение RuBERT, общий интерфейс инференса
и Telegram-бот выполняются в следующих задачах. Установка библиотек не означает,
что бот уже готов.

## Установка

Проверяемая версия — **CPython 3.13.7** (`.python-version`); она совпадает
с версией в сохранённых отчётах T04/T05. Нужен рабочий Git в PATH:
`git --version` должен завершаться успешно. Команды ниже выполняются
из корня этого репозитория. Для установки нужен интернет; для быстрых проверок
не нужны GPU, токен Telegram, веса моделей и финальные тестовые записи.

Создайте новое окружение. Если `.venv` уже содержит другой проект или версии,
создайте отдельное окружение, а не устанавливайте поверх него.

macOS / Linux:

```bash
python3.13 --version
python3.13 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.13 --version
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

В обоих случаях вывод версии должен быть `3.13.7`.
Если активация запрещена политикой PowerShell, вызывайте
`.\.venv\Scripts\python.exe` вместо `python`; менять системную политику не нужно.

**Выберите один вариант установки PyTorch**, затем установите общие зависимости.
Версия `torch==2.10.0` в requirements допускает сборки `+cpu` и `+cu126`.

macOS Apple Silicon:

```bash
python -m pip install -r requirements.txt
```

Linux / Windows без NVIDIA GPU — сначала CPU-сборка, чтобы не загружать
ненужные CUDA-библиотеки:

```bash
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Машина команды с NVIDIA, использованная в T05:

```powershell
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Варианты сборок соответствуют [инструкциям PyTorch 2.10](https://pytorch.org/get-started/previous-versions/#v2100).
CUDA должна быть доступна для профилей T05. Общие тесты выполняются и на CPU;
они не подтверждают доступность GPU или способность обучить модель на конкретной машине.

Версии научных библиотек закреплены по отчётам T04/T05, чтобы не менять окружение
уже обученного baseline. aiogram и pytest также закреплены и проверяются импортом.
Прямые зависимости находятся в `requirements.txt`; полный снимок локальной
проверки с транзитивными зависимостями — в `reports/environment/`.
Снимок одной ОС не является универсальным lock-файлом для CUDA и всех платформ.

### Декодирование аудио

Установленный PyAV используется faster-whisper и поставляет библиотеки FFmpeg
в бинарном пакете: отдельная команда `ffmpeg` для текущего пути распознавания
не обязательна ([описание faster-whisper](https://github.com/SYSTRAN/faster-whisper#requirements)).
Для ручной конвертации записей можно дополнительно установить FFmpeg:
`brew install ffmpeg` на macOS, `sudo apt install ffmpeg` на Ubuntu
или `winget install --id Gyan.FFmpeg -e` в Windows.
Проверка такой установки — `ffmpeg -version`.

## Быстрая проверка окружения

```bash
python -m pip check
python tools/check_environment.py
python -m pytest -q
python src/train_baseline.py --smoke-test
```

`check_environment.py` сверяет Python и версии пакетов, импортирует в том числе
реальные классы BERT, Whisper и aiogram. Он не скачивает модели и не читает датасет.
Ненулевой код возврата означает несовместимую версию или ошибку импорта.
Smoke-тест baseline обучается на встроенных искусственных примерах во временной
директории; это проверка установки, не эксперимент на финальном test.

Тесты используют временные данные. Финальные текстовые метрики и 36 голосовых
отзывов для этих команд не нужны. Отчёт о выполненной проверке T01 —
[reports/environment/verification.md](reports/environment/verification.md).

## Конфигурация, секреты и пути

Происхождение, контрольные суммы и правила классов описаны в
[NLP_dataset/README.md](NLP_dataset/README.md). Воспроизвести аудит T02 и
подготовить локальную выборку для ручного просмотра:

```bash
python tools/audit_dataset.py
```

Тексты выборки сохраняются в игнорируемом `artifacts/dataset_audit/`.
Статус просмотра участником — в [reports/dataset/manual_review.md](reports/dataset/manual_review.md).

Все рабочие пути задаются относительно корня репозитория в существующих JSON.
В JSON используются прямые слеши и на Windows. Единого дублирующего конфига
поверх T03/T04/T05 нет: параметры каждого этапа находятся в его файле.

| Этап | Конфигурация | Вход | Локальный результат | Метаданные в Git |
|---|---|---|---|---|
| Подготовка T03 | `configs/data_prep.json` | `NLP_dataset/*.parquet` | `data/processed/`, тексты для аудита в `artifacts/data_prep/` | `reports/data_prep/` |
| Baseline T04 | `configs/baseline.json` | Подготовленные train/validation | `models/baseline/` | `reports/baseline/` |
| Проверка T05 | `configs/resource_check.json` | Train, метаданные T04, отдельное отладочное аудио | `artifacts/resource_check/` | `reports/resources/` |
| Окружение T01 | `.python-version`, `requirements.txt` | Чистая `.venv/` | Установленные пакеты | `reports/environment/` |

Папки `src/`, `configs/`, `tests/` и `reports/` уже содержат реальные
модули, конфиги, проверки и отчёты. Заглушки обученных моделей не добавляются.

`.env.example` — шаблон имён переменных без секретов. При необходимости
скопируйте его в локальный `.env`. **Текущие скрипты не загружают этот файл
автоматически.** `TELEGRAM_BOT_TOKEN` зарезервирован для будущего бота и для
T03/T04/T05 не нужен. Способ чтения токена будет подключён в T09.

Для локального кэша моделей перед командами T05 можно установить:

```bash
export HF_HOME=.cache/huggingface
export HF_HUB_DISABLE_TELEMETRY=1
```

В PowerShell:

```powershell
$env:HF_HOME = ".cache/huggingface"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
```

Это выбирает новый каталог кэша; для `--offline` модели должны быть уже загружены
именно в него. Пустой новый кэш нельзя использовать для офлайн-проверки.

## Точки входа

Сначала можно посмотреть справку без запуска обучения:

```bash
python src/prepare_data.py --help
python src/train_baseline.py --help
python src/check_resources.py --help
```

Подготовка данных и обучение — отдельные команды:

```bash
python src/prepare_data.py --config configs/data_prep.json
python src/train_baseline.py --config configs/baseline.json
```

Эти команды создают локальные данные/модель и обновляют отчёты своего этапа.
Проверяйте изменения отчётов перед коммитом. T03 проверяет целостность всех
сплитов; T04 обучается на train и выбирает параметры по validation, не оценивая
финальный test. Подробности: [T03](README_T03.md), [T04](README_T04.md).

Проверка ресурсов после получения локальных данных T03 и артефактов T04:

```bash
python src/check_resources.py --preflight --config configs/resource_check.json --profile primary
```

Это проверка условий запуска, не реальный шаг обучения.
Профили `primary` и `reserve` в текущем конфиге предназначены для Windows-машины
команды с CUDA. На Mac или машине без CUDA их отказ ожидаем; проверка импортов
T01 от этого не зависит. Для другого оборудования нужен отдельно проверенный
конфиг, а не заявление о пройденной T05.

Реальные замеры с отладочным аудио и условия использования `--confirm-debug-audio`
описаны в [T05](README_T05.md). Финальные 36 записей для них не используются.

Запуск Telegram и полное обучение RuBERT будут отдельными точками входа задач
T09 и T06. Сейчас таких команд в репозитории нет; повторный запуск обучения
не должен становиться частью запуска бота.

## Что хранится в Git

Исходные `NLP_dataset/*.parquet` уже отслеживаются и сохраняются без изменений.
Конфиги, решения по дубликатам, контрольные суммы, ID без текстов, код и
агрегированные метрики также сохраняются.

Новые записи кладите в `artifacts/audio/`, результаты с текстами —
в `artifacts/`, модели — в `models/`, производные данные — в `data/processed/`.
Кэш `.cache/`, локальные окружения, `.env` и его варианты, аудио, видео,
сериализованные модели, токенизаторы и checkpoints исключены через `.gitignore`.
JSON с текстами вне этих локальных каталогов автоматически не исключается:
не сохраняйте расшифровки в `reports/`.

Проверка правил без создания файлов:

```bash
git check-ignore -v .env .env.local artifacts/audio/debug.wav models/baseline/model.joblib checkpoints/run/model.safetensors .cache/huggingface/tokenizer.json data/processed/train.parquet
git check-ignore --no-index NLP_dataset/train-00000-of-00001.parquet .env.example configs/data_prep.json reports/environment/verification.md
```

Первая команда должна вывести правило для каждого локального артефакта.
Вторая должна не вывести ничего и завершиться с кодом 1: перечисленные исходники,
шаблон, конфиг и отчёт не игнорируются. Уже отслеживаемый файл не скрывается
добавлением в `.gitignore`; перед коммитом проверяйте `git status --short`.
