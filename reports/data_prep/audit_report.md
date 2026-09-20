# T03 — аудит подготовки данных

Отчёт генерируется детерминированно командой `python src/prepare_data.py --config configs/data_prep.json`.

- Версия конвейера: **t03.2**
- Статус запуска: **completed**

## Контрольные суммы исходных файлов

| Split | SHA-256 до | SHA-256 после | Не изменён |
|---|---|---|---|
| train | `c749cfd841f995259be122135d1c9c5da530c4de07b27e43c016a547524c32da` | `c749cfd841f995259be122135d1c9c5da530c4de07b27e43c016a547524c32da` | True |
| validation | `ecd03e961597e3df087884bc08c34f5b60a010ddb0f687550db727e1ad0863ae` | `ecd03e961597e3df087884bc08c34f5b60a010ddb0f687550db727e1ad0863ae` | True |
| test | `c5ca9a8ddf21995cc33bab64b4d8467af632a86dcda95f26a4557548ae03bf1f` | `c5ca9a8ddf21995cc33bab64b4d8467af632a86dcda95f26a4557548ae03bf1f` | True |

## Итоговые split'ы

| Split | Строк | Negative | Neutral | Positive | Фильмов |
|---|---:|---:|---:|---:|---:|
| train | 76971 | 21482 | 3290 | 52199 | 3941 |
| validation | 10396 | 2825 | 436 | 7135 | 495 |
| test | 9324 | 2619 | 411 | 6294 | 492 |

## Утечки после очистки

- Пересечения `movie_id`: `{'train__validation': 0, 'train__test': 0, 'validation__test': 0}`
- Пересечения нормализованных точных текстов: `{'train__validation': 0, 'train__test': 0, 'validation__test': 0}`

## Точная дедупликация

- removed_exact_train_vs_eval: **2**
- removed_exact_validation_vs_test: **0**
- removed_exact_within_train: **7**
- removed_exact_within_validation: **0**
- test_exact_duplicate_extra_rows: **0**
- test_exact_duplicate_groups: **0**

## Конфликты меток

- Групп «одинаковый текст, разные метки»: **0**
- Строк в таких группах: **0**
- Из них групп внутри одного split: **0**
- Из них групп между split'ами: **0**
- Конфликты не скрываются дедупликацией: полный список идентификаторов — `reports/data_prep/label_conflicts.csv` (без текстов отзывов).

## Похожие тексты

- Зафиксированный порог Jaccard char-n-gram: **0.9**
- Размер n-граммы: **3**
- Найдено кандидатов для ручного аудита: **59**
- Кандидатов БЕЗ ручного решения (не обработаны): **0**
- Подтверждено вручную как копии: **58**
- Удалено строк по подтверждённым near-duplicates: **49**
- Подтверждённых пар целиком внутри test: **7**
- Строк test, сохранённых в near-duplicate компонентах: **14**
- Компонент связности: **54**
- Компонент крупнее пары: **2**
- Пар внутри компонент БЕЗ прямого подтверждения человеком: **0**
- Удалённых строк, метка которых отличается от представителя: **0**
- Блоков, пропущенных при поиске: **0** (блоки не отбрасываются)
- Самый большой блок: **359** записей
- Пар, сопоставленных внутри блоков: **334709**
- Список кандидатов обрезан по max_candidates: **False**
- Высокое сходство само по себе не удаляет запись: удаляются только пары, отмеченные `duplicate` в файле решений.

## Артефакты

- `reports/data_prep/excluded_ids.csv`
- `reports/data_prep/split_ids.csv`
- `reports/data_prep/label_conflicts.csv`
- `reports/data_prep/undecided_candidates.csv`
- `reports/data_prep/near_duplicate_candidates.csv`
- `artifacts/data_prep/near_duplicate_review.csv`
- `reports/data_prep/audit.json`
- `reports/data_prep/audit_report.md`
