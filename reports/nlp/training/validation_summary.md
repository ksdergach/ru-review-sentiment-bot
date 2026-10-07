# T04 — validation summary

Выбор конфигурации выполнен **только по validation macro-F1**. Финальный test в T04 не читается и не используется.

| Запуск | C | Веса классов | macro-F1 | F1 negative | F1 neutral | F1 positive | Сошёлся | Fit, s | Predict val, s |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|
| run_1_C_0.5 | 0.5 | balanced_train | 0.705051 | 0.862542 | 0.324934 | 0.927678 | да | 4.534 | 0.003 |
| run_2_C_1 | 1 | balanced_train | 0.716641 | 0.869857 | 0.344402 | 0.935666 | да | 4.362 | 0.003 |
| run_3_C_2 **(winner)** | 2 | balanced_train | 0.718015 | 0.875110 | 0.338894 | 0.940040 | да | 6.871 | 0.003 |

Победитель: **run_3_C_2** (C = 2), validation macro-F1 = 0.718015.
Ближайший конкурент: run_2_C_1 (C = 1), macro-F1 = 0.716641; разница +0.001373.

Правило выбора: максимальный validation macro-F1; при точном равенстве — меньший C. Других критериев нет.
Accuracy — дополнительная метрика; из-за редкого neutral основной критерий — macro-F1.
Время predict не включает TF-IDF transform (см. `validation_transform_seconds` в run_metadata.json).

## Матрица ошибок победителя (validation)

Строки — истинный класс, столбцы — предсказанный; порядок: negative(0), neutral(1), positive(2).

| истина \ прогноз | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 2484 | 195 | 146 |
| neutral | 117 | 193 | 126 |
| positive | 251 | 315 | 6569 |
