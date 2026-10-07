# T04 — validation summary

Выбор конфигурации выполнен **только по validation macro-F1**. Финальный test в T04 не читается и не используется.

| Запуск | C | Веса классов | macro-F1 | F1 negative | F1 neutral | F1 positive | Сошёлся | Fit, s | Predict val, s |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|
| run_1_C_0.5 | 0.5 | balanced_train | 0.704187 | 0.862389 | 0.322709 | 0.927464 | да | 5.699 | 0.004 |
| run_2_C_1 | 1 | balanced_train | 0.715979 | 0.870318 | 0.342550 | 0.935069 | да | 7.170 | 0.003 |
| run_3_C_2 **(winner)** | 2 | balanced_train | 0.722499 | 0.875440 | 0.351706 | 0.940351 | да | 4.642 | 0.004 |

Победитель: **run_3_C_2** (C = 2), validation macro-F1 = 0.722499.
Ближайший конкурент: run_2_C_1 (C = 1), macro-F1 = 0.715979; разница +0.006520.

Правило выбора: максимальный validation macro-F1; при точном равенстве — меньший C. Других критериев нет.
Accuracy — дополнительная метрика; из-за редкого neutral основной критерий — macro-F1.
Время predict не включает TF-IDF transform (см. `validation_transform_seconds` в run_metadata.json).

## Матрица ошибок победителя (validation)

Строки — истинный класс, столбцы — предсказанный; порядок: negative(0), neutral(1), positive(2).

| истина \ прогноз | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 2488 | 196 | 141 |
| neutral | 112 | 201 | 123 |
| positive | 259 | 310 | 6566 |
