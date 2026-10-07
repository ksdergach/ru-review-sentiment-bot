# T04 — validation summary

Выбор конфигурации выполнен **только по validation macro-F1**. Финальный test в T04 не читается и не используется.

| Запуск | C | Веса классов | macro-F1 | F1 negative | F1 neutral | F1 positive | Сошёлся | Fit, s | Predict val, s |
|---|---:|---|---:|---:|---:|---:|---|---:|---:|
| run_1_C_0.5 | 0.5 | balanced_train | 0.704446 | 0.860764 | 0.324718 | 0.927856 | да | 3.113 | 0.003 |
| run_2_C_1 | 1 | balanced_train | 0.716173 | 0.870011 | 0.343122 | 0.935386 | да | 4.294 | 0.003 |
| run_3_C_2 **(winner)** | 2 | balanced_train | 0.718533 | 0.874956 | 0.340088 | 0.940554 | да | 5.771 | 0.003 |

Победитель: **run_3_C_2** (C = 2), validation macro-F1 = 0.718533.
Ближайший конкурент: run_2_C_1 (C = 1), macro-F1 = 0.716173; разница +0.002360.

Правило выбора: максимальный validation macro-F1; при точном равенстве — меньший C. Других критериев нет.
Accuracy — дополнительная метрика; из-за редкого neutral основной критерий — macro-F1.
Время predict не включает TF-IDF transform (см. `validation_transform_seconds` в run_metadata.json).

## Матрица ошибок победителя (validation)

Строки — истинный класс, столбцы — предсказанный; порядок: negative(0), neutral(1), positive(2).

| истина \ прогноз | negative | neutral | positive |
|---|---:|---:|---:|
| negative | 2484 | 194 | 147 |
| neutral | 120 | 193 | 123 |
| positive | 249 | 312 | 6574 |
