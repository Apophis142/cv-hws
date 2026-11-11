# EuroSAT RGB: результаты

| Вариант | Лучшая эпоха | Val accuracy | Test accuracy | Test macro F1 |
|---|---:|---:|---:|---:|
| head_only | 5 | 0.9368 | 0.9356 | 0.9329 |
| fine_tuned | 8 | 0.9723 | 0.9719 | 0.9708 |

Изменение test accuracy после fine-tuning: +3.63 п.п.

Оба варианта проверены на одном test. Веса выбраны только по validation.

Матрицы ошибок: [head_only](head_only/confusion_matrix.png), [fine_tuned](fine_tuned/confusion_matrix.png).
