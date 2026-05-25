# Scholarship Prediction

**EN.** Binary classifier predicting whether a Russian-university student will maintain a clean academic record (no blocking grades, no retakes) in the next semester. Dataset: anonymised grade-level data conforming to ГОСТ Р 70946-2023, Приложение 8.

## Задача

Для пары (семестр N, семестр N+1) одного студента предсказать, будет ли семестр N+1 чистым (без блокирующих оценок и без пересдач). По стандартным академическим правилам РФ чистый семестр N+1 даёт право на стипендию в семестре N+2; модель работает как ранний прогноз на один семестр вперёд.

## Данные

| | |
|---|---|
| Строк (оценок) | 155 295 |
| Студентов | 5 933 |
| Учебных планов | 345 |
| Пар (сем. N, сем. N+1) | 11 508 |

Фильтрация: только очная форма; типы ведомостей {Основная, Перезачёт, Пересдача, Пересдача с комиссией}; соседние семестры одного студента.

## Целевая переменная

`target_clean_next_sem` принимает значения 0 или 1: 1, если в семестре N+1 нет блокирующих оценок и нет пересдач.

## Признаки (13)

| Признак | Описание |
|---|---|
| `sem_num` | Номер семестра (1-10) |
| `n_subjects` | Число дисциплин в семестре |
| `n_blocks`, `any_block` | Количество и наличие блокирующих оценок |
| `n_retakes`, `any_retake` | Количество и наличие пересдач |
| `gpa_overall` | Средний балл по числовым оценкам |
| `min_grade` | Минимальная оценка |
| `std_grade` | Стандартное отклонение оценок |
| `share_5`, `share_3` | Доли пятёрок и троек среди оцениваемых дисциплин |
| `share_zachet` | Доля пас-фейл дисциплин в нагрузке |
| `had_clean_current_sem` | Был ли чистым текущий семестр |

## Модель

`scholarship_predict_histgb.py` обучает `sklearn.ensemble.HistGradientBoostingClassifier` с встроенным ранним остановом. Разбиение 80/20 по студентам (один студент не попадает одновременно в train и test). Permutation-importance используется как мера важности признаков. Результаты сохраняются в `outputs/histgb/`.

## Метрики

| Метрика | Значение |
|---|---|
| Accuracy | 83.14 % |
| F1 (macro) | 0.830 |
| ROC-AUC | 0.914 |

Test: 2 343 пары, train: 9 169 пар. Сильнейший baseline на той же выборке: accuracy ~82.3 % (правило «следующий семестр будет таким же, как текущий»). Полные таблицы (confusion matrix, разбивка по семестрам, 4-классовая транзиция Чисто/Провал × Чисто/Провал) находятся в `outputs/histgb/summary.md`.

## Запуск

```bash
pip install pandas numpy scikit-learn openpyxl
python scholarship_predict_histgb.py
```

Флаги: `--data path.xlsx`, `--output dir/`.

## Структура

```
data/                            входные xlsx
outputs/histgb/                  результаты модели
scholarship_predict_histgb.py
```
