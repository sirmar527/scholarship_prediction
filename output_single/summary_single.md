# Scholarship Prediction — Single-Model Pipeline (Option B)

**Дата:** 2026-05-14 13:47

## Подход

Одна модель CatBoostClassifier на все пары, `prev_scholarship` как обычный признак.

Лучшая итерация: 160

## Данные

- Train: 9,177 пар (4,505 студентов)
- Test: 2,331 пар (1,127 студентов)
- Признаков: 12

## Baselines (test set)

| Baseline | Accuracy | F1 macro | Описание |
|---|---|---|---|
| majority | 55.4% | 0.357 | Always predict 0 |
| persistence | 74.9% | 0.749 | Predict next = prev_scholarship |
| rule_nofail | 83.3% | 0.831 | No fails & no retakes → scholarship |
| gpa_4.0 | 80.8% | 0.808 | GPA ≥ 4.0 → scholarship |
| gpa_4.2 | 82.0% | 0.817 | GPA ≥ 4.2 → scholarship |

## Метрики модели

| Метрика | Значение |
|---|---|
| Accuracy | 84.1% |
| F1 (macro) | 0.840 |
| ROC-AUC | 0.919 |

## Classification report

```
               precision    recall  f1-score   support

Без стипендии      0.889     0.815     0.850      1292
Со стипендией      0.791     0.873     0.830      1039

     accuracy                          0.841      2331
    macro avg      0.840     0.844     0.840      2331
 weighted avg      0.845     0.841     0.841      2331
```

## Confusion matrix (binary)

```
           Пред: без  Пред: со
Факт: без       1053       239
Факт: со         132       907
```

## Variant B — 4 класса

```
                       Пред: 1  Пред: 2  Пред: 3  Пред: 4
Факт: Была→сохранил        848       29        0        0
Факт: Была→потерял         201      221        0        0
Факт: Не было→получил        0        0       59      103
Факт: Не было→нет            0        0       38      832
```

## Per-class accuracy

- **Была→сохранил**: 96.7%
- **Была→потерял**: 52.4%
- **Не было→получил**: 36.4%
- **Не было→нет**: 95.6%

## Feature importance

- `n_retakes`: 19.5
- `sem_num`: 16.8
- `share_5`: 11.9
- `share_3`: 9.5
- `n_fails`: 9.1
- `n_subjects`: 7.6
- `gpa_overall`: 7.2
- `std_grade`: 6.1
- `any_retake`: 4.4
- `min_grade`: 3.3
- `prev_scholarship`: 3.2
- `any_fail`: 1.3
