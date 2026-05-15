# Scholarship Prediction — XGBoost

**Дата:** 2026-05-15 07:29

## Данные

- Train: 9,177 пар (4,505 студентов)
- Test: 2,331 пар (1,127 студентов)
- Признаков: 13
- Лучшая итерация: 350

## Baselines (test set)

| Baseline | Accuracy | F1 macro | Описание |
|---|---|---|---|
| majority | 55.4% | 0.357 | Always predict 0 |
| persistence | 74.9% | 0.749 | Predict next = prev_scholarship |
| rule_no_blocking | 83.3% | 0.831 | No blocking grades & no retakes → scholarship |
| gpa_4.0 | 80.8% | 0.808 | GPA ≥ 4.0 → scholarship |
| gpa_4.2 | 82.8% | 0.826 | GPA ≥ 4.2 → scholarship |

## Метрики XGBoost

| Метрика | Значение |
|---|---|
| Accuracy | 83.7% |
| F1 (macro) | 0.836 |
| ROC-AUC | 0.920 |

## Classification report

```
               precision    recall  f1-score   support

Без стипендии      0.883     0.813     0.846      1292
Со стипендией      0.788     0.866     0.825      1039

     accuracy                          0.837      2331
    macro avg      0.836     0.839     0.836      2331
 weighted avg      0.841     0.837     0.837      2331
```

## Confusion matrix (binary)

```
           Пред: без  Пред: со
Факт: без       1050       242
Факт: со         139       900
```

## Variant B — 4 класса

```
                       Пред: 1  Пред: 2  Пред: 3  Пред: 4
Факт: Была→сохранил        843       34        0        0
Факт: Была→потерял         195      227        0        0
Факт: Не было→получил        0        0       57      105
Факт: Не было→нет            0        0       47      823
```

## Per-class accuracy

- **Была→сохранил**: 96.1%
- **Была→потерял**: 53.8%
- **Не было→получил**: 35.2%
- **Не было→нет**: 94.6%

## Feature importance

- `n_blocks`: 0.4025
- `share_3`: 0.2250
- `gpa_overall`: 0.1134
- `n_retakes`: 0.0485
- `min_grade`: 0.0421
- `prev_scholarship`: 0.0344
- `any_retake`: 0.0336
- `any_block`: 0.0236
- `share_5`: 0.0196
- `sem_num`: 0.0192
- `share_zachet`: 0.0137
- `std_grade`: 0.0124
- `n_subjects`: 0.0120
