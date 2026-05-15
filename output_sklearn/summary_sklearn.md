# Scholarship Prediction — Scikit-Learn Multi-Model Pipeline

**Дата:** 2026-05-14 15:00

## Подход

Сравнение нескольких моделей через scikit-learn API, `prev_scholarship` как обычный признак.

## Данные

- Train: 9,177 пар (4,505 студентов)
- Test: 2,331 пар (1,127 студентов)
- Признаков: 12

## Model Comparison

| Model | Accuracy | F1 macro | ROC-AUC | Best? |
|---|---|---|---|---|
| LogisticRegression | 84.3% | 0.842 | 0.917 |  |
| RandomForest | 82.9% | 0.829 | 0.918 |  |
| GradientBoosting | 82.8% | 0.828 | 0.918 | ✓ |
| XGBoost | 83.7% | 0.836 | 0.918 |  |
| LightGBM | 83.2% | 0.832 | 0.918 |  |

## Baselines (test set)

| Baseline | Accuracy | F1 macro | Описание |
|---|---|---|---|
| majority | 55.4% | 0.357 | Always predict 0 |
| persistence | 74.9% | 0.749 | Predict next = prev_scholarship |
| rule_no_blocking | 83.3% | 0.831 | No blocking grades & no retakes → scholarship |
| gpa_4.0 | 80.8% | 0.808 | GPA ≥ 4.0 → scholarship |
| gpa_4.2 | 82.0% | 0.817 | GPA ≥ 4.2 → scholarship |

## Метрики лучшей модели (GradientBoosting)

| Метрика | Значение |
|---|---|
| Accuracy | 82.8% |
| F1 (macro) | 0.828 |
| ROC-AUC | 0.918 |

## Classification report

```
               precision    recall  f1-score   support

Без стипендии      0.884     0.795     0.837      1292
Со стипендией      0.773     0.870     0.819      1039

     accuracy                          0.828      2331
    macro avg      0.829     0.832     0.828      2331
 weighted avg      0.835     0.828     0.829      2331
```

## Confusion matrix (binary)

```
           Пред: без  Пред: со
Факт: без       1027       265
Факт: со         135       904
```

## Variant B — 4 класса

```
                       Пред: 1  Пред: 2  Пред: 3  Пред: 4
Факт: Была→сохранил        850       27        0        0
Факт: Была→потерял         208      214        0        0
Факт: Не было→получил        0        0       54      108
Факт: Не было→нет            0        0       57      813
```

## Per-class accuracy

- **Была→сохранил**: 96.9%
- **Была→потерял**: 50.7%
- **Не было→получил**: 33.3%
- **Не было→нет**: 93.5%

## Feature importance

- `min_grade`: 0.3074
- `n_blocks`: 0.2628
- `gpa_overall`: 0.1413
- `share_3`: 0.0712
- `any_block`: 0.0682
- `sem_num`: 0.0389
- `n_retakes`: 0.0326
- `share_5`: 0.0269
- `std_grade`: 0.0175
- `n_subjects`: 0.0127
- `prev_scholarship`: 0.0113
- `any_retake`: 0.0091
