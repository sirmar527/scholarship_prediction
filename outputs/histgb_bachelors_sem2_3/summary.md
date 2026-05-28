# Scholarship Prediction - HistGradientBoosting (bachelor sem 2 -> 3)

**Дата:** 2026-05-28 08:51

## Данные

- Train: 789 пар (789 студентов)
- Test: 198 пар (198 студентов)
- Признаков: 13
- Итераций до остановки: 146

## Baselines (test set)

| Baseline | Accuracy | F1 macro | Описание |
|---|---|---|---|
| majority | 57.6% | 0.365 | Always predict 0 |
| persistence | 81.3% | 0.810 | Predict next clean = current clean (had_clean_current_sem) |
| rule_no_blocking | 81.3% | 0.810 | No blocking grades & no retakes -> clean next sem |
| gpa_4.0 | 76.3% | 0.762 | GPA >= 4.0 -> clean next sem |
| gpa_4.2 | 82.3% | 0.822 | GPA >= 4.2 -> clean next sem |

## Метрики HistGradientBoosting

| Метрика | Значение |
|---|---|
| Accuracy | 80.3% |
| F1 (macro) | 0.800 |
| ROC-AUC | 0.885 |

## Classification report

```
                precision    recall  f1-score   support

Не чисто (N+1)      0.850     0.798     0.824       114
   Чисто (N+1)      0.747     0.810     0.777        84

      accuracy                          0.803       198
     macro avg      0.799     0.804     0.800       198
  weighted avg      0.807     0.803     0.804       198
```

## Confusion matrix (binary)

```
                Пред: не чисто  Пред: чисто
Факт: не чисто              91           23
Факт: чисто                 16           68
```

## Variant B - 4 класса

```
                      Пред: 1  Пред: 2  Пред: 3  Пред: 4
Факт: Чисто->чисто         63        5        0        0
Факт: Чисто->провал        18        3        0        0
Факт: Провал->чисто         0        0        5       11
Факт: Провал->провал        0        0        5       88
```

## Per-class accuracy

- **Чисто->чисто**: 92.7%
- **Чисто->провал**: 14.3%
- **Провал->чисто**: 31.2%
- **Провал->провал**: 94.6%

## Feature importance (permutation, ROC-AUC)

- `gpa_overall`: 0.8321
- `n_blocks`: 0.1085
- `n_retakes`: 0.0310
- `had_clean_current_sem`: 0.0165
- `n_subjects`: 0.0119
- `any_block`: 0.0000
- `sem_num`: 0.0000
- `min_grade`: 0.0000
- `any_retake`: 0.0000
- `std_grade`: 0.0000
- `share_5`: 0.0000
- `share_3`: 0.0000
- `share_zachet`: 0.0000
