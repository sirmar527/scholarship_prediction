# Scholarship Prediction — XGBoost

**Дата:** 2026-05-25 13:40

## Данные

- Train: 9,169 пар (4,507 студентов)
- Test: 2,343 пар (1,127 студентов)
- Признаков: 13
- Лучшая итерация: 205

## Baselines (test set)

| Baseline | Accuracy | F1 macro | Описание |
|---|---|---|---|
| majority | 57.5% | 0.365 | Always predict 0 |
| persistence | 82.3% | 0.819 | Predict next clean = current clean (had_clean_current_sem) |
| rule_no_blocking | 82.3% | 0.819 | No blocking grades & no retakes → clean next sem |
| gpa_4.0 | 79.3% | 0.793 | GPA ≥ 4.0 → clean next sem |
| gpa_4.2 | 81.8% | 0.815 | GPA ≥ 4.2 → clean next sem |

## Метрики XGBoost

| Метрика | Значение |
|---|---|
| Accuracy | 83.2% |
| F1 (macro) | 0.830 |
| ROC-AUC | 0.916 |

## Classification report

```
                precision    recall  f1-score   support

Не чисто (N+1)      0.887     0.811     0.847      1347
   Чисто (N+1)      0.771     0.860     0.813       996

      accuracy                          0.832      2343
     macro avg      0.829     0.836     0.830      2343
  weighted avg      0.838     0.832     0.833      2343
```

## Confusion matrix (binary)

```
                Пред: не чисто  Пред: чисто
Факт: не чисто            1092          255
Факт: чисто                139          857
```

## Variant B — 4 класса

```
                     Пред: 1  Пред: 2  Пред: 3  Пред: 4
Факт: Чисто→чисто        784        0        0        0
Факт: Чисто→провал       198        5        0        0
Факт: Провал→чисто         0        0       73      139
Факт: Провал→провал        0        0       57     1087
```

## Per-class accuracy

- **Чисто→чисто**: 100.0%
- **Чисто→провал**: 2.5%
- **Провал→чисто**: 34.4%
- **Провал→провал**: 95.0%

## Feature importance

- `any_block`: 0.5591
- `min_grade`: 0.2312
- `n_blocks`: 0.1495
- `had_clean_current_sem`: 0.0139
- `gpa_overall`: 0.0126
- `share_3`: 0.0120
- `n_retakes`: 0.0078
- `any_retake`: 0.0037
- `share_5`: 0.0031
- `sem_num`: 0.0025
- `share_zachet`: 0.0017
- `std_grade`: 0.0015
- `n_subjects`: 0.0014
