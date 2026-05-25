# Scholarship Prediction — HistGradientBoosting

**Дата:** 2026-05-25 14:18

## Данные

- Train: 9,169 пар (4,507 студентов)
- Test: 2,343 пар (1,127 студентов)
- Признаков: 13
- Итераций до остановки: 248

## Baselines (test set)

| Baseline | Accuracy | F1 macro | Описание |
|---|---|---|---|
| majority | 57.5% | 0.365 | Always predict 0 |
| persistence | 82.3% | 0.819 | Predict next clean = current clean (had_clean_current_sem) |
| rule_no_blocking | 82.3% | 0.819 | No blocking grades & no retakes → clean next sem |
| gpa_4.0 | 79.3% | 0.793 | GPA ≥ 4.0 → clean next sem |
| gpa_4.2 | 81.8% | 0.815 | GPA ≥ 4.2 → clean next sem |

## Метрики HistGradientBoosting

| Метрика | Значение |
|---|---|
| Accuracy | 83.1% |
| F1 (macro) | 0.830 |
| ROC-AUC | 0.914 |

## Classification report

```
                precision    recall  f1-score   support

Не чисто (N+1)      0.890     0.807     0.846      1347
   Чисто (N+1)      0.768     0.864     0.813       996

      accuracy                          0.831      2343
     macro avg      0.829     0.836     0.830      2343
  weighted avg      0.838     0.831     0.832      2343
```

## Confusion matrix (binary)

```
                Пред: не чисто  Пред: чисто
Факт: не чисто            1087          260
Факт: чисто                135          861
```

## Variant B — 4 класса

```
                     Пред: 1  Пред: 2  Пред: 3  Пред: 4
Факт: Чисто→чисто        784        0        0        0
Факт: Чисто→провал       200        3        0        0
Факт: Провал→чисто         0        0       77      135
Факт: Провал→провал        0        0       60     1084
```

## Per-class accuracy

- **Чисто→чисто**: 100.0%
- **Чисто→провал**: 1.5%
- **Провал→чисто**: 36.3%
- **Провал→провал**: 94.8%

## Feature importance (permutation, ROC-AUC)

- `gpa_overall`: 0.5639
- `n_retakes`: 0.1685
- `sem_num`: 0.0708
- `share_zachet`: 0.0574
- `n_blocks`: 0.0321
- `share_5`: 0.0286
- `had_clean_current_sem`: 0.0283
- `share_3`: 0.0281
- `n_subjects`: 0.0123
- `std_grade`: 0.0100
- `min_grade`: 0.0001
- `any_block`: 0.0000
- `any_retake`: 0.0000
