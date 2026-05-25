# Patch summary — `scholarship_predict_xgb_four.py`

## What's in the patch

| # | Change | Type | Lines touched |
|---|---|---|---|
| 1 | Pass/fail classified by `ВидКонтроля == "Зачет"` (not by grade label) | Bug fix | `build_features` |
| 2 | Dedup priority: pass/fail `Зачет` moved to last; `Курсовая`/ГАК before it | Bug fix | new `DEDUP_PRIORITY` constant + `build_features` |
| 3 | `has_retake` includes `ИтоговаяОтметка == "Неявка"` signal | Bug fix | `load_data` |
| 4 | `share_5` / `share_3` computed from numeric `grade_num`, not text label | Bug fix | `build_features` |
| 5 | Removed `fillna(0)` for numeric features; let XGBoost handle NaN natively (kept `std_grade` fillna at 0 since single-graded-subject std is genuinely undefined) | Bug fix | `_prepare_features` |
| 6 | Include `Перезачет` rows (transferred credit) with `is_scholarship_blocking=0` and `has_retake=0` (Option 1 from analysis) | Coverage | `load_data` |
| 7 | Include orphan `Пересдача` / `Пересдача с комиссией` rows (where no Основная partner exists); drop the 23 duplicate ones | Coverage | `load_data` |
| 8 | Removed `sem_num == 1` clean-record override (was masking real first-semester failures) | Semantics | `build_features` |
| 9 | Renamed `scholarship` → `clean_record`, `target_scholarship` → `target_clean_next_sem`, `prev_scholarship` → `had_clean_current_sem` throughout | Naming | feature pipeline, pairs, predictions CSV, model card, summary, human report |
| 10 | Updated docstrings, comments, and Russian-language report to describe the actual task (clean record forecast) and its relation to scholarship under Russian rules | Docs | top docstring, FEATURE_DESCRIPTIONS, `_save_human_report` |

## Metric comparison (same seed, same hyperparameters)

| Metric | Original | Patched | Δ |
|---|---|---|---|
| Accuracy | 83.44% | 83.40% | −0.04 pp |
| F1 macro | 0.834 | 0.832 | −0.001 |
| ROC-AUC | 0.920 | 0.916 | −0.004 |

Aggregate metrics are essentially unchanged. The patch fixes feature-level
correctness for the affected populations (a few percent of rows in total),
not the bulk task. The model still beats the strongest baseline by ~1 pp.

## What changed materially in the outputs

**Records processed**: 155,295 → 155,426 (+131 from `Перезачет` + 2 orphan
retake rows, partially offset by 23 deduplicated `Пересдача` rows).

**Student-semesters**: 17,506 → 17,510 (+4 from formerly-vanishing students
whose entire semester was `Перезачет`).

**Transition pairs**: 11,508 → 11,512 (+4).

**4-class distribution** (because `sem_num == 1` override is gone, ~1,555
first-semester students with blocking grades correctly move from class 1/2
to class 3/4):

| Class | Original | Patched |
|---|---|---|
| 1 (Чисто→чисто) | 36.4% | 34.2% |
| 2 (Чисто→провал) | 18.1% | 8.2% |
| 3 (Провал→чисто) | 6.9% | 9.1% |
| 4 (Провал→провал) | 38.6% | 48.5% |

This is a more honest reflection of student trajectories: the original code
was incorrectly classifying many "never had clean record" students as "had
clean record then lost it."

## Output schema changes

`predictions.csv` columns renamed:

| Original | Patched |
|---|---|
| `prev_scholarship` | `had_clean_current_sem` |
| `prob_scholarship` | `prob_clean_next_sem` |
| `pred_scholarship` | `pred_clean_next_sem` |
| `actual_scholarship` | `actual_clean_next_sem` |
| `risk_score` | `risk_score` (kept, semantics clarified in code comment) |

`model_card.json` `target` description updated to reflect the actual task.

`summary.md`, `report_human.md` text updated.

## What did NOT change

- Model hyperparameters
- Train/val/test split logic and seeds
- Output file paths and names (`predictions.csv`, `summary.md`, etc.)
- File name `scholarship_model.json` — kept since it's still a
  scholarship-related model in the broader sense

## Open assumptions documented but not changed

- The "any retake → no scholarship" policy is unchanged. The script
  currently disqualifies students who retook to improve a passing grade
  (1,207 such rows). If the institution's policy allows grade-improvement
  retakes, swap `has_retake` for a policy-aware version.
- `Перезачет` grades are treated as never-blocking (Option 1). If
  institutional regulations actually count transferred-credit `Удовлетворительно`
  marks as blocking, switch to Option 2 by removing the `perezachet_mask`
  override in `load_data`.
- `Неявка` severity is kept uniform (grade_num = 2). The "commission no-show
  is worse" rule from Russian regulations exists but cannot be reliably
  identified from this data alone.

## Files in this delivery

- `scholarship_predict_xgb_four_patched.py` — full patched script
- `scholarship_predict_xgb_four.diff` — unified diff against the original
- `CHANGES.md` — this document
