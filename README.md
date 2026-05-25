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

## Модели

Два канонических пайплайна, разбиение 80/20 по студентам, общий набор признаков и целевая переменная.

| Скрипт | Модель | Выход |
|---|---|---|
| `scholarship_predict_xgb.py` | XGBoost, early stopping | `outputs/xgb/` |
| `scholarship_predict_histgb.py` | sklearn HistGradientBoosting | `outputs/histgb/` |

## Метрики (XGBoost)

| Метрика | Значение |
|---|---|
| Accuracy | 83.40 % |
| F1 (macro) | 0.832 |
| ROC-AUC | 0.916 |

Сильнейший baseline на той же выборке: accuracy ~82.4 %. Полные таблицы (confusion matrix, разбивка по семестрам, 4-классовая транзиция Чисто/Провал × Чисто/Провал) находятся в `outputs/xgb/summary.md`. Метрики HistGradientBoosting появятся в `outputs/histgb/summary.md` после первого запуска.

## Запуск

```bash
pip install pandas numpy scikit-learn xgboost openpyxl
python scholarship_predict_xgb.py
python scholarship_predict_histgb.py
```

Флаги: `--data path.xlsx`, `--output dir/`.

## Структура

```
data/                            входные xlsx
outputs/
├── xgb/                         результаты XGBoost
└── histgb/                      результаты HistGradientBoosting
scholarship_predict_xgb.py
scholarship_predict_histgb.py
CHANGELOG.md                     история патчей XGBoost-пайплайна
```

Старые варианты моделей (CatBoost в трёх конфигурациях, sklearn-зоопарк, baseline по матанализу, TabPFN, кластеризатор ФГОС) и их выводы лежат в `legacy/` и `outputs/legacy/`. Обе папки добавлены в `.gitignore`.

## История

Изменения канонического пайплайна задокументированы в [CHANGELOG.md](CHANGELOG.md). Состояние до большой реструктуризации (май 2026) сохранено в git-теге `v1.0-snapshot`.
