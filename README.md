# Scholarship Prediction

**EN.** Machine-learning pipelines that forecast whether a Russian-university student will keep a clean academic record in the next semester — which under standard rules entitles them to a stipend the semester after. Built on anonymised grade-level data conforming to **ГОСТ Р 70946-2023, Приложение 8**. Multiple model approaches are kept side-by-side (XGBoost, CatBoost, sklearn, math-analysis baselines) and a separate clustering tool reconstructs ФГОС block structure from grade data.

---

## Что это

Проект предсказывает, **сохранит ли студент «чистую успеваемость» в следующем семестре** (без блокирующих оценок и пересдач). По правилам в РФ чистый семестр N+1 даёт право на стипендию в N+2 — то есть модель работает как **раннее предупреждение за один семестр вперёд**: признаки берутся из сем. N, целевая переменная — состояние в сем. N+1.

Помимо предсказательной модели в репозитории есть кластеризатор учебных планов (`classify_fgos.py`), реконструирующий структуру ФГОС из данных об оценках.

## Структура проекта

```
scholarship_pred/
├── data/                                  # входные xlsx
│   ├── ГОСТ Р 70946-2023-Приложение-8_sorted.xlsx
│   └── output_bachelors_*.xlsx
├── outputs/                               # результаты моделей
│   ├── xgb/             ← каноническая модель (XGBoost, после патча)
│   ├── catboost/
│   ├── catboost_single/
│   ├── catboost_split/
│   ├── sklearn/
│   ├── baseline/
│   ├── baseline_binary/
│   └── fgos/            ← кластеры учебных планов
├── scholarship_predict_xgb.py             # каноническая XGB-модель
├── scholarship_predict.py                 # CatBoost
├── scholarship_predict_single.py          # CatBoost, одна модель на всё (Option B)
├── scholarship_predict_split.py           # CatBoost, две модели: удержание + получение (Approach C)
├── scholarship_predict_sklearn.py         # sklearn
├── baseline_matanaliz.py                  # baseline: только мат. анализ
├── baseline_matanaliz_binary.py           # бинарный вариант того же baseline
├── classify_fgos.py                       # кластеризация учебных планов
├── CHANGELOG.md                           # история патчей XGB-модели
└── README.md
```

## Данные

Источник — выгрузка по структуре **ГОСТ Р 70946-2023, Приложение 8** (анонимизированные оценки).

| Показатель | Значение |
|---|---|
| Строк (оценок) | 155 295 |
| Уникальных студентов | 5 933 |
| Учебных планов | 345 |
| Завершивших программу | 1 852 (1 565 бакалавров, 287 специалистов) |
| Пар (сем. N, сем. N+1) | 11 508 |

Фильтрация в каноническом XGB-пайплайне:
- Только очная форма обучения
- Типы ведомостей: Основная, Перезачёт, Пересдача, Пересдача с комиссией
- Пары соседних семестров одного студента

## Целевая переменная

`target_clean_next_sem` — бинарный признак: будет ли в следующем семестре **чистая успеваемость**, то есть:
- нет блокирующих оценок (двоек, неявок без уважительной причины),
- нет пересдач.

По стандартным академическим правилам в РФ чистый сем. N+1 даёт право на стипендию в N+2, поэтому модель работает как ранний прогноз стипендии на два семестра вперёд.

## Признаки (13)

| Признак | Описание |
|---|---|
| `sem_num` | Номер текущего семестра (1–10) |
| `n_subjects` | Количество предметов в семестре |
| `n_blocks` | Количество блокирующих оценок |
| `n_retakes` | Количество пересдач |
| `any_block` | Был ли хотя бы один блок |
| `any_retake` | Была ли хотя бы одна пересдача |
| `gpa_overall` | Средний балл по числовым оценкам |
| `min_grade` | Минимальная оценка |
| `std_grade` | Стандартное отклонение оценок |
| `share_5` / `share_3` | Доля пятёрок / троек среди числовых оценок |
| `share_zachet` | Доля «зачётов» среди всех контрольных |
| `had_clean_current_sem` | Был ли чистым текущий семестр |

Подробное обоснование изменений признаков — в [CHANGELOG.md](CHANGELOG.md).

## Метрики (каноническая XGB-модель, после патча)

| Метрика | Значение |
|---|---|
| Accuracy | 83.40 % |
| F1 macro | 0.832 |
| ROC-AUC | 0.916 |

Сильнейший baseline даёт ~82.4 % — модель уверенно его превосходит. Сравнение «до и после» патча целиком — в [CHANGELOG.md](CHANGELOG.md).

## Модели

Каждый скрипт — самостоятельный пайплайн, читает данные из `data/`, пишет в свою папку в `outputs/`.

| Скрипт | Подход | Выход |
|---|---|---|
| `scholarship_predict_xgb.py` | **XGBoost (канонический)** — early stopping, разбиение по студентам | `outputs/xgb/` |
| `scholarship_predict.py` | CatBoost — мультиклассовый | `outputs/catboost/` |
| `scholarship_predict_single.py` | CatBoost — одна модель на все пары (Option B) | `outputs/catboost_single/` |
| `scholarship_predict_split.py` | CatBoost — две модели: удержание стипендии + получение (Approach C) | `outputs/catboost_split/` |
| `scholarship_predict_sklearn.py` | sklearn — для сравнения | `outputs/sklearn/` |
| `baseline_matanaliz.py` | Baseline: только оценки по матанализу | `outputs/baseline/` |
| `baseline_matanaliz_binary.py` | Бинарный вариант baseline | `outputs/baseline_binary/` |
| `classify_fgos.py` | Кластеризация учебных планов в направления ФГОС | `outputs/fgos/` |

## Запуск

Все скрипты ищут xlsx-файл в `data/` по умолчанию. Можно переопределить через `--data` и `--output`:

```bash
# Каноническая XGB-модель
python scholarship_predict_xgb.py

# С нестандартными путями
python scholarship_predict_xgb.py --data path/to/file.xlsx --output путь/к/результатам/

# Другие модели
python scholarship_predict.py
python scholarship_predict_sklearn.py
python baseline_matanaliz.py

# Кластеризация ФГОС
python classify_fgos.py
```

После прогона `classify_fgos.py` рекомендуется вручную проверить `outputs/fgos/block_mapping.csv` и затем запускать предсказательные модели с ключом `--blocks block_mapping.csv` (если поддерживается).

## Зависимости

```
pandas
numpy
scikit-learn
xgboost          # для scholarship_predict_xgb.py
catboost         # для scholarship_predict*.py (кроме xgb и sklearn)
openpyxl         # чтение xlsx
```

Установка:
```bash
pip install pandas numpy scikit-learn xgboost catboost openpyxl
```

## История

Все изменения канонической XGB-модели задокументированы в [CHANGELOG.md](CHANGELOG.md). Состояние репозитория до большой реструктуризации (май 2026) сохранено в git-теге `v1.0-snapshot`.
