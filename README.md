# Scholarship Prediction

**EN.** Machine-learning pipelines that forecast whether a Russian-university student will keep a clean academic record in the next semester, which under standard rules entitles them to a stipend the semester after. Built on anonymised grade-level data conforming to **ГОСТ Р 70946-2023, Приложение 8**. Two canonical models are kept side-by-side: XGBoost and sklearn HistGradientBoosting.

---

## Что это

Проект предсказывает, **сохранит ли студент «чистую успеваемость» в следующем семестре** (без блокирующих оценок и пересдач). По правилам в РФ чистый семестр N+1 даёт право на стипендию в N+2, то есть модель работает как **раннее предупреждение за один семестр вперёд**: признаки берутся из сем. N, целевая переменная это состояние в сем. N+1.

## Структура проекта

```
scholarship_pred/
├── data/                                  # входные xlsx
│   ├── ГОСТ Р 70946-2023-Приложение-8_sorted.xlsx
│   └── output_bachelors_*.xlsx
├── outputs/                               # результаты моделей
│   ├── xgb/             канонический XGBoost
│   └── histgb/          канонический HistGradientBoosting
├── scholarship_predict_xgb.py             # канонический XGBoost
├── scholarship_predict_histgb.py          # канонический HistGradientBoosting
├── CHANGELOG.md                           # история патчей XGB-модели
└── README.md
```

Старые варианты моделей (CatBoost в трёх конфигурациях, sklearn-зоопарк, baseline по матанализу, TabPFN, кластеризатор ФГОС `classify_fgos.py`) и их выводы лежат локально в `legacy/` и `outputs/legacy/`. Эти папки добавлены в `.gitignore` и не входят в репозиторий.

## Данные

Источник: выгрузка по структуре **ГОСТ Р 70946-2023, Приложение 8** (анонимизированные оценки).

| Показатель | Значение |
|---|---|
| Строк (оценок) | 155 295 |
| Уникальных студентов | 5 933 |
| Учебных планов | 345 |
| Завершивших программу | 1 852 (1 565 бакалавров, 287 специалистов) |
| Пар (сем. N, сем. N+1) | 11 508 |

Фильтрация в каноническом пайплайне:
- Только очная форма обучения
- Типы ведомостей: Основная, Перезачёт, Пересдача, Пересдача с комиссией
- Пары соседних семестров одного студента

## Целевая переменная

`target_clean_next_sem`: бинарный признак, будет ли в следующем семестре **чистая успеваемость**, то есть:
- нет блокирующих оценок (двоек, неявок без уважительной причины),
- нет пересдач.

По стандартным академическим правилам в РФ чистый сем. N+1 даёт право на стипендию в N+2, поэтому модель работает как ранний прогноз стипендии на два семестра вперёд.

## Признаки (13)

| Признак | Описание |
|---|---|
| `sem_num` | Номер текущего семестра (1-10) |
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

Обоснование изменений признаков лежит в [CHANGELOG.md](CHANGELOG.md).

## Метрики (XGBoost, после патча)

| Метрика | Значение |
|---|---|
| Accuracy | 83.40 % |
| F1 macro | 0.832 |
| ROC-AUC | 0.916 |

Сильнейший baseline даёт ~82.4 %, модель уверенно его превосходит. Сравнение «до и после» патча в [CHANGELOG.md](CHANGELOG.md). Метрики HistGradientBoosting появятся в `outputs/histgb/summary.md` после первого запуска.

## Модели

Каждый скрипт это самостоятельный пайплайн: читает данные из `data/`, пишет в свою папку в `outputs/`.

| Скрипт | Подход | Выход |
|---|---|---|
| `scholarship_predict_xgb.py` | **XGBoost**: early stopping, разбиение по студентам | `outputs/xgb/` |
| `scholarship_predict_histgb.py` | **HistGradientBoosting** (sklearn, аналог LightGBM): нативный NaN, permutation importance | `outputs/histgb/` |

Технические различия между XGB и HistGB описаны в docstring каждого скрипта.

## Запуск

Все скрипты ищут xlsx-файл в `data/` по умолчанию. Можно переопределить через `--data` и `--output`:

```bash
# Канонический XGBoost
python scholarship_predict_xgb.py

# Канонический HistGradientBoosting
python scholarship_predict_histgb.py

# С нестандартными путями
python scholarship_predict_xgb.py --data path/to/file.xlsx --output путь/к/результатам/
```

## Зависимости

```
pandas
numpy
scikit-learn       # включает HistGradientBoosting, permutation_importance, joblib
xgboost            # для scholarship_predict_xgb.py
openpyxl           # чтение xlsx
```

Установка:
```bash
pip install pandas numpy scikit-learn xgboost openpyxl
```

## История

Все изменения канонической XGB-модели задокументированы в [CHANGELOG.md](CHANGELOG.md). Состояние репозитория до большой реструктуризации (май 2026) сохранено в git-теге `v1.0-snapshot`.
