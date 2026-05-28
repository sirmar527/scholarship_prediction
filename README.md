# Scholarship Prediction

**EN.** This is a university project.

Binary classifier predicting whether a Russian-university student will maintain a clean academic record (no blocking grades, no retakes) in the next semester. Dataset: anonymised grade-level data conforming to ГОСТ Р 70946-2023, Приложение 8.

## Задача

По данным семестра N одного студента предсказать, будет ли следующий семестр (N+1) чистым (без блокирующих оценок и без пересдач). По академическим правилам чистый семестр N даёт право на стипендию в семестре N+1; Модель работает как ранний прогноз на один семестр вперёд.


## Данные

Основной набор (`data/ГОСТ Р 70946-2023-Приложение-8_sorted.xlsx`):

| | |
|---|---|
| Строк (оценок) | 155 426 |
| Студентов | 5 934 |
| Учебных планов | 345 |
| Пар (сем. N, сем. N+1) | 11 512 |

Фильтрация: только очная форма; типы ведомостей {Основная, Перезачёт, Пересдача, Пересдача с комиссией}; соседние семестры одного студента.

Дополнительный срез (`data/output_bachelors_both_sem2_3_ochnaya.xlsx`): бакалавриат, очная, только семестры 2 и 3, 987 студентов, у каждого есть записи в обоих семестрах. 18 523 строки, ровно одна пара (2 -> 3) на студента. Схема колонок идентична основному набору; те же фильтры и константы применимы без изменений.

## Целевая переменная

`target_clean_next_sem` принимает значения 0 или 1: 1, если в семестре N+1 нет блокирующих оценок и нет пересдач.

## Признаки (13)

| Признак | Описание                                         |
|---|--------------------------------------------------|
| `sem_num` | Номер семестра (1-10)                            |
| `n_subjects` | Число дисциплин в семестре                       |
| `n_blocks`, `any_block` | Количество и наличие блокирующих оценок          |
| `n_retakes`, `any_retake` | Количество и наличие пересдач                    |
| `gpa_overall` | Средний балл по числовым оценкам                 |
| `min_grade` | Минимальная оценка                               |
| `std_grade` | Стандартное отклонение оценок                    |
| `share_5`, `share_3` | Доли пятёрок и троек среди оцениваемых дисциплин |
| `share_zachet` | Доля зачётных дисциплин в нагрузке               |
| `had_clean_current_sem` | Был ли чистым текущий семестр                    |

## Модели

`scholarship_predict_histgb.py` - основной скрипт. Обучает `sklearn.ensemble.HistGradientBoostingClassifier` с встроенным ранней остановкой. Разбиение 80/20 по студентам. Permutation-importance используется как мера важности признаков.

`scholarship_predict_histgb_bachelors_sem2_3.py` - дубликат основного скрипта, адаптированный под `output_bachelors_both_sem2_3_ochnaya.xlsx`.

## Метрики

Основной набор (test: 2 343 пары, train: 9 169 пар):

| Метрика | Значение |
|---|---|
| Accuracy | 83.14 % |
| F1 (macro) | 0.830 |
| ROC-AUC | 0.914 |

Лучший бейзлайн на той же выборке: accuracy ~82.3 % (по правилу "следующий семестр будет таким же, как текущий"). Полные таблицы находятся в `outputs/histgb/summary.md`.

Срез бакалавриата 2 -> 3 (test: 198 пар, train: 789 пар):

| Метрика | Значение |
|---|---|
| Accuracy | 80.3 % |
| F1 (macro) | 0.800 |
| ROC-AUC | 0.885 |

Лучший бейзлайн на этом срезе: accuracy 82.3 % (правило `gpa >= 4.2`). Полные таблицы в `outputs/histgb_bachelors_sem2_3/summary.md`.

## Запуск

```bash
pip install -r requirements.txt
python scholarship_predict_histgb.py                       # основной набор
python scholarship_predict_histgb_bachelors_sem2_3.py      # срез бакалавриата 2 -> 3
```

## Документация

Полное описание архитектуры приведено в [`technical_report.md`](technical_report.md).

## Структура

```
data/                                              входные xlsx
outputs/histgb/                                    результаты основного скрипта
outputs/histgb_bachelors_sem2_3/                   результаты дубликата
scholarship_predict_histgb.py                      основной скрипт
scholarship_predict_histgb_bachelors_sem2_3.py     дубликат для среза 2 -> 3
requirements.txt                                   зависимости
technical_report.md                                тех. отчёт
```
