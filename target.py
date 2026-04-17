"""
Расчёт целевой переменной.

Правила назначения стипендии (по уточнённому ТЗ):
    scholarship(студент, семестр N) = 1 тогда и только тогда, когда
    по всем строкам основной ведомости этого (студент, семестр):
        * ИтоговаяОтметка ∈ {'Отлично', 'Хорошо', 'Зачтено'}
        * Пересдача == 0
        * Комиссия == 0
        * ПересдачаДляДиплома == 0
    Во всех остальных случаях — 0.

    Отдельное правило: в 1-м семестре все только что поступившие бюджетники
    получают стипендию по умолчанию (scholarship = 1), независимо от оценок
    этой сессии.

Целевая переменная (вариант B из ТЗ):
    1 — была стипендия, сохранил        (prev=1, next=1)
    2 — была стипендия, потерял         (prev=1, next=0)
    3 — не было, получил                (prev=0, next=1)
    4 — не было, и сейчас нет           (prev=0, next=0)
"""
import pandas as pd


PASSING_GRADES = {'Отлично', 'Хорошо', 'Зачтено'}
FAILING_GRADES = {'Удовлетворительно', 'Неудовлетворительно', 'Не зачтено', 'Неявка'}
RETAKE_COLUMNS = ['Пересдача', 'Комиссия', 'ПересдачаДляДиплома']

TRANSITION_LABELS = {
    1: 'Была → сохранил',
    2: 'Была → потерял',
    3: 'Не было → получил',
    4: 'Не было → нет',
}


def compute_scholarship_per_semester(df: pd.DataFrame) -> pd.DataFrame:
    """Считает статус стипендии для каждой пары (студент, семестр).

    Вход: DataFrame, уже обработанный data_prep (имеет sem_num, period_type)
          и отфильтрованный по ТипВедомости == 'Основная'. Используем только
          regular-периоды.

    Выход: DataFrame с колонками
        ЗачетнаяКнижка, sem_num, scholarship (0/1), n_records,
        УчебныйПлан, УровеньПодготовки.

    В 1-м семестре scholarship принудительно = 1.
    """
    reg = df[df['period_type'] == 'regular'].copy()

    reg['_fail'] = reg['ИтоговаяОтметка'].isin(FAILING_GRADES)
    reg['_pass'] = reg['ИтоговаяОтметка'].isin(PASSING_GRADES)
    reg['_retake'] = (reg[RETAKE_COLUMNS] > 0).any(axis=1)

    agg = reg.groupby(['ЗачетнаяКнижка', 'sem_num']).agg(
        any_fail=('_fail', 'any'),
        all_pass=('_pass', 'all'),
        any_retake=('_retake', 'any'),
        n_records=('_fail', 'size'),
        УчебныйПлан=('УчебныйПлан', 'first'),
        УровеньПодготовки=('УровеньПодготовки', 'first'),
    ).reset_index()

    agg['scholarship'] = (
        (~agg['any_fail']) & agg['all_pass'] & (~agg['any_retake'])
    ).astype(int)

    # Принудительно: в 1-м семестре все со стипендией
    agg.loc[agg['sem_num'] == 1, 'scholarship'] = 1

    return agg[['ЗачетнаяКнижка', 'sem_num', 'scholarship',
                'n_records', 'УчебныйПлан', 'УровеньПодготовки']]


def build_transition_table(scholarship_df: pd.DataFrame) -> pd.DataFrame:
    """Из таблицы scholarship по (студент, семестр) строит таблицу переходов.

    Для каждого студента и каждой доступной пары (N, N+1) создаёт строку с
    prev_scholarship, next_scholarship, class_B (1..4).
    """
    sorted_df = scholarship_df.sort_values(['ЗачетнаяКнижка', 'sem_num'])
    sorted_df['prev_sem'] = sorted_df.groupby('ЗачетнаяКнижка')['sem_num'].shift(1)
    sorted_df['prev_scholarship'] = sorted_df.groupby('ЗачетнаяКнижка')['scholarship'].shift(1)

    # Оставляем только строки, где предыдущий семестр ровно на 1 меньше текущего
    mask = (sorted_df['sem_num'] - sorted_df['prev_sem']) == 1
    trans = sorted_df[mask].rename(columns={
        'sem_num': 'to_sem',
        'scholarship': 'next_scholarship',
    }).copy()
    trans['from_sem'] = (trans['to_sem'] - 1).astype(int)
    trans['prev_scholarship'] = trans['prev_scholarship'].astype(int)
    trans['next_scholarship'] = trans['next_scholarship'].astype(int)

    trans['class_B'] = (
        trans['prev_scholarship'] * 2 + (1 - trans['next_scholarship']) + 1
    ).astype(int)
    # map: (1,1)→1, (1,0)→2, (0,1)→3, (0,0)→4
    # prev=1,next=1: 1*2 + 0 + 1 = 3  -- не то

    # Перепишу явно, без трюков:
    def cls(row):
        p, n = row['prev_scholarship'], row['next_scholarship']
        if p == 1 and n == 1: return 1
        if p == 1 and n == 0: return 2
        if p == 0 and n == 1: return 3
        return 4
    trans['class_B'] = trans.apply(cls, axis=1)

    return trans[['ЗачетнаяКнижка', 'УчебныйПлан', 'УровеньПодготовки',
                  'from_sem', 'to_sem',
                  'prev_scholarship', 'next_scholarship', 'class_B']]


def print_scholarship_summary(scholarship_df: pd.DataFrame) -> None:
    print("Распределение стипендии по (студент, семестр):")
    print(f"  всего пар:           {len(scholarship_df):,}")
    print(f"  scholarship = 1:     {(scholarship_df['scholarship']==1).sum():,}")
    print(f"  scholarship = 0:     {(scholarship_df['scholarship']==0).sum():,}")
    print()
    print("Доля получателей по семестрам:")
    by_sem = scholarship_df.groupby('sem_num')['scholarship'].agg(['mean', 'count'])
    by_sem.columns = ['доля_со_стипендией', 'студентов']
    print(by_sem.to_string(float_format=lambda x: f'{x:.3f}'))


def print_transition_summary(trans_df: pd.DataFrame) -> None:
    print(f"Всего переходов (пар соседних семестров): {len(trans_df):,}")
    print("\nРаспределение по классам варианта B:")
    counts = trans_df['class_B'].value_counts().sort_index()
    total = counts.sum()
    for cls, cnt in counts.items():
        print(f"  класс {cls} ({TRANSITION_LABELS[cls]:<20}): {cnt:>7,}  ({cnt/total*100:5.1f}%)")
    print("\nРаспределение по горизонту (from_sem → to_sem):")
    by_horizon = (trans_df.groupby(['from_sem', 'to_sem']).size()
                  .rename('count').reset_index())
    print(by_horizon.to_string(index=False))


if __name__ == '__main__':
    import data_prep as dp

    raw = dp.load_raw()
    df = dp.add_semester_info(raw)
    df = dp.filter_main_sheets(df)

    print("=" * 70)
    print("Шаг 1: расчёт статуса стипендии по каждому (студент, семестр)")
    print("=" * 70)
    sch = compute_scholarship_per_semester(df)
    print_scholarship_summary(sch)

    print()
    print("=" * 70)
    print("Шаг 2: построение таблицы переходов (обучающих пар)")
    print("=" * 70)
    trans = build_transition_table(sch)
    print_transition_summary(trans)

    print()
    print("=" * 70)
    print("Шаг 3: контрольный пример — случайный студент")
    print("=" * 70)
    sample_id = sch.groupby('ЗачетнаяКнижка').size().sort_values(ascending=False).index[0]
    print(f"Студент {sample_id}:")
    print("\n  Статус стипендии по семестрам:")
    print(sch[sch['ЗачетнаяКнижка'] == sample_id][
        ['sem_num', 'n_records', 'scholarship']
    ].to_string(index=False))
    print("\n  Переходы:")
    sub = trans[trans['ЗачетнаяКнижка'] == sample_id]
    if len(sub):
        print(sub[['from_sem', 'to_sem', 'prev_scholarship',
                   'next_scholarship', 'class_B']].to_string(index=False))
    else:
        print("  (нет пар соседних семестров)")
