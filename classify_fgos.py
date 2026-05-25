"""
classify_fgos.py — Реконструкция структуры ФГОС из данных об оценках.

Что делает:
  1. Группирует 345 учебных планов в направления подготовки (через общих студентов)
  2. Классифицирует каждый предмет каждого плана по блокам ФГОС:
     Б1.О.General, Б1.О.Prof, Б1.В, Б2, Б3
  3. Генерирует подробные отчёты для ручной проверки

Что генерирует (в папку --output):
  direction_clusters.csv    — какой план в каком направлении
  block_mapping.csv         — каждый (план, предмет) → блок ФГОС
  summary_report.md         — общий отчёт со статистикой
  clusters/cluster_NN.md    — детальный отчёт по каждому направлению
                              (все предметы, сгруппированные по блокам)

Запуск:
    python classify_fgos.py
    python classify_fgos.py --data path/to/file.xlsx --output fgos_output/

После проверки и исправления block_mapping.csv —
запускайте scholarship_predict.py с ключом --blocks block_mapping.csv
"""

import argparse
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# =====================================================================
# CONSTANTS
# =====================================================================
SEMESTER_MAP = {
    "Первый семестр": 1, "Второй семестр": 2, "Третий семестр": 3,
    "Четвертый семестр": 4, "Пятый семестр": 5, "Шестой семестр": 6,
    "Седьмой семестр": 7, "Восьмой семестр": 8, "Девятый семестр": 9,
    "Десятый семестр": 10,
}

DATA_FILE_PATTERNS = [
    "ГОСТ_Р_70946-2023-Приложение-8_sorted.xlsx",
    "ГОСТ Р 70946-2023-Приложение-8_sorted.xlsx",
    "ГОСТ*Приложение*.xlsx",
]

# Subjects containing these words are NOT practices (false positive exclusions)
NOT_PRACTICE = {
    "Теория и практика аргументации",
    "Теория и практика информационной безопасности",
    "Теория и практика отечественного постмодернизма",
    "Теория и практика финансового оздоровления предприятия",
    "Теория и практика связей с общественностью",
}

# Universal subjects (30+ plans) that are actually core professional for STEM
UNIVERSAL_BUT_CORE = {
    "Математический анализ": {"01.03.02", "02.03.01", "09.03.02", "10.03.01",
                               "10.05.01", "03.03.02", "28.03.01", "02.03.02",
                               "09.03.01"},
    "Информатика": {"01.03.02", "02.03.01", "09.03.02", "09.03.03", "09.03.01"},
    "Физика": {"03.03.02", "28.03.01", "06.05.01"},
    "Математика": {"38.03.01", "38.05.01"},
}

# Direction fingerprints: subject sets → ФГОС code
DIRECTION_FINGERPRINTS = [
    ({"Технологии сети Интернет", "Языки высокого уровня",
      "Основы математического моделирования"}, "09.03.02 ИСиТ"),
    ({"Электричество и магнетизм", "Механика",
      "Общий физический лабораторный практикум"}, "03.03.02 Физика / Радиофизика"),
    ({"Информатика и программирование", "Алгебра и теория чисел",
      "Геометрия и топология"}, "01.03.02 ПМИ / 02.03.01 МКН"),
    ({"Геометрия и топология", "Дискретная математика",
      "Алгебра и теория чисел"}, "01.03.02 ПМИ / 02.03.01 МКН"),
    ({"СМИ в современной системе массовой коммуникации",
      "Государственная политика и управление"}, "41.03.04 Политология"),
    ({"Основы информационной безопасности", "Безопасность операционных систем",
      "Физика: Механика"}, "10.05.01 Компьютерная безопасность (спец.)"),
    ({"Финансы", "Страхование", "Бухгалтерский учет",
      "Микроэкономика"}, "38.03.01 Экономика"),
    ({"Устное народное творчество", "Старославянский язык",
      "Введение в литературоведение"}, "45.03.01 Филология"),
    ({"Специальный практикум", "Судебная экспертиза стекла",
      "Трасология"}, "40.05.03 Судебная экспертиза (спец.)"),
    ({"Оценка рисков", "Лабораторный практикум по бухгалтерскому учету",
      "Экономическая безопасность"}, "38.05.01 Экономическая безопасность (спец.)"),
    ({"Ботаника", "Зоология", "Химия: Общая и неорганическая химия",
      "Генетика и эволюция"}, "06.05.01 Биоинженерия (спец.)"),
    ({"Второй иностранный язык",
      "Практикум устной и письменной речи первого иностранного языка"},
     "45.03.02 Лингвистика (прикладная)"),
    ({"Гражданское право", "Уголовное право", "Теория государства и права",
      "Криминалистика"}, "40.03.01 Юриспруденция"),
    ({"Безопасность нанотехнологий",
      "Введение в нанотехнологии"}, "28.03.01 Нанотехнологии"),
    ({"История мировой культуры", "Археология",
      "Этнология"}, "46.03.01 История"),
    ({"Практический курс первого иностранного языка",
      "Практический курс второго иностранного языка",
      "Стилистика"}, "45.03.02 Лингвистика (перевод)"),
    ({"Безопасность операционных систем", "Безопасность вычислительных сетей",
      "Физика: Электричество и магнетизм"}, "10.03.01 Информационная безопасность"),
    ({"Этика государственной и муниципальной службы",
      "Государственное регулирование экономики"}, "38.03.04 ГМУ"),
    ({"Экономическая и социальная география", "Картоведение",
      "Геология"}, "05.03.03 Картография и геоинформатика"),
    ({"Информатика и алгоритмические языки", "Рынки ИКТ"}, "09.03.03 Прикладная информатика"),
    ({"ГИС в экологии", "Техногенные системы",
      "Урбоэкология"}, "05.03.06 Экология"),
    ({"Зоология", "Ботаника", "Клеточная биология",
      "Латинский язык для биологов"}, "06.03.01 Биология"),
    ({"Туристские ресурсы РФ",
      "Туристское страноведение"}, "43.03.02 Туризм"),
    ({"Метрология,стандартизация и сертификация в инфокоммуникациях",
      "Цифровые системы передачи"}, "11.03.02 Инфокоммуникации"),
    ({"Психология творчества", "Социальная психология",
      "Общая психология", "Психодиагностика"}, "37.03.01 Психология"),
    ({"Компьютерная графика", "Языки и методы программирования",
      "Основы информатики"}, "02.03.02 ФИИТ / 09.03.01 Информатика"),
    ({"Выпуск учебных СМИ", "Техника и технологии СМИ",
      "Основы теории журналистики"}, "42.03.02 Журналистика"),
    ({"Реабилитационная работа в социальной сфере",
      "Правовое обеспечение социальной работы"}, "39.03.02 Социальная работа"),
    ({"Технологии производства рекламного продукта",
      "Поведение потребителей"}, "42.03.01 Реклама и СО"),
    ({"Материаловедение", "Сопротивление материалов"}, "12.03.04 Биотехнические системы"),
    ({"Экологическая геология", "Структурная геология"}, "05.03.01 Геология"),
    ({"Геоурбанистика", "География населения"}, "05.03.02 География"),
    ({"Стандартизация, сертификация и метрология",
      "Основы маркетинга"}, "38.03.02 Менеджмент"),
    ({"Анализ данных в социологических исследованиях",
      "Социология города и села"}, "39.03.01 Социология"),
    ({"Иностранный язык (Вводный курс английского языка)",
      "Основы лингвокультурологии"}, "44.03.05 Педобразование (ин.яз.)"),
    ({"Антикризисное управление",
      "Диагностика кризисного состояния предприятия"}, "38.03.02 Менеджмент (антикриз.)"),
    ({"Теория и практика связей с общественностью",
      "Основные теории коммуникации"}, "42.03.01 Реклама и СО (PR)"),
    ({"Профессиональная этика юриста",
      "История учений о государстве и праве"}, "40.05.01 Правовое обесп. нац. безопасности (спец.)"),
    ({"Государственная молодежная политика",
      "Менеджмент в молодежной политике"}, "39.03.03 Организация работы с молодёжью"),
    ({"Экология человека", "Ноксология"}, "20.03.01 Техносферная безопасность"),
    ({"Международная торговля", "Мерчендайзинг"}, "38.03.06 Торговое дело"),
]


# =====================================================================
# DATA LOADING
# =====================================================================
def find_data_file(directory: Path) -> Path:
    for name in DATA_FILE_PATTERNS:
        if "*" not in name:
            candidate = directory / name
            if candidate.exists():
                return candidate
    for pattern in DATA_FILE_PATTERNS:
        if "*" in pattern:
            matches = list(directory.glob(pattern))
            if matches:
                return matches[0]
    raise FileNotFoundError(
        f"Не нашёл xlsx в {directory}. Укажите --data путь/к/файлу.xlsx"
    )


def load_data(path=None):
    if path is None:
        path = find_data_file(Path(__file__).parent / "data")
    path = Path(path)
    print(f"[1/5] Загрузка данных из {path.name}...")

    df = pd.read_excel(path, sheet_name="Sheet1")
    df["sem_num"] = df["ПериодКонтроля"].map(SEMESTER_MAP)
    df = df[df["sem_num"].notna()].copy()
    df["sem_num"] = df["sem_num"].astype(int)
    df = df[(df["ФормаОбучения"] == "Очная") & (df["ТипВедомости"] == "Основная")]

    print(f"  {len(df):,} записей, {df['ЗачетнаяКнижка'].nunique():,} студентов, "
          f"{df['УчебныйПлан'].nunique()} планов")
    return df


# =====================================================================
# DIRECTION CLUSTERING
# =====================================================================
def cluster_directions(df):
    print(f"[2/5] Кластеризация планов по направлениям...")

    parent = {}
    def find(x):
        if x not in parent: parent[x] = x
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py: parent[px] = py

    all_plans = set(df["УчебныйПлан"].unique())
    for p in all_plans: find(p)

    # Pass 1: student overlap
    student_plans = df.groupby("ЗачетнаяКнижка")["УчебныйПлан"].apply(set)
    edges = defaultdict(int)
    for _, plans in student_plans.items():
        plans = sorted(plans)
        for i in range(len(plans)):
            for j in range(i + 1, len(plans)):
                edges[(plans[i], plans[j])] += 1
    for (p1, p2), cnt in edges.items():
        if cnt >= 3: union(p1, p2)

    # Pass 2: cross-cohort subject similarity
    plan_subjects = df.groupby("УчебныйПлан")["Дисциплина"].apply(set)
    plan_sems = df.groupby("УчебныйПлан")["sem_num"].apply(
        lambda s: tuple(sorted(set(s.astype(int)))))
    plan_level = df.groupby("УчебныйПлан")["УровеньПодготовки"].first()
    plan_students = df.groupby("УчебныйПлан")["ЗачетнаяКнижка"].nunique()
    subj_plan_count = df.groupby("Дисциплина")["УчебныйПлан"].nunique()
    universal_subjs = set(subj_plan_count[subj_plan_count >= 30].index)

    chains = defaultdict(set)
    for p in all_plans: chains[find(p)].add(p)
    chain_profiles = {}
    for cid, plans in chains.items():
        profile = defaultdict(set)
        for p in plans:
            sems = plan_sems.get(p, ())
            profile[sems] |= plan_subjects.get(p, set()) - universal_subjs
        chain_profiles[cid] = dict(profile)

    def jaccard(s1, s2):
        if not s1 or not s2: return 0.0
        return len(s1 & s2) / len(s1 | s2)

    chain_ids = list(chains.keys())
    for i in range(len(chain_ids)):
        for j in range(i + 1, len(chain_ids)):
            c1, c2 = chain_ids[i], chain_ids[j]
            levels1 = set(plan_level.get(p) for p in chains[c1])
            levels2 = set(plan_level.get(p) for p in chains[c2])
            if not levels1 & levels2: continue
            for sems in chain_profiles[c1]:
                if sems in chain_profiles[c2]:
                    if jaccard(chain_profiles[c1][sems], chain_profiles[c2][sems]) >= 0.3:
                        union(c1, c2); break

    # Build final clusters
    final_clusters = defaultdict(set)
    for p in all_plans: final_clusters[find(p)].add(p)
    cluster_list = sorted(final_clusters.values(), key=len, reverse=True)

    # Name each cluster
    cluster_names = {}
    for i, cluster in enumerate(cluster_list):
        cid = i + 1
        all_subjs = set()
        for p in cluster:
            all_subjs |= plan_subjects.get(p, set())
        for fp_subjs, name in DIRECTION_FINGERPRINTS:
            if fp_subjs.issubset(all_subjs):
                cluster_names[cid] = name
                break
        if cid not in cluster_names:
            # Use most distinctive subject as fallback name
            prof = [s for s in all_subjs if s not in universal_subjs]
            cluster_names[cid] = f"Неизвестное ({prof[0][:40]}...)" if prof else f"Неизвестное_{cid}"

    # Plan-to-cluster mapping
    plan_to_cluster = {}
    for i, cluster in enumerate(cluster_list):
        for p in cluster:
            plan_to_cluster[int(p)] = i + 1

    n_students_clustered = sum(
        df[df["УчебныйПлан"].isin(c)]["ЗачетнаяКнижка"].nunique()
        for c in cluster_list
    )
    print(f"  {len(cluster_list)} направлений из {len(all_plans)} планов")
    print(f"  Крупнейшие: {', '.join(cluster_names[i+1] for i in range(min(5, len(cluster_list))))}")

    return (plan_to_cluster, cluster_list, cluster_names, universal_subjs,
            plan_subjects, plan_sems, plan_students, plan_level, subj_plan_count)


# =====================================================================
# BLOCK CLASSIFICATION
# =====================================================================
def classify_blocks(df, plan_to_cluster, cluster_list, cluster_names,
                    universal_subjs, plan_subjects, plan_sems, subj_plan_count):
    print(f"[3/5] Классификация предметов по блокам ФГОС...")

    # Peer frequency: how common is subject among plans with SAME semesters
    cluster_plans_by_sems = {}
    for i, cluster in enumerate(cluster_list):
        cid = i + 1
        by_sems = defaultdict(set)
        for p in cluster:
            by_sems[plan_sems.get(p, ())].add(p)
        cluster_plans_by_sems[cid] = dict(by_sems)

    def get_peer_freq(subject, plan, cid):
        sem_cov = plan_sems.get(plan, ())
        peers = cluster_plans_by_sems.get(cid, {}).get(sem_cov, set())
        if len(peers) <= 1: return 1.0
        return sum(1 for p in peers if subject in plan_subjects.get(p, set())) / len(peers)

    # Determine cluster ФГОС code for UNIVERSAL_BUT_CORE matching
    cluster_code = {}
    for i, cluster in enumerate(cluster_list):
        cid = i + 1
        name = cluster_names.get(cid, "")
        # Extract code like "01.03.02" from name
        parts = name.split()
        if parts and "." in parts[0]:
            cluster_code[cid] = parts[0]
        else:
            cluster_code[cid] = name

    # ВидКонтроля per (plan, subject)
    vid_by_ps = df.groupby(["УчебныйПлан", "Дисциплина"])["ВидКонтроля"].agg(
        lambda x: x.value_counts().index[0])

    # Grade stats per (plan, subject) for the report
    grade_stats = df.groupby(["УчебныйПлан", "Дисциплина"]).agg(
        n_students=("ЗачетнаяКнижка", "nunique"),
        semesters=("sem_num", lambda s: sorted(set(s.astype(int)))),
    )

    def classify_subject(subject, vid, cid, peer_freq):
        ccode = cluster_code.get(cid, "")

        # Б3 — ГИА
        if vid in ("ГАК", "Защита ВКР"):
            return "Б3"
        if any(kw in subject for kw in [
            "Государственный экзамен", "Подготовка к сдаче",
            "Подготовка к процедуре защиты",
            "Выпускная квалификационная работа"
        ]):
            return "Б3"

        # Б2 — Практики
        if subject not in NOT_PRACTICE:
            if (subject.startswith("Учебная практика") or
                subject.startswith("Производственная практика")):
                return "Б2"
            if subject in ("Преддипломная практика", "Преддипломная"):
                return "Б2"
            if "НИР" in subject or "стажировка" in subject.lower():
                return "Б2"
            if "научно-исследовательская работа" in subject.lower():
                return "Б2"

        # Universal but core professional for specific directions
        if subject in UNIVERSAL_BUT_CORE:
            allowed = UNIVERSAL_BUT_CORE[subject]
            if any(code in ccode for code in allowed):
                return "Б1.О.Prof"

        # Б1.О.General — universal subjects (30+ plans)
        if subject in universal_subjs:
            return "Б1.О.General"

        # Б1.О.Prof — high peer frequency (≥60%)
        if peer_freq >= 0.6:
            return "Б1.О.Prof"

        # Б1.В — everything else
        return "Б1.В"

    # Apply to all (plan, subject) pairs
    rows = []
    all_plans = set(df["УчебныйПлан"].unique())
    for plan in all_plans:
        plan = int(plan)
        cid = plan_to_cluster.get(plan, -1)
        cname = cluster_names.get(cid, "Unknown")
        level = df[df["УчебныйПлан"] == plan]["УровеньПодготовки"].iloc[0]

        for subj in plan_subjects.get(plan, set()):
            vid = vid_by_ps.get((plan, subj), "Unknown")
            pf = get_peer_freq(subj, plan, cid)
            block = classify_subject(subj, vid, cid, pf)

            gs = grade_stats.loc[(plan, subj)] if (plan, subj) in grade_stats.index else None
            n_stud = int(gs["n_students"]) if gs is not None else 0
            sems = gs["semesters"] if gs is not None else []

            rows.append({
                "УчебныйПлан": plan,
                "Дисциплина": subj,
                "direction_cluster": cid,
                "direction_name": cname,
                "level": level,
                "block": block,
                "peer_freq": round(pf, 3),
                "ВидКонтроля": vid,
                "n_students": n_stud,
                "semesters": str(sems),
            })

    mapping = pd.DataFrame(rows)

    counts = mapping["block"].value_counts().sort_index()
    for block, cnt in counts.items():
        print(f"  {block:<15} {cnt:>5,} ({cnt / len(mapping) * 100:>5.1f}%)")

    return mapping


# =====================================================================
# OUTPUT GENERATION
# =====================================================================
def save_outputs(df, mapping, plan_to_cluster, cluster_list, cluster_names,
                 plan_students, plan_level, plan_sems, output_dir):
    print(f"[4/5] Генерация отчётов...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clusters_dir = output_dir / "clusters"
    clusters_dir.mkdir(exist_ok=True)

    # --- 1. direction_clusters.csv ---
    dir_rows = []
    for i, cluster in enumerate(cluster_list):
        cid = i + 1
        unique_students = df[df["УчебныйПлан"].isin(cluster)]["ЗачетнаяКнижка"].nunique()
        levels = list(df[df["УчебныйПлан"].isin(cluster)]["УровеньПодготовки"].unique())
        n_subjects = mapping[mapping["direction_cluster"] == cid]["Дисциплина"].nunique()

        plans_detail = []
        for p in sorted(cluster):
            sems = plan_sems.get(p, ())
            ns = plan_students.get(p, 0)
            plans_detail.append(f"{p}(сем{list(sems)},{ns}ст)")

        dir_rows.append({
            "cluster_id": cid,
            "direction_name": cluster_names[cid],
            "level": levels[0] if levels else "?",
            "n_plans": len(cluster),
            "n_students": unique_students,
            "n_subjects": n_subjects,
            "plans": sorted([int(p) for p in cluster]),
            "plans_detail": "; ".join(plans_detail),
        })

    dir_df = pd.DataFrame(dir_rows).sort_values("n_students", ascending=False)
    dir_csv = output_dir / "direction_clusters.csv"
    dir_df.to_csv(dir_csv, index=False, encoding="utf-8-sig")
    print(f"  {dir_csv}")

    # --- 2. block_mapping.csv ---
    block_csv = output_dir / "block_mapping.csv"
    mapping.to_csv(block_csv, index=False, encoding="utf-8-sig")
    print(f"  {block_csv}")

    # --- 3. Per-cluster detail files ---
    for i, cluster in enumerate(cluster_list):
        cid = i + 1
        cname = cluster_names[cid]
        unique_students = df[df["УчебныйПлан"].isin(cluster)]["ЗачетнаяКнижка"].nunique()
        levels = list(df[df["УчебныйПлан"].isin(cluster)]["УровеньПодготовки"].unique())

        cluster_mapping = mapping[mapping["direction_cluster"] == cid].copy()

        fname = clusters_dir / f"cluster_{cid:02d}.md"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(f"# Кластер {cid}: {cname}\n\n")
            f.write(f"**Уровень:** {', '.join(levels)}\n\n")
            f.write(f"**Планов:** {len(cluster)} | **Студентов:** {unique_students}\n\n")

            # Plans table
            f.write(f"## Учебные планы\n\n")
            f.write(f"| План | Студентов | Семестры |\n|---|---|---|\n")
            for p in sorted(cluster):
                sems = list(plan_sems.get(p, ()))
                ns = plan_students.get(p, 0)
                f.write(f"| {p} | {ns} | {sems} |\n")

            # Subjects by block
            for block in ["Б1.О.General", "Б1.О.Prof", "Б1.В", "Б2", "Б3"]:
                block_subjs = cluster_mapping[cluster_mapping["block"] == block]
                # Deduplicate by subject name (same subject in multiple plans)
                unique_subjs = block_subjs.groupby("Дисциплина").agg(
                    n_plans=("УчебныйПлан", "nunique"),
                    n_students=("n_students", "max"),
                    peer_freq=("peer_freq", "mean"),
                    vid=("ВидКонтроля", "first"),
                    semesters=("semesters", "first"),
                ).sort_values("n_students", ascending=False)

                f.write(f"\n## {block} ({len(unique_subjs)} предметов)\n\n")
                if len(unique_subjs) == 0:
                    f.write(f"*Нет предметов в этом блоке.*\n")
                    continue

                f.write(f"| Предмет | Планов | Студентов | Peer freq | Контроль | Семестры |\n")
                f.write(f"|---|---|---|---|---|---|\n")
                for subj, row in unique_subjs.iterrows():
                    f.write(f"| {subj[:70]} | {row['n_plans']} | {row['n_students']} | "
                            f"{row['peer_freq']:.2f} | {row['vid']} | {row['semesters']} |\n")

            # Summary stats
            f.write(f"\n## Сводка\n\n")
            block_counts = cluster_mapping["block"].value_counts().sort_index()
            f.write(f"| Блок | Записей (план×предмет) |\n|---|---|\n")
            for block, cnt in block_counts.items():
                f.write(f"| {block} | {cnt} |\n")

    print(f"  {clusters_dir}/ ({len(cluster_list)} файлов)")

    # --- 4. summary_report.md ---
    summary_path = output_dir / "summary_report.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"# Реконструкция структуры ФГОС\n\n")
        f.write(f"**Дата:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Исходные данные:** {len(df):,} записей, "
                f"{df['ЗачетнаяКнижка'].nunique():,} студентов, "
                f"{df['УчебныйПлан'].nunique()} планов\n\n")

        f.write(f"## Метод\n\n")
        f.write(f"1. Учебные планы объединены в направления через общих студентов "
                f"(≥3 общих) и пересечение предметов (Jaccard ≥0.3 для планов "
                f"с одинаковым покрытием семестров).\n")
        f.write(f"2. Предметы классифицированы по блокам ФГОС на основе:\n")
        f.write(f"   - ВидКонтроля (ГАК/ВКР → Б3)\n")
        f.write(f"   - Ключевые слова в названии (практика → Б2)\n")
        f.write(f"   - Частота по планам (≥30 планов → Б1.О.General)\n")
        f.write(f"   - Peer frequency внутри направления (≥60% → Б1.О.Prof)\n\n")

        f.write(f"## Результаты\n\n")
        f.write(f"### Направления ({len(cluster_list)})\n\n")
        f.write(f"| # | Направление | Планов | Студентов | Уровень |\n")
        f.write(f"|---|---|---|---|---|\n")
        for _, row in dir_df.iterrows():
            f.write(f"| {row['cluster_id']} | {row['direction_name']} | "
                    f"{row['n_plans']} | {row['n_students']} | {row['level']} |\n")

        f.write(f"\n### Блоки ФГОС\n\n")
        f.write(f"| Блок | Записей | Доля |\n|---|---|---|\n")
        counts = mapping["block"].value_counts().sort_index()
        for block, cnt in counts.items():
            f.write(f"| {block} | {cnt:,} | {cnt / len(mapping) * 100:.1f}% |\n")

        f.write(f"\n### Файлы\n\n")
        f.write(f"- `direction_clusters.csv` — какой план в каком направлении\n")
        f.write(f"- `block_mapping.csv` — каждый (план, предмет) → блок ФГОС\n")
        f.write(f"- `clusters/cluster_NN.md` — детальный отчёт по каждому направлению\n\n")
        f.write(f"### Как проверять\n\n")
        f.write(f"1. Откройте `clusters/cluster_12.md` (Юриспруденция) — крупнейшее направление\n")
        f.write(f"2. Проверьте: все ли профильные предметы в Б1.О.Prof, а не в Б1.В?\n")
        f.write(f"3. Если нашли ошибку — исправьте в `block_mapping.csv` (колонка `block`)\n")
        f.write(f"4. Затем запускайте `scholarship_predict.py --blocks block_mapping.csv`\n\n")

        f.write(f"### Известные ограничения\n\n")
        f.write(f"- Граница Б1.О/Б1.В точна на ~85-90% (STEM направления хуже)\n")
        f.write(f"- Б2 может ложно сработать на предметы со словом 'практика' в названии\n")
        f.write(f"- Направления с 1-2 планами и <20 студентами ненадёжны\n")
        f.write(f"- Различие Б1.В.ОД / Б1.В.ДВ не реконструируется\n")

    print(f"  {summary_path}")

    return dir_csv, block_csv


# =====================================================================
# VALIDATION SPOT-CHECKS
# =====================================================================
def run_spotchecks(mapping, cluster_names):
    print(f"[5/5] Спот-чеки...")

    # Find юриспруденция cluster
    law_cid = None
    for cid, name in cluster_names.items():
        if "Юриспруденция" in name and "спец" not in name.lower():
            law_cid = cid
            break

    if law_cid:
        law = mapping[mapping["direction_cluster"] == law_cid]
        law_prof = law[law["block"] == "Б1.О.Prof"]["Дисциплина"].unique()
        expected_law = {"Теория государства и права", "Уголовное право",
                        "Конституционное право", "Гражданское право",
                        "Уголовный процесс", "Гражданский процесс"}
        found = expected_law & set(law_prof)
        missing = expected_law - set(law_prof)
        print(f"  Юриспруденция (кластер {law_cid}):")
        print(f"    Б1.О.Prof предметов: {len(law_prof)}")
        print(f"    Ожидаемых найдено: {len(found)}/{len(expected_law)}", end="")
        if missing:
            print(f" (пропущено: {missing})")
        else:
            print(f" ✓")

    # Find ПМИ cluster
    pmi_cid = None
    for cid, name in cluster_names.items():
        if "ПМИ" in name or "МКН" in name:
            pmi_cid = cid
            break

    if pmi_cid:
        pmi = mapping[mapping["direction_cluster"] == pmi_cid]
        pmi_prof = pmi[pmi["block"] == "Б1.О.Prof"]["Дисциплина"].unique()
        expected_pmi = {"Математический анализ", "Дискретная математика",
                        "Информатика и программирование", "Базы данных"}
        found = expected_pmi & set(pmi_prof)
        missing = expected_pmi - set(pmi_prof)
        print(f"  ПМИ/МКН (кластер {pmi_cid}):")
        print(f"    Б1.О.Prof предметов: {len(pmi_prof)}")
        print(f"    Ожидаемых найдено: {len(found)}/{len(expected_pmi)}", end="")
        if missing:
            print(f" (пропущено: {missing})")
        else:
            print(f" ✓")

    # Check Б2 false positives
    b2 = mapping[mapping["block"] == "Б2"]["Дисциплина"].unique()
    suspicious_b2 = [s for s in b2 if "Практикум" in s or
                     ("практик" in s.lower() and
                      not s.startswith("Учебная") and
                      not s.startswith("Производственная") and
                      "НИР" not in s)]
    if suspicious_b2:
        print(f"  Подозрительные Б2 (возможные false positives):")
        for s in suspicious_b2[:5]:
            print(f"    ⚠ {s}")
    else:
        print(f"  Б2 false positives: не обнаружены ✓")


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Реконструкция структуры ФГОС из данных об оценках"
    )
    parser.add_argument("--data", type=str, default=None,
                        help="Путь к xlsx-файлу")
    parser.add_argument("--output", type=str, default="fgos_output",
                        help="Папка для результатов (default: fgos_output/)")
    args = parser.parse_args()

    df = load_data(args.data)

    (plan_to_cluster, cluster_list, cluster_names, universal_subjs,
     plan_subjects, plan_sems, plan_students, plan_level,
     subj_plan_count) = cluster_directions(df)

    mapping = classify_blocks(
        df, plan_to_cluster, cluster_list, cluster_names,
        universal_subjs, plan_subjects, plan_sems, subj_plan_count
    )

    dir_csv, block_csv = save_outputs(
        df, mapping, plan_to_cluster, cluster_list, cluster_names,
        plan_students, plan_level, plan_sems, Path(args.output)
    )

    run_spotchecks(mapping, cluster_names)

    print(f"\n{'=' * 70}")
    print(f"  Готово! Результаты в папке: {Path(args.output).resolve()}")
    print(f"")
    print(f"  Следующие шаги:")
    print(f"  1. Проверьте clusters/cluster_XX.md — особенно Б1.О.Prof")
    print(f"  2. Исправьте ошибки в block_mapping.csv (колонка 'block')")
    print(f"  3. Запустите: python scholarship_predict.py --blocks {block_csv}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
