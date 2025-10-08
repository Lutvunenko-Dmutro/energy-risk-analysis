import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np
import seaborn as sns
from lifelines import CoxPHFitter, KaplanMeierFitter

sns.set_style("whitegrid")

# Налаштування назв колонок
DURATION_COL = "тривалість"
EVENT_COL = "подія"
FEATURE_COLUMNS = [
    "навантаження_мвт", "потужність_мвт", "завантаженість",
    "температура_с", "вітер_м_с", "свято", "вік_років"
]

def ensure_columns(df, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"У CSV відсутні колонки: {missing}\nЄ колонки: {df.columns.tolist()}")

def train_cox_model(df):
    # lifelines очікує EVENT як 0/1 або True/False
    if df[EVENT_COL].dtype == object:
        df[EVENT_COL] = df[EVENT_COL].astype(int)
    cph = CoxPHFitter()
    cph.fit(df[[DURATION_COL, EVENT_COL] + FEATURE_COLUMNS],
            duration_col=DURATION_COL, event_col=EVENT_COL)
    return cph

def generate_random_data(n=120):
    rng = np.random.default_rng()
    df = pd.DataFrame({
        "ід": np.arange(1, n+1),
        DURATION_COL: rng.integers(10, 200, size=n),
        EVENT_COL: rng.integers(0, 2, size=n),
        "навантаження_мвт": rng.uniform(3000, 6000, size=n),
        "потужність_мвт": rng.uniform(4000, 7000, size=n),
        "завантаженість": rng.uniform(0.5, 0.9, size=n),
        "температура_с": rng.uniform(-20, 35, size=n),
        "вітер_м_с": rng.uniform(0, 15, size=n),
        "свято": rng.integers(0, 2, size=n),
        "вік_років": rng.integers(1, 50, size=n)
    })
    # Синтетична назва та тип елемента (якщо у CSV немає таких)
    df["назва"] = ["Елемент-" + str(i) for i in df["ід"]]
    # Простий генератор типів: частково залежить від потужності, щоб було реалістично
    def infer_type_by_power(p):
        if p > 6000:
            return "Генератор"
        if p > 5200:
            return "Підстанція"
        return "Лінія"
    df["тип"] = [infer_type_by_power(p) for p in df["потужність_мвт"]]
    return df

def run_dashboard():
    # Спроба читання CSV
    try:
        df = pd.read_csv("cox_energy_dataset.csv", encoding="utf-8-sig")
        ensure_columns(df, [DURATION_COL, EVENT_COL] + FEATURE_COLUMNS)
        cph = train_cox_model(df.copy())
        source_label = "cox_energy_dataset.csv"
    except Exception as e:
        # Фолбек на випадкові дані
        df = generate_random_data(120)
        cph = train_cox_model(df.copy())
        source_label = "випадкові дані"
        messagebox.showwarning(
            "⚠️ Дані — Система моніторингу",
            f"CSV не завантажено для Системи моніторингу завантаженості енергосистеми, використані випадкові дані.\nПричина:\n{e}"
        )
    # Якщо у CSV немає стовпців "назва"/"тип" — підставимо (або зробимо інфраструктурну інтерпретацію)
    if "назва" not in df.columns:
        df["назва"] = df["ід"].astype(str).apply(lambda x: f"Елемент-{x}")
    if "тип" not in df.columns:
        # якщо є "потужність_мвт" — зробимо просту інтерпретацію типу
        def infer_type_row(r):
            try:
                p = float(r.get("потужність_мвт", 0))
                if p > 6000:
                    return "Генератор"
                if p > 5200:
                    return "Підстанція"
            except Exception:
                pass
            return "Лінія"
        df["тип"] = df.apply(infer_type_row, axis=1)

    root = tk.Tk()
    root.title(f"Система моніторингу завантаженості енергосистеми — {source_label}")
    root.geometry("1350x1000")

    # --- Сучасна стилізація ---
    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass
    # Загальні налаштування шрифтів і кнопок
    style.configure(".", font=("Segoe UI", 10))
    style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
    style.configure("TButton", padding=6)
    style.configure("Info.TLabel", background="#f0f4ff", foreground="#0b3d91", font=("Segoe UI", 10, "bold"))
    # Сучасні кольори для прогресбару (win32 ttk may ignore some options)
    style.configure("green.Horizontal.TProgressbar", troughcolor="#e6f4ea", background="#2a9d8f")

    # Сучасна тема для графіків
    sns.set_theme(style="whitegrid", palette="deep")
    # Спроба застосувати один із сучасних стилів; якщо недоступний — використаємо те, що є від seaborn
    preferred_styles = ["seaborn-darkgrid", "seaborn-v0_8-darkgrid", "ggplot"]
    for s in preferred_styles:
        if s in plt.style.available:
            plt.style.use(s)
            break
    # якщо жоден з перелічених не доступний — залишаємо rcParams від seaborn (без додаткового стилю)
    plt.rcParams.update({
        "figure.dpi": 100,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9
    })

    title = ttk.Label(root, text=f"Система моніторингу завантаженості енергосистеми ({source_label})",
                      style="Title.TLabel")
    title.pack(pady=10)

    top_frame = ttk.Frame(root)
    top_frame.pack(pady=5)

    # верхня панель: вибір запису — індекс рядка та Combobox з ID, коротка картка запису, ризик та індикатор
    ttk.Label(top_frame, text="Оберіть елемент мережі (ID) або індекс рядка:").pack(side=tk.LEFT, padx=5)

    obj_var = tk.IntVar(value=0)
    spin = ttk.Spinbox(top_frame, from_=0, to=max(0, len(df)-1),
                       textvariable=obj_var, width=8)
    spin.pack(side=tk.LEFT, padx=5)

    # Combobox з ID (колонка "ід" якщо є)
    if "ід" in df.columns:
        id_list = df["ід"].astype(str).tolist()
    else:
        id_list = [str(i) for i in range(len(df))]
    id_var = tk.StringVar(value=id_list[0] if id_list else "")
    id_menu = ttk.Combobox(top_frame, textvariable=id_var, values=id_list, state="readonly", width=10)
    id_menu.pack(side=tk.LEFT, padx=5)

    def on_id_change(event=None):
        # при виборі ID знаходимо його індекс і встановлюємо спінбокс
        try:
            sel = id_var.get()
            idxs = df.index[df["ід"].astype(str) == sel].tolist() if "ід" in df.columns else []
            if idxs:
                spin_val = int(idxs[0])
                obj_var.set(spin_val)
        except Exception:
            pass
    id_menu.bind("<<ComboboxSelected>>", on_id_change)

    # Коротка картка поточного запису (id, навантаження, потужність, завантаженість)
    record_frame = ttk.Frame(top_frame)
    record_frame.pack(side=tk.LEFT, padx=10)
    record_summary_label = ttk.Label(record_frame, text="Елемент: —", justify=tk.LEFT)
    record_summary_label.pack()

    risk_label = ttk.Label(top_frame, text="", font=("Arial", 14))
    risk_label.pack(side=tk.LEFT, padx=15)

    # --- Індикатор поточного навантаження (праворуч) ---
    load_frame = ttk.Frame(top_frame)
    load_frame.pack(side=tk.RIGHT, padx=10)
    ttk.Label(load_frame, text="Поточне навантаження:", style="Info.TLabel").pack(side=tk.TOP, padx=4, pady=0)
    load_val_label = ttk.Label(load_frame, text="— МВт", font=("Segoe UI", 10, "bold"))
    load_val_label.pack(side=tk.TOP, anchor="e")
    load_bar = ttk.Progressbar(load_frame, style="green.Horizontal.TProgressbar", orient="horizontal", length=220, mode="determinate", maximum=100)
    load_bar.pack(side=tk.TOP, pady=4)

    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True)

    # --- Вкладки ---
    frame_surv = ttk.Frame(notebook)
    frame_hr = ttk.Frame(notebook)
    frame_ch = ttk.Frame(notebook)
    frame_groups = ttk.Frame(notebook)

    notebook.add(frame_surv, text="Прогноз виживаності")
    notebook.add(frame_hr, text="Вплив факторів (HR)")
    notebook.add(frame_ch, text="Кумулятивний ризик відмов")
    notebook.add(frame_groups, text="Виживаність по групах")

    # --- Графіки ---
    fig_surv, ax_surv = plt.subplots(figsize=(6, 4))
    fig_surv.patch.set_facecolor("#fafafa")
    ax_surv.set_facecolor("#ffffff")
    canvas_surv = FigureCanvasTkAgg(fig_surv, master=frame_surv)
    canvas_surv.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    fig_hr, ax_hr = plt.subplots(figsize=(6, 4))
    fig_hr.patch.set_facecolor("#fafafa")
    ax_hr.set_facecolor("#ffffff")
    canvas_hr = FigureCanvasTkAgg(fig_hr, master=frame_hr)
    canvas_hr.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    fig_ch, ax_ch = plt.subplots(figsize=(6, 4))
    fig_ch.patch.set_facecolor("#fafafa")
    ax_ch.set_facecolor("#ffffff")
    canvas_ch = FigureCanvasTkAgg(fig_ch, master=frame_ch)
    canvas_ch.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- Вкладка виживаності по групах ---
    group_controls = ttk.Frame(frame_groups)
    group_controls.pack(pady=5)

    # Доступні змінні для групування: всі фічі + (свято залишаємо, навіть якщо бінарна)
    groupable_vars = [c for c in FEATURE_COLUMNS if c in df.columns]
    group_var = tk.StringVar(value=groupable_vars[0] if groupable_vars else FEATURE_COLUMNS[0])

    ttk.Label(group_controls, text="Групувати за змінною:").pack(side=tk.LEFT, padx=5)
    group_menu = ttk.Combobox(group_controls, textvariable=group_var, values=groupable_vars, state="readonly")
    group_menu.pack(side=tk.LEFT, padx=5)

    fig_groups, ax_groups = plt.subplots(figsize=(6, 4))
    canvas_groups = FigureCanvasTkAgg(fig_groups, master=frame_groups)
    canvas_groups.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- Текстовий звіт ---
    report_text = tk.Text(root, height=16, wrap="word", font=("Arial", 11))
    report_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def update_dashboard():
        # Безпека індексу
        try:
            idx = int(obj_var.get())
        except Exception:
            idx = 0
            obj_var.set(0)

        if idx < 0 or idx >= len(df):
            messagebox.showerror("Помилка", f"Немає елемента мережі з індексом {idx}")
            return

        row = df.iloc[[idx]][FEATURE_COLUMNS]

        # Оновити Combobox ID щоб відображав реальний 'ід' поточного рядка (якщо є)
        try:
            if "ід" in df.columns:
                current_id = str(df.iloc[idx]["ід"])
                if id_var.get() != current_id:
                    id_var.set(current_id)
        except Exception:
            pass

        # --- Survival ---
        surv = cph.predict_survival_function(row)
        ax_surv.clear()
        yvals = surv.values.flatten()
        ax_surv.plot(surv.index, yvals, linewidth=2, color="blue", label="Ймовірність виживання")
        # Медіана виживаності (перше t, де S(t) <= 0.5)
        try:
            mask = yvals <= 0.5
            if mask.any():
                t50 = surv.index[np.argmax(mask)]  # перший True
                ax_surv.axvline(t50, color="red", linestyle="--", label=f"Медіана ≈ {t50:.0f}")
        except Exception:
            pass
        ax_surv.set_title(f"Крива виживаності для елемента #{idx}")
        ax_surv.set_xlabel("Час")
        ax_surv.set_ylabel("Ймовірність")
        ax_surv.legend()
        canvas_surv.draw()

        # --- Hazard Ratios ---
        ax_hr.clear()
        summary = cph.summary
        if not summary.empty:
            hr = summary["exp(coef)"].sort_values()
            colors = ["red" if v > 1 else "green" for v in hr.values]
            ax_hr.barh(hr.index, hr.values, color=colors)
            ax_hr.set_title("Hazard Ratios")
            ax_hr.set_xlabel("HR")
        else:
            ax_hr.text(0.5, 0.5, "Немає коефіцієнтів", ha="center", va="center")
        canvas_hr.draw()

        # --- Кумулятивний ризик ---
        ch = cph.predict_cumulative_hazard(row)
        ax_ch.clear()
        ax_ch.plot(ch.index, ch.values.flatten(), linewidth=2, color="purple")
        ax_ch.set_title(f"Кумулятивний ризик для елемента #{idx}")
        ax_ch.set_xlabel("Час")
        ax_ch.set_ylabel("Кумулятивний ризик")
        canvas_ch.draw()

        # --- Виживаність по групах ---
        ax_groups.clear()
        kmf = KaplanMeierFitter()
        var = group_var.get()

        # Якщо вибраної змінної немає (наприклад, у CSV), підставимо першу доступну
        if var not in df.columns:
            if groupable_vars:
                var = groupable_vars[0]
                group_var.set(var)
            else:
                ax_groups.text(0.5, 0.5, "Немає змінних для групування", ha="center", va="center")
                canvas_groups.draw()
                return

        unique_vals = df[var].nunique()
        # Числова і має достатньо різних значень → квантили
        if pd.api.types.is_numeric_dtype(df[var]) and unique_vals > 3:
            df["_group"] = pd.qcut(df[var], q=3,
                                   labels=["Низьке", "Середнє", "Високе"],
                                   duplicates="drop")
        else:
            # Категоріальна або двійкова/мало унікальних → як є
            df["_group"] = df[var].astype(str)

        for val in sorted(df["_group"].unique()):
            mask = df["_group"] == val
            if mask.sum() == 0:
                continue
            kmf.fit(df.loc[mask, DURATION_COL],
                    event_observed=df.loc[mask, EVENT_COL])
            kmf.plot_survival_function(ax=ax_groups, label=f"{var}={val}")

        ax_groups.set_title(f"Крива виживаності по групах ({var})")
        ax_groups.set_xlabel("Час")
        ax_groups.set_ylabel("Ймовірність виживання")
        ax_groups.legend()
        canvas_groups.draw()

        # --- Risk label ---
        risk_score = float(cph.predict_partial_hazard(row).values[0])
        # Інтерпретація часткового ризику у термінах навантаження/стану
        if risk_score < 0.8:
            risk_label.config(text="🟢 Низьке навантаження", foreground="green")
        elif risk_score < 1.2:
            risk_label.config(text="🟡 Помірне навантаження", foreground="orange")
        else:
            risk_label.config(text="🔴 Високе навантаження", foreground="red")

        # --- Оновлення індикатора поточного навантаження ---
        try:
            curr_load = float(row["навантаження_мвт"].values[0])
            min_load, max_load = float(df["навантаження_мвт"].min()), float(df["навантаження_мвт"].max())
            if max_load > min_load:
                pct = (curr_load - min_load) / (max_load - min_load) * 100
            else:
                pct = 0.0
            pct = max(0.0, min(100.0, pct))
            load_bar['value'] = pct
            load_val_label.config(text=f"{curr_load:.0f} МВт  ({pct:.0f}%)")
        except Exception:
            load_bar['value'] = 0
            load_val_label.config(text="— МВт")

        # --- Оновити картку запису (коротка інформація) ---
        try:
            info_parts = []
            if "ід" in df.columns:
                info_parts.append(f"ID: {df.iloc[idx]['ід']}")
            # додамо назву/тип якщо є
            if "назва" in df.columns:
                info_parts.insert(0, f"{df.iloc[idx]['назва']}")
            if "тип" in df.columns:
                info_parts.insert(1, f"({df.iloc[idx]['тип']})")
            if "навантаження_мвт" in row.columns:
                info_parts.append(f"Навантаження: {float(row['навантаження_мвт']):.0f} МВт")
            if "потужність_мвт" in row.columns:
                info_parts.append(f"Потужність: {float(row['потужність_мвт']):.0f} МВт")
            if "завантаженість" in row.columns:
                info_parts.append(f"Завантаженість: {float(row['завантаженість']):.2f}")
            record_summary_label.config(text="  • ".join(info_parts) if info_parts else "Елемент: —")
        except Exception:
            record_summary_label.config(text="Елемент: —")

        # --- Оновити панель детальної інформації (повний рядок + інтерпретація) ---
        try:
            details_text.delete("1.0", tk.END)
            # повний рядок (усі колонки)
            full_row = df.iloc[idx].to_dict()
            details_text.insert(tk.END, "Повні дані елемента:\n")
            for k, v in full_row.items():
                details_text.insert(tk.END, f"  {k}: {v}\n")
            # якщо немає 'тип' — додамо інтерпретацію і підкажемо користувачу
            if ("тип" not in df.columns) or (not str(full_row.get("тип")).strip()):
                inferred = infer_type_row(full_row) if 'infer_type_row' in globals() else "Н/д"
                details_text.insert(tk.END, f"\nІнтерпретація типу: {inferred}\n")
        except Exception:
            pass

        # --- Автоматичний звіт ---
        report_text.delete("1.0", tk.END)
        if not summary.empty:
            report_text.insert(tk.END, "📊 Звіт моніторингу завантаженості (модель Кокса)\n\n")
            sorted_summary = summary.reindex(
                summary["exp(coef)"].sub(1).abs().sort_values(ascending=False).index
            )
            for feature, row_sum in sorted_summary.iterrows():
                hr_val = row_sum["exp(coef)"]
                ci_low = row_sum["exp(coef) lower 95%"]
                ci_high = row_sum["exp(coef) upper 95%"]
                percent = (hr_val - 1) * 100
                if hr_val > 1.1:
                    effect = f"підвищує ризик приблизно на {percent:.1f}%"
                elif hr_val < 0.9:
                    effect = f"знижує ризик приблизно на {abs(percent):.1f}%"
                else:
                    effect = "майже не впливає на ризик"
                report_text.insert(
                    tk.END,
                    f"• {feature}: {effect} (HR={hr_val:.2f}, CI {ci_low:.2f}–{ci_high:.2f})\n"
                )
            report_text.insert(tk.END, f"\nНайбільший вплив має: {sorted_summary.index[0]}\n")
            report_text.insert(tk.END, f"Індивідуальний стан навантаження елемента #{idx}: {risk_label.cget('text')}\n")
        else:
            report_text.insert(tk.END, "⚠️ Немає коефіцієнтів для формування звіту")

    def regenerate_data():
        nonlocal df, cph, source_label, groupable_vars
        df = generate_random_data(150)
        cph = train_cox_model(df.copy())
        obj_var.set(0)
        source_label = "випадкові дані"
        title.config(text=f"Система моніторингу завантаженості енергосистеми ({source_label})")
        root.title(f"Система моніторингу завантаженості енергосистеми — {source_label}")
        # Оновити список змінних для групування
        groupable_vars = [c for c in FEATURE_COLUMNS if c in df.columns]
        group_menu["values"] = groupable_vars
        if groupable_vars:
            group_var.set(groupable_vars[0])
        # Оновити список ID та спінбокс верхнього контролу
        try:
            if "ід" in df.columns:
                new_ids = df["ід"].astype(str).tolist()
            else:
                new_ids = [str(i) for i in range(len(df))]
            id_menu["values"] = new_ids
            if new_ids:
                id_var.set(new_ids[0])
        except Exception:
            pass
        # Якщо немає колонок назва/тип — додати їх для тестових даних (вже робиться у генераторі)
        if "назва" not in df.columns:
            df["назва"] = df["ід"].astype(str).apply(lambda x: f"Елемент-{x}")
        if "тип" not in df.columns:
            df["тип"] = df["потужність_мвт"].apply(lambda p: ("Генератор" if p>6000 else ("Підстанція" if p>5200 else "Лінія")))
        spin.config(to=max(0, len(df)-1))
        update_dashboard()

    # Кнопки
    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=5)
    ttk.Button(btn_frame, text="🔄 Згенерувати тестові дані", command=regenerate_data).pack(side=tk.LEFT, padx=5)
    ttk.Button(btn_frame, text="Оновити", command=update_dashboard).pack(side=tk.LEFT, padx=5)
    # Кнопка показу детальної картки елемента
    show_details = tk.BooleanVar(value=False)
    def toggle_details():
        if show_details.get():
            details_frame.pack_forget()
            show_details.set(False)
            details_btn.config(text="Показати деталі")
        else:
            details_frame.pack(fill=tk.BOTH, expand=False, padx=10, pady=5)
            show_details.set(True)
            details_btn.config(text="Сховати деталі")
    details_btn = ttk.Button(btn_frame, text="Показати деталі", command=toggle_details)
    details_btn.pack(side=tk.LEFT, padx=5)

    # --- Панель детальної інформації (прихована за замовчуванням) ---
    details_frame = ttk.Frame(root, relief=tk.RIDGE, padding=6)
    details_text = tk.Text(details_frame, height=6, wrap="word", font=("Segoe UI", 10))
    details_text.pack(fill=tk.BOTH, expand=True)
    # початково прихована; показується кнопкою

    # Початкове оновлення та запуск
    update_dashboard()
    root.mainloop()

if __name__ == "__main__":
    run_dashboard()
