import os
import html
import base64
from io import BytesIO
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import customtkinter as ctk
from tkinter import filedialog, messagebox

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


NUMERIC_COLUMNS = ["revenue", "expenses", "customers", "orders", "profit"]


@dataclass
class MetricRow:
    total: float
    mean: float
    first: float
    last: float
    abs_change: float
    pct_change: float
    trend_slope: float


def _trend_slope(y: pd.Series) -> float:
    """Linear trend slope for numeric series (y vs index)."""
    yv = pd.to_numeric(y, errors="coerce").astype(float).to_numpy()
    mask = np.isfinite(yv)
    if mask.sum() < 2:
        return float("nan")
    x = np.arange(len(yv), dtype=float)[mask]
    y2 = yv[mask]
    slope = np.polyfit(x, y2, 1)[0]
    return float(slope)


def compute_metrics(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    df = df.sort_values("date").copy()

    # Keep only columns we know how to analyze
    numeric_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    if not numeric_cols:
        raise ValueError("В CSV не найдены ожидаемые числовые колонки.")

    # Percent change might produce inf; normalize to NaN
    df_numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    totals = df_numeric.sum()
    means = df_numeric.mean()

    first = df_numeric.iloc[0]
    last = df_numeric.iloc[-1]
    abs_change = last - first
    pct_change = (abs_change / first.replace(0, np.nan)) * 100.0

    trend_slopes = {c: _trend_slope(df_numeric[c]) for c in numeric_cols}

    metrics = []
    for c in numeric_cols:
        metrics.append(
            {
                "metric": c,
                "total": float(totals[c]),
                "mean": float(means[c]),
                "first": float(first[c]),
                "last": float(last[c]),
                "abs_change": float(abs_change[c]),
                "pct_change": float(pct_change[c]) if np.isfinite(pct_change[c]) else float("nan"),
                "trend_slope": float(trend_slopes[c]),
            }
        )

    metrics_df = pd.DataFrame(metrics).set_index("metric")

    # Долевые показатели: доля в абсолютной сумме (для корректности при отрицательных profit)
    abs_totals = df_numeric.abs().sum()
    shares_df = (abs_totals / abs_totals.sum()).to_frame("share")

    # Корреляции (зависимости): Пирсон
    corr_df = df_numeric.dropna().corr(numeric_only=True)

    extra = {
        "numeric_cols": numeric_cols,
        "corr_df": corr_df,
    }
    return metrics_df, shares_df, extra


def summarize_trends(metrics_df: pd.DataFrame) -> List[str]:
    conclusions = []
    for metric, row in metrics_df.iterrows():
        slope = row.get("trend_slope", float("nan"))
        pct = row.get("pct_change", float("nan"))
        if not np.isfinite(slope) or np.isnan(slope):
            continue
        direction = "в целом растет" if slope > 0 else "в целом снижается"
        if np.isfinite(pct):
            conclusions.append(f"{metric}: {direction} (изменение за период: {pct:.1f}%).")
        else:
            conclusions.append(f"{metric}: {direction}.")
    return conclusions


def build_comparison(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    numeric_cols = [c for c in NUMERIC_COLUMNS if c in df.columns]
    if not numeric_cols:
        raise ValueError("Нет числовых колонок для сравнения.")

    n = len(df)
    if n < 4:
        raise ValueError("Недостаточно данных для сравнения периодов.")

    mid = n // 2
    first = df.iloc[:mid]
    second = df.iloc[mid:]

    first_means = first[numeric_cols].mean(numeric_only=True)
    second_means = second[numeric_cols].mean(numeric_only=True)
    diff = second_means - first_means
    pct = (diff / first_means.replace(0, np.nan)) * 100.0

    comp = pd.DataFrame(
        {
            "mean_first_half": first_means,
            "mean_second_half": second_means,
            "diff_second_minus_first": diff,
            "pct_change": pct,
        }
    )
    comp.index.name = "metric"
    return comp


def top_correlations(corr_df: pd.DataFrame, k: int = 5) -> List[str]:
    if corr_df is None or corr_df.empty:
        return []

    # Flatten upper triangle (excluding diagonal)
    cols = corr_df.columns.tolist()
    pairs = []
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            a, b = cols[i], cols[j]
            r = corr_df.loc[a, b]
            if not np.isfinite(r):
                continue
            pairs.append((abs(r), r, a, b))

    pairs.sort(reverse=True, key=lambda x: x[0])
    out = []
    for _, r, a, b in pairs[:k]:
        sign = "положительная" if r > 0 else "отрицательная"
        out.append(f"{a} и {b}: {sign} корреляция Пирсона r={r:.2f}.")
    return out


class AnalyticsApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Анализ показателей деятельности организации (PP12_ISP22)")
        self.geometry("1800x1120")

        self.file_path_var = ctk.StringVar()
        self.metrics_df: Optional[pd.DataFrame] = None
        self.shares_df: Optional[pd.DataFrame] = None
        self.corr_df: Optional[pd.DataFrame] = None
        self.df: Optional[pd.DataFrame] = None
        self._heatmap_holder: Optional[ctk.CTkFrame] = None

        # Cached rendered graphs for stable embedding and report export.
        self._plot_widgets: List[ctk.CTkLabel] = []
        # Cache rendered PNGs in memory only (no files on disk).
        self._generated_pngs: Dict[str, bytes] = {}
        self._generated_image_order: List[str] = []

        self._build_ui()

    def _build_ui(self):
        ui_font = ctk.CTkFont(family="Arial", size=13)
        ui_font_big = ctk.CTkFont(family="Arial", size=14, weight="bold")

        # Top bar
        top = ctk.CTkFrame(self)
        top.pack(fill="x", padx=12, pady=10)

        ctk.CTkLabel(top, text="CSV файл:", font=ui_font_big).pack(side="left", padx=(12, 8), pady=10)
        ctk.CTkEntry(top, textvariable=self.file_path_var, width=650).pack(
            side="left", padx=(0, 8), pady=10, expand=True, fill="x"
        )
        ctk.CTkButton(top, text="Выбрать файл...", command=self.on_choose_file).pack(
            side="left", padx=8, pady=10
        )
        ctk.CTkButton(top, text="Загрузить и проанализировать", command=self.on_load_analyze).pack(
            side="left", padx=10, pady=10
        )

        # Body
        body = ctk.CTkFrame(self)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        self.tabs = ctk.CTkTabview(body)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_overview = self.tabs.add("Обзор")
        self.tab_graphs = self.tabs.add("Графики")
        self.tab_compare = self.tabs.add("Сравнение периодов")
        self.tab_deps = self.tabs.add("Зависимости (корреляции)")

        # Overview tab
        self.text_overview = ctk.CTkTextbox(self.tab_overview, wrap="word", font=ui_font)
        self.text_overview.pack(fill="both", expand=True, padx=10, pady=10)

        # Graphs tab layout
        graphs_outer = ctk.CTkFrame(self.tab_graphs)
        graphs_outer.pack(fill="both", expand=True, padx=10, pady=10)

        self.graphs_row1 = ctk.CTkFrame(graphs_outer, height=560)
        # Allow first row to take vertical space too (charts stay readable).
        self.graphs_row1.pack(fill="both", expand=True, padx=10, pady=(10, 6))

        self.graph_frame_line = ctk.CTkFrame(self.graphs_row1)
        self.graph_frame_line.pack(side="left", fill="both", expand=True, padx=(0, 6), pady=6)
        ctk.CTkLabel(
            self.graph_frame_line,
            text="Линейные тренды: revenue/expenses/profit",
            font=ui_font_big,
        ).pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.graph_line_canvas_holder = ctk.CTkFrame(self.graph_frame_line, fg_color="transparent", height=430)
        self.graph_line_canvas_holder.pack(fill="both", expand=True, padx=10, pady=6)

        self.graph_frame_line2 = ctk.CTkFrame(self.graphs_row1)
        self.graph_frame_line2.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        ctk.CTkLabel(self.graph_frame_line2, text="Линейные тренды: customers/orders", font=ui_font_big).pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.graph_line2_canvas_holder = ctk.CTkFrame(self.graph_frame_line2, fg_color="transparent", height=430)
        self.graph_line2_canvas_holder.pack(fill="both", expand=True, padx=10, pady=6)

        # Row2 charts
        self.graphs_row2 = ctk.CTkFrame(graphs_outer, height=560)
        self.graphs_row2.pack(fill="both", expand=True, padx=10, pady=(6, 10))

        self.graph_frame_bar = ctk.CTkFrame(self.graphs_row2)
        self.graph_frame_bar.pack(side="left", fill="both", expand=True, padx=(0, 6), pady=6)
        ctk.CTkLabel(self.graph_frame_bar, text="Суммы по месяцам", font=ui_font_big).pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.graph_bar_canvas_holder = ctk.CTkFrame(self.graph_frame_bar, fg_color="transparent", height=430)
        self.graph_bar_canvas_holder.pack(fill="both", expand=True, padx=10, pady=6)

        self.graph_frame_pie = ctk.CTkFrame(self.graphs_row2)
        self.graph_frame_pie.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        ctk.CTkLabel(self.graph_frame_pie, text="Доли (в абсолютной сумме)", font=ui_font_big).pack(
            anchor="w", padx=10, pady=(10, 0)
        )
        self.graph_pie_canvas_holder = ctk.CTkFrame(self.graph_frame_pie, fg_color="transparent", height=430)
        self.graph_pie_canvas_holder.pack(fill="both", expand=True, padx=10, pady=6)

        # Compare tab
        self.text_compare = ctk.CTkTextbox(self.tab_compare, wrap="word", font=ui_font)
        self.text_compare.pack(fill="both", expand=True, padx=10, pady=10)

        # Deps tab
        self.text_deps = ctk.CTkTextbox(self.tab_deps, wrap="word", font=ui_font)
        self.text_deps.pack(fill="both", expand=True, padx=10, pady=10)

        # Bottom buttons
        bottom = ctk.CTkFrame(self)
        bottom.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkButton(bottom, text="Сохранить отчет...", command=self.on_save_report).pack(
            side="left", padx=12, pady=10
        )
        ctk.CTkLabel(
            bottom,
            text="Отчет включает: описание данных, расчеты, графики и аналитические выводы.",
            text_color="gray",
        ).pack(side="left", padx=12, pady=10)

    def on_choose_file(self):
        default_path = os.path.join(os.getcwd(), "PP12_ISP22_analytics.csv")
        file_path = filedialog.askopenfilename(
            title="Выберите CSV файл",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=os.path.dirname(default_path),
        )
        if file_path:
            self.file_path_var.set(file_path)

    def _set_default_file(self):
        # Try to auto-fill if the file exists next to script
        candidate = os.path.join(os.getcwd(), "PP12_ISP22_analytics.csv")
        if os.path.exists(candidate) and not self.file_path_var.get():
            self.file_path_var.set(candidate)

    def _clear_plots(self):
        # Destroy all previous plot widgets
        for widget in self._plot_widgets:
            try:
                widget.destroy()
            except Exception:
                pass
        self._plot_widgets = []
        if self._heatmap_holder is not None:
            try:
                self._heatmap_holder.destroy()
            except Exception:
                pass
            self._heatmap_holder = None

    def on_load_analyze(self):
        self._set_default_file()
        path = self.file_path_var.get().strip()
        if not path or not os.path.exists(path):
            messagebox.showerror("Ошибка", "Выберите существующий CSV файл.")
            return

        try:
            df = pd.read_csv(path)
        except Exception as e:
            messagebox.showerror("Ошибка чтения CSV", str(e))
            return

        # Basic validation
        if "date" not in df.columns:
            messagebox.showerror("Ошибка данных", "В CSV нет колонки `date`.")
            return

        # Parse date and clean numeric columns
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        for c in NUMERIC_COLUMNS:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        # Удаление пропусков: убираем строки, где пропущены date или значения из наших numeric колонки
        required = ["date"] + [c for c in NUMERIC_COLUMNS if c in df.columns]
        df = df.dropna(subset=required).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

        if len(df) < 2:
            messagebox.showerror("Ошибка данных", "После очистки осталось слишком мало строк.")
            return

        try:
            metrics_df, shares_df, extra = compute_metrics(df)
            corr_df = extra.get("corr_df", None)
            comp = build_comparison(df)
            deps_list = top_correlations(corr_df, k=5)
            trends_list = summarize_trends(metrics_df)

            self.df = df
            self.metrics_df = metrics_df
            self.shares_df = shares_df
            self.corr_df = corr_df

            # Fill overview text
            overview_text = self._format_overview(df, metrics_df, shares_df, trends_list, deps_list)
            self.text_overview.configure(state="normal")
            self.text_overview.delete("1.0", "end")
            self.text_overview.insert("1.0", overview_text)
            self.text_overview.configure(state="disabled")

            # Compare tab text
            compare_text = self._format_comparison(comp)
            self.text_compare.configure(state="normal")
            self.text_compare.delete("1.0", "end")
            self.text_compare.insert("1.0", compare_text)
            self.text_compare.configure(state="disabled")

            deps_text = self._format_deps(corr_df, deps_list)
            self.text_deps.configure(state="normal")
            self.text_deps.delete("1.0", "end")
            self.text_deps.insert("1.0", deps_text)
            self.text_deps.configure(state="disabled")

            # Draw plots
            self._clear_plots()
            # Ensure layout is computed before we measure holder sizes.
            self.update_idletasks()
            self.update()
            # Reset in-memory PNG cache
            self._generated_pngs = {}
            self._generated_image_order = []
            self._draw_graphs(df, metrics_df, shares_df, corr_df)

        except Exception as e:
            messagebox.showerror("Ошибка анализа", str(e))

    def _format_overview(
        self,
        df: pd.DataFrame,
        metrics_df: pd.DataFrame,
        shares_df: pd.DataFrame,
        trends_list: List[str],
        deps_list: List[str],
    ) -> str:
        # Data description
        col_list = ", ".join([c for c in df.columns if c != "date"])
        start_date = df["date"].iloc[0].date()
        end_date = df["date"].iloc[-1].date()

        lines = []
        lines.append("ОПИСАНИЕ ДАННЫХ")
        lines.append(f"Файл: {os.path.basename(self.file_path_var.get())}")
        lines.append(f"Период: {start_date} — {end_date}")
        lines.append(f"Строк (после очистки): {len(df)}")
        lines.append(f"Колонки показателей: {col_list}")
        lines.append("")
        lines.append("РАСЧЕТ ПОКАЗАТЕЛЕЙ")
        lines.append("Колонка | Сумма | Среднее | Изменение (последнее - первое) | Динамика % | Тренд (наклон)")

        # stable ordering
        for metric in metrics_df.index.tolist():
            row = metrics_df.loc[metric]
            pct = row["pct_change"]
            pct_s = "н/д" if not np.isfinite(pct) else f"{pct:.1f}%"
            lines.append(
                f"{metric} | {row['total']:.2f} | {row['mean']:.2f} | {row['abs_change']:.2f} | {pct_s} | {row['trend_slope']:.6f}"
            )

        lines.append("")
        lines.append("ДОЛЕВЫЕ ПОКАЗАТЕЛИ (доля в абсолютной сумме)")
        for metric in shares_df.index.tolist():
            lines.append(f"{metric}: {shares_df.loc[metric, 'share'] * 100.0:.2f}%")

        lines.append("")
        lines.append("АНАЛИТИЧЕСКИЕ ВЫВОДЫ")
        if trends_list:
            lines.append("Тренды:")
            for t in trends_list:
                lines.append(f"- {t}")

        if deps_list:
            lines.append("")
            lines.append("Зависимости (наиболее сильные корреляции):")
            for d in deps_list:
                lines.append(f"- {d}")

        return "\n".join(lines)

    def _format_comparison(self, comp: pd.DataFrame) -> str:
        lines = []
        lines.append("СРАВНЕНИЕ ПЕРИОДОВ")
        lines.append("Сравнение выполнено по средним значениям: первая половина vs вторая половина.")
        lines.append("")
        lines.append("metric | mean_first_half | mean_second_half | diff (2nd-1st) | pct_change")
        for metric in comp.index.tolist():
            r = comp.loc[metric]
            pct = r["pct_change"]
            pct_s = "н/д" if not np.isfinite(pct) else f"{pct:.1f}%"
            lines.append(
                f"{metric} | {r['mean_first_half']:.2f} | {r['mean_second_half']:.2f} | {r['diff_second_minus_first']:.2f} | {pct_s}"
            )
        lines.append("")
        # Short conclusion
        best = comp["pct_change"].replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
        worst = comp["pct_change"].replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=True)
        if len(best) > 0:
            lines.append(f"Наиболее вырос показатель: {best.index[0]} ({best.iloc[0]:.1f}%).")
        if len(worst) > 0:
            lines.append(f"Наиболее снизился показатель: {worst.index[0]} ({worst.iloc[0]:.1f}%).")
        return "\n".join(lines)

    def _format_deps(self, corr_df: pd.DataFrame, deps_list: List[str]) -> str:
        lines = []
        lines.append("ЗАВИСИМОСТИ (КОРРЕЛЯЦИИ)")
        lines.append("Используется корреляция Пирсона по числовым колонкам.")
        lines.append("")
        if deps_list:
            for d in deps_list:
                lines.append(f"- {d}")
        else:
            lines.append("Недостаточно данных для расчета корреляций.")
        return "\n".join(lines)

    def _embed_figure(
        self,
        fig: plt.Figure,
        holder: ctk.CTkFrame,
        img_filename: Optional[str] = None,
    ):
        """
        Renders a Matplotlib figure into a PNG image and embeds it into Tk as `CTkImage`.
        Using `Agg` backend avoids Tk "after" callback issues from `FigureCanvasTkAgg`.
        """
        png_bytes: bytes
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=220, bbox_inches="tight", pad_inches=0.12)
        png_bytes = buf.getvalue()
        buf.close()

        # Close figure to release memory/resources.
        plt.close(fig)

        # Convert bytes -> PIL image (Pillow is an indirect Matplotlib dependency).
        from PIL import Image  # local import to keep top-level imports light

        img = Image.open(BytesIO(png_bytes)).convert("RGBA")

        # Scale the image to the holder size to keep it readable (no clipping).
        try:
            holder.update_idletasks()
            target_w = holder.winfo_width()
            target_h = holder.winfo_height()
            # If tab/frame is not fully visible yet, winfo_width/height can be too small.
            req_w = holder.winfo_reqwidth()
            req_h = holder.winfo_reqheight()
            target_w = max(target_w, req_w)
            target_h = max(target_h, req_h)
        except Exception:
            target_w = 0
            target_h = 0

        # Fallback if widgets not laid out yet.
        if target_w <= 2 or target_h <= 2:
            target_w, target_h = 1200, 700

        # Add a small padding so axes labels don't touch edges.
        pad = 10
        target_w = max(450, target_w - pad)
        target_h = max(300, target_h - pad)

        # Scale to fit the holder (allow a bit of upscaling to improve readability).
        scale = min(target_w / img.width, target_h / img.height)
        # Allow moderate upscaling for readability, but avoid huge overflows.
        scale = min(scale, 1.7)
        new_w = max(1, int(img.width * scale))
        new_h = max(1, int(img.height * scale))
        if new_w != img.width or new_h != img.height:
            img = img.resize((new_w, new_h), resample=Image.Resampling.LANCZOS)

        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(img.width, img.height))
        label = ctk.CTkLabel(holder, text="", image=ctk_img)
        label.image = ctk_img  # keep reference to prevent garbage-collection
        label.pack(fill="both", expand=True)

        self._plot_widgets.append(label)

        if img_filename:
            # Store for report export (in-memory only).
            self._generated_pngs[img_filename] = png_bytes
            if img_filename not in self._generated_image_order:
                self._generated_image_order.append(img_filename)

    def _draw_graphs(self, df: pd.DataFrame, metrics_df: pd.DataFrame, shares_df: pd.DataFrame, corr_df: pd.DataFrame):
        # 1) line plot for revenue/expenses/profit
        fig1, ax1 = plt.subplots(figsize=(9.2, 5.6), dpi=100)
        dates = df["date"]
        cols_line = [c for c in ["revenue", "expenses", "profit"] if c in df.columns]
        for c in cols_line:
            ax1.plot(dates, df[c], label=c)
        ax1.set_title("Тренды: revenue, expenses, profit")
        ax1.set_xlabel("date")
        ax1.set_ylabel("value")
        ax1.grid(True, alpha=0.25)
        # Reduce tick density for readability.
        ax1.xaxis.set_major_locator(MaxNLocator(6))
        ax1.tick_params(axis="x", labelsize=8, rotation=0)
        ax1.legend(loc="best")
        fig1.tight_layout()
        self._embed_figure(fig1, self.graph_line_canvas_holder, img_filename="graph_01.png")

        # 2) line plot for customers/orders
        fig2, ax2 = plt.subplots(figsize=(9.2, 5.6), dpi=100)
        cols_line2 = [c for c in ["customers", "orders"] if c in df.columns]
        for c in cols_line2:
            ax2.plot(dates, df[c], label=c)
        ax2.set_title("Тренды: customers, orders")
        ax2.set_xlabel("date")
        ax2.set_ylabel("value")
        ax2.grid(True, alpha=0.25)
        ax2.xaxis.set_major_locator(MaxNLocator(6))
        ax2.tick_params(axis="x", labelsize=8, rotation=0)
        ax2.legend(loc="best")
        fig2.tight_layout()
        self._embed_figure(fig2, self.graph_line2_canvas_holder, img_filename="graph_02.png")

        # 3) bar chart for monthly sums
        df_month = df.copy()
        df_month["month"] = df_month["date"].dt.to_period("M").astype(str)
        group = df_month.groupby("month", sort=True)
        months = group.size().index.tolist()

        fig3, ax3 = plt.subplots(figsize=(10.0, 5.6), dpi=100)
        bar_cols = [c for c in ["revenue", "expenses", "profit"] if c in df.columns]
        x = np.arange(len(months))
        width = 0.25
        for i, c in enumerate(bar_cols):
            values = group[c].sum().reindex(months).to_numpy()
            ax3.bar(x + i * width, values, width=width, label=c)
        ax3.set_title("Суммы по месяцам")
        ax3.set_xlabel("month")
        ax3.set_ylabel("sum")
        centers = x + width
        # Keep month labels readable by showing only a subset if needed.
        if len(months) > 10:
            step = max(1, len(months) // 8)
            tick_pos = centers[::step]
            tick_labels = np.array(months, dtype=str)[::step].tolist()
        else:
            tick_pos = centers
            tick_labels = months
        ax3.set_xticks(tick_pos)
        ax3.set_xticklabels(tick_labels, rotation=45, ha="right", fontsize=8)
        ax3.tick_params(axis="x", labelsize=8)
        ax3.grid(True, axis="y", alpha=0.25)
        ax3.legend(loc="best")
        fig3.tight_layout()
        self._embed_figure(fig3, self.graph_bar_canvas_holder, img_filename="graph_03.png")

        # 4) pie chart for shares
        fig4, ax4 = plt.subplots(figsize=(8.2, 5.6), dpi=100)
        share_items = shares_df.index.tolist()
        share_values = shares_df["share"].reindex(share_items).to_numpy()
        labels = [str(x) for x in share_items]
        ax4.pie(share_values, labels=labels, autopct="%1.1f%%", startangle=90)
        ax4.set_title("Доли в абсолютной сумме")
        fig4.tight_layout()
        self._embed_figure(fig4, self.graph_pie_canvas_holder, img_filename="graph_04.png")

        # 5) correlation heatmap inside overview tab (reuse Deps tab text with chart)
        if corr_df is not None and not corr_df.empty:
            # We'll add heatmap into Deps tab by creating a temporary holder at bottom.
            if self._heatmap_holder is None:
                self._heatmap_holder = ctk.CTkFrame(self.tab_deps, fg_color="transparent", height=340)
                self._heatmap_holder.pack(fill="x", expand=False, padx=10, pady=(10, 10))
            heat_holder = self._heatmap_holder
            fig5, ax5 = plt.subplots(figsize=(10.0, 6.2), dpi=100, constrained_layout=True)
            corr = corr_df.copy()
            corr = corr.reindex(index=corr.columns, columns=corr.columns)
            n = corr.shape[0]
            im = ax5.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, interpolation="nearest")
            ax5.set_title("Корреляционная матрица (Пирсон)")
            tick_labels = corr.columns.tolist()
            ax5.set_xticks(np.arange(len(tick_labels)))
            ax5.set_yticks(np.arange(len(tick_labels)))
            ax5.set_xticklabels(tick_labels, rotation=45, ha="right")
            ax5.set_yticklabels(tick_labels)
            # Ensure the heatmap grid occupies the center of the axes area.
            ax5.set_aspect("equal", adjustable="box")
            ax5.set_xlim(-0.5, n - 0.5)
            ax5.set_ylim(n - 0.5, -0.5)

            # Annotate values
            for i in range(len(tick_labels)):
                for j in range(len(tick_labels)):
                    val = corr.values[i, j]
                    if np.isfinite(val):
                        ax5.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
            fig5.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)
            self._embed_figure(fig5, heat_holder, img_filename="graph_05.png")

    def on_save_report(self):
        if self.df is None or self.metrics_df is None or self.shares_df is None:
            messagebox.showerror("Ошибка", "Сначала выполните загрузку и анализ данных.")
            return

        deps_list = top_correlations(self.corr_df, k=5) if self.corr_df is not None else []
        trends_list = summarize_trends(self.metrics_df)
        comp = build_comparison(self.df)

        # Prepare report text
        # Create a combined report text
        report_text = []
        report_text.append("АНАЛИТИЧЕСКИЙ ОТЧЕТ")
        report_text.append("=" * 80)
        report_text.append("")
        report_text.append("1) Описание данных")
        report_text.append(f"Файл: {os.path.basename(self.file_path_var.get())}")
        report_text.append(f"Строк (после очистки): {len(self.df)}")
        report_text.append(f"Период: {self.df['date'].iloc[0].date()} — {self.df['date'].iloc[-1].date()}")
        report_text.append(f"Показатели: {', '.join([c for c in self.df.columns if c != 'date'])}")
        report_text.append("")
        report_text.append("2) Расчет показателей")
        report_text.append(self._format_overview(self.df, self.metrics_df, self.shares_df, trends_list, deps_list))
        report_text.append("")
        report_text.append("3) Сравнение периодов")
        report_text.append(self._format_comparison(comp))
        report_text.append("")
        report_text.append("4) Выводы")
        if trends_list:
            report_text.extend(["- " + t for t in trends_list])
        if deps_list:
            report_text.append("")
            report_text.extend(deps_list)

        report_content_text = "\n".join(report_text)

        # Save graphs as PNG and embed them in HTML
        default_name = "analysis_report_PP12_ISP22.html"
        out_path = filedialog.asksaveasfilename(
            title="Сохранить отчет",
            defaultextension=".html",
            initialfile=default_name,
            filetypes=[("HTML files", "*.html"), ("All files", "*.*")],
        )
        if not out_path:
            return

        if not self._generated_pngs or not self._generated_image_order:
            messagebox.showerror(
                "Ошибка",
                "Графики не подготовлены. Сначала выполните загрузку и анализ данных.",
            )
            return

        # Embed images as base64 inside HTML (no extra files on disk).
        imgs_html_parts: List[str] = []
        for i, img_name in enumerate(self._generated_image_order, start=1):
            png_bytes = self._generated_pngs.get(img_name)
            if not png_bytes:
                continue
            b64 = base64.b64encode(png_bytes).decode("ascii")
            imgs_html_parts.append(
                f"<div><h3>График {i}</h3><img src='data:image/png;base64,{b64}' style='max-width: 100%; height: auto;'/></div>"
            )
        imgs_html = "\n".join(imgs_html_parts)

        report_html = (
            "<!doctype html><html lang='ru'><head>"
            "<meta charset='utf-8'/>"
            "<title>Аналитический отчет</title>"
            "<style>body{{font-family:Arial,Helvetica,sans-serif; padding:14px;}} img{{border:1px solid #ddd; margin:8px 0;}}</style>"
            "</head><body>"
            "<h1>Аналитический отчет по PP12_ISP22</h1>"
            "<pre style='white-space: pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;'>{}</pre>"
            "<hr/>"
            "{}"
            "</body></html>"
        ).format(html.escape(report_content_text), imgs_html)

        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(report_html)
            messagebox.showinfo(
                "Готово",
                "Отчет сохранен.\n\n"
                f"HTML: {out_path}\n"
                f"Графики: {images_dir}",
            )
        except Exception as e:
            messagebox.showerror("Ошибка сохранения", str(e))


def main():
    app = AnalyticsApp()
    app.mainloop()


if __name__ == "__main__":
    main()

