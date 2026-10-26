"""
desktop.py — Ultimate Tkinter GUI for DataYab
Desktop interface for local tabular analysis.
Run with: python desktop.py   or via app.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import threading
import queue
from pathlib import Path
import io
from PIL import Image, ImageTk
import base64
import sys
import os

# Prevent joblib multiprocessing issues on Windows
os.environ["JOBLIB_MULTIPROCESSING"] = "0"

# Add parent directory to path to find backend module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.core import DataYabCore

# ================================
# Global styling
# ================================
BG_COLOR = "#f4f6f9"  # Very light gray-blue – softer than #f5f5f5
ACCENT_COLOR = "#10b981"  # Emerald green – high contrast, modern
HEADER_BG = "#1e293b"  # Slate 800 – deeper, more elegant
CARD_BG = "#ffffff"  # Pure white – clean, best readability
FONT_TITLE = ("Segoe UI", 14, "bold")
FONT_NORMAL = ("Segoe UI", 10)
FONT_MONO = ("Consolas", 9)

# Dark mode colors
DARK_BG = "#0f172a"  # Slate 900 – richer dark background
DARK_HEADER = "#1e293b"  # Slate 800 – header matches light mode header
DARK_CARD = "#1e293b"  # Slate 800 – subtle card distinction
DARK_TEXT = "#f1f5f9"  # Slate 100 – soft white, not harsh
DARK_ACCENT = "#38bdf8"  # Sky 400 – bright blue, visible on dark


class DataYabApp:
    """Main Tkinter Application for DataYab"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("DataYab | Dataset workbench")
        self.root.geometry("1180x800")
        self.root.minsize(980, 660)
        self.root.configure(bg=BG_COLOR)

        # Core backend instance
        self.core = DataYabCore()

        # UI state
        self.current_image = None
        self.is_processing = False
        self.dark_mode = False

        # Style
        self.style = ttk.Style()
        self.style.theme_use("clam")
        self.setup_menu()
        self.update_style()

        self.setup_notebook()
        self.setup_status_bar()
        self.toggle_log_panel()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def update_style(self):
        """Apply a consistent accessible theme to ttk and native controls."""
        bg = DARK_BG if self.dark_mode else BG_COLOR
        panel = DARK_CARD if self.dark_mode else CARD_BG
        fg = DARK_TEXT if self.dark_mode else "#172b3a"
        accent = "#13795b" if self.dark_mode else "#08785d"
        self.root.configure(bg=bg)
        self.style.configure(".", font=FONT_NORMAL, background=bg, foreground=fg)
        self.style.configure("TFrame", background=bg)
        self.style.configure("TNotebook", background=bg, borderwidth=0)
        self.style.configure(
            "TNotebook.Tab", background=HEADER_BG, foreground="white", padding=[14, 9]
        )
        self.style.map(
            "TNotebook.Tab",
            background=[("selected", accent)],
            foreground=[("selected", "white")],
        )
        self.style.configure("TLabel", background=bg, foreground=fg)
        self.style.configure("TLabelframe", background=bg, foreground=fg)
        self.style.configure(
            "TLabelframe.Label",
            background=bg,
            foreground=fg,
            font=("Segoe UI", 10, "bold"),
        )
        self.style.configure(
            "TButton", background=HEADER_BG, foreground="white", padding=[10, 7]
        )
        self.style.map(
            "TButton",
            background=[("disabled", "#73818c"), ("active", "#285469")],
            foreground=[("disabled", "#e1e8ec")],
        )
        self.style.configure("Accent.TButton", background=accent, foreground="white")
        self.style.configure("TEntry", fieldbackground=panel, foreground=fg)
        self.style.configure("TCombobox", fieldbackground=panel, foreground=fg)
        self.style.map(
            "TCombobox",
            fieldbackground=[("readonly", panel)],
            foreground=[("readonly", fg)],
        )
        self.style.configure("TCheckbutton", background=bg, foreground=fg)
        self.style.map("TCheckbutton", background=[("active", bg)])
        self.style.configure(
            "Treeview",
            background=panel,
            fieldbackground=panel,
            foreground=fg,
            rowheight=27,
        )
        self.style.configure(
            "Treeview.Heading", background=HEADER_BG, foreground="white", padding=7
        )
        self.style.map(
            "Treeview",
            background=[("selected", accent)],
            foreground=[("selected", "white")],
        )

        def recolor(parent):
            for widget in parent.winfo_children():
                if isinstance(widget, (tk.Text, tk.Entry, tk.Spinbox)):
                    widget.configure(bg=panel, fg=fg, insertbackground=fg)
                elif isinstance(widget, tk.Canvas):
                    widget.configure(bg=bg)
                elif isinstance(widget, tk.Label) and widget.cget("bg") != HEADER_BG:
                    widget.configure(bg=panel, fg=fg)
                recolor(widget)

        recolor(self.root)
        self.update_menu_colors()

    def update_menu_colors(self):
        """Recursively update menu bar colors for dark/light mode."""
        if self.dark_mode:
            bg = DARK_HEADER
            fg = DARK_TEXT
            active_bg = DARK_ACCENT
        else:
            bg = HEADER_BG
            fg = "white"
            active_bg = ACCENT_COLOR
        self.menubar.config(
            bg=bg, fg=fg, activebackground=active_bg, activeforeground="white"
        )
        for menu in self.menubar.winfo_children():
            if isinstance(menu, tk.Menu):
                menu.config(
                    bg=bg, fg=fg, activebackground=active_bg, activeforeground="white"
                )

    # ------------------------------------------------------------
    # UI SETUP
    # ------------------------------------------------------------
    def setup_menu(self):
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        file_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(
            label="Load Dataset", command=lambda: self.notebook.select(0)
        )
        file_menu.add_separator()
        file_menu.add_command(label="Save Pipeline", command=self.save_pipeline)
        file_menu.add_command(label="Load Pipeline", command=self.load_pipeline)
        file_menu.add_separator()
        file_menu.add_command(label="Export HTML Report", command=self.export_report)
        file_menu.add_command(label="Export Cleaned CSV", command=self.export_cleaned)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        view_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Show Log Panel", command=self.toggle_log_panel)
        view_menu.add_command(label="Dark Mode", command=self.toggle_dark_mode)

        help_menu = tk.Menu(self.menubar, tearoff=0)
        self.menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def setup_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        # Create all 7 tabs
        self.tab_load = LoadTab(self.notebook, self)
        self.tab_clean = CleanTab(self.notebook, self)
        self.tab_eda = EDATab(self.notebook, self)
        self.tab_train = TrainTab(self.notebook, self)
        self.tab_evaluate = EvaluateTab(self.notebook, self)
        self.tab_insights = InsightsTab(self.notebook, self)
        self.tab_predict = PredictTab(self.notebook, self)

        self.notebook.add(self.tab_load, text="📁 Load Data")
        self.notebook.add(self.tab_clean, text="🔧 Clean Data")
        self.notebook.add(self.tab_eda, text="📊 EDA")
        self.notebook.add(self.tab_train, text="🤖 Train Model")
        self.notebook.add(self.tab_evaluate, text="📈 Evaluate")
        self.notebook.add(self.tab_insights, text="💡 Insights")
        self.notebook.add(self.tab_predict, text="🔮 Predict")

    def setup_status_bar(self):
        self.status_frame = tk.Frame(self.root, bg=HEADER_BG, height=32)
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.status_label = tk.Label(
            self.status_frame,
            text="✅ Ready",
            bg=HEADER_BG,
            fg="white",
            anchor=tk.W,
            font=("Segoe UI", 9),
        )
        self.status_label.pack(side=tk.LEFT, padx=12, pady=6)

        self.progress_bar = ttk.Progressbar(
            self.status_frame, mode="indeterminate", length=180
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=12, pady=5)

        self.log_panel_visible = False
        self.log_panel_frame = None
        self.log_text = None

    def toggle_log_panel(self):
        if self.log_panel_visible:
            if self.log_panel_frame:
                self.log_panel_frame.pack_forget()
            self.log_panel_visible = False
        else:
            if self.log_panel_frame is None:
                # Create frame with larger height
                self.log_panel_frame = tk.Frame(self.root, height=200, bg="#1e1e1e")
                self.log_panel_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=6)
                self.log_text = scrolledtext.ScrolledText(
                    self.log_panel_frame,
                    height=8,
                    bg="#1e1e1e",
                    fg="#00ff00",
                    font=("Consolas", 8),
                )
                self.log_text.pack(fill=tk.BOTH, expand=True)
                # Add a test message to confirm it's working
                self.log_text.insert(
                    tk.END, "[LOG PANEL READY] – Messages will appear here\n"
                )
            else:
                self.log_panel_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=12, pady=6)
            self.log_panel_visible = True
            # Bring to front (optional)
            self.log_panel_frame.lift()

    def toggle_dark_mode(self):
        self.dark_mode = not self.dark_mode
        self.update_style()
        self.set_status("Dark mode toggled")

    def set_status(self, text, is_error=False):
        self.status_label.config(
            text=f"📌 {text}", fg="#ff6b6b" if is_error else "white"
        )
        self.root.update_idletasks()
        if self.log_panel_visible and self.log_text:
            from datetime import datetime

            timestamp = datetime.now().strftime("%H:%M:%S")
            self.log_text.insert(tk.END, f"[{timestamp}] {text}\n")
            self.log_text.see(tk.END)

    def start_processing(self):
        self.is_processing = True
        self.progress_bar.start(10)
        self.set_status("Processing... ⏳")
        self._disabled_controls = []

        def disable(parent):
            for widget in parent.winfo_children():
                if isinstance(
                    widget,
                    (ttk.Button, ttk.Combobox, ttk.Entry, ttk.Checkbutton, ttk.Scale),
                ):
                    self._disabled_controls.append((widget, str(widget.cget("state"))))
                    widget.configure(state="disabled")
                disable(widget)

        disable(self.root)
        # Disable critical action buttons during processing
        if hasattr(self, "tab_train") and hasattr(self.tab_train, "train_btn"):
            self.tab_train.train_btn.config(state=tk.DISABLED)
        if hasattr(self, "tab_predict") and hasattr(self.tab_predict, "predict_btn"):
            self.tab_predict.predict_btn.config(state=tk.DISABLED)

    def stop_processing(self):
        self.is_processing = False
        self.progress_bar.stop()
        self.set_status("Ready ✓")
        for widget, state in getattr(self, "_disabled_controls", []):
            if widget.winfo_exists():
                widget.configure(state=state)
        # Re-enable critical action buttons
        if hasattr(self, "tab_train") and hasattr(self.tab_train, "train_btn"):
            self.tab_train.train_btn.config(state=tk.NORMAL)
        if hasattr(self, "tab_predict") and hasattr(self.tab_predict, "predict_btn"):
            self.tab_predict.predict_btn.config(state=tk.NORMAL)

    def run_threaded(self, target_func, on_complete=None):
        if self.is_processing:
            self.set_status("Please wait for the current operation.")
            return
        results = queue.Queue()

        def worker():
            try:
                result = target_func()
                if isinstance(result, dict) and result.get("success") is False:
                    raise ValueError(result.get("message", "Operation failed."))
                results.put((result, None))
            except Exception as error:
                results.put((None, str(error)))

        def poll():
            try:
                result, error = results.get_nowait()
            except queue.Empty:
                self.root.after(50, poll)
                return
            self.stop_processing()
            if error:
                messagebox.showerror("Operation failed", error)
            elif on_complete:
                try:
                    on_complete(result)
                except Exception as error:
                    messagebox.showerror("Display error", str(error))

        self.start_processing()
        threading.Thread(target=worker, daemon=True).start()
        self.root.after(50, poll)

    def display_image(self, base64_str, target_label):
        try:
            img_data = base64.b64decode(base64_str)
            img = Image.open(io.BytesIO(img_data))
            img.thumbnail((700, 450), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            target_label.config(image=photo)
            target_label.image = photo
            return True
        except Exception as e:
            self.set_status(f"Image error: {e}", is_error=True)
            return False

    def show_about(self):
        messagebox.showinfo(
            "About DataYab",
            "DataYab — Ultimate Dataset Analyzer\n\n"
            "Complete pipeline for any dataset:\n"
            "• Load any CSV/Excel/JSON/Parquet\n"
            "• Auto-clean & preprocess\n"
            "• Full EDA with charts\n"
            "• Train classification/regression models\n"
            "• Automatic insights generation\n"
            "• Hyperparameter tuning, SMOTE, feature selection\n"
            "• Batch predictions\n\n"
            "Local tools for exploratory analysis",
        )

    def save_pipeline(self):
        if self.core.model is None:
            messagebox.showwarning("No Model", "Train a model first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".pkl", filetypes=[("Pickle files", "*.pkl")]
        )
        if path:
            if Path(path).exists() and not messagebox.askyesno(
                "Overwrite", f"File exists. Overwrite {Path(path).name}?"
            ):
                return
            self.run_threaded(
                lambda: self.core.save_pipeline(path),
                lambda r: (
                    messagebox.showinfo("Success", f"Pipeline saved to {path}")
                    if r.get("success")
                    else messagebox.showerror(
                        "Error", r.get("message", "Unknown error")
                    )
                ),
            )

    def load_pipeline(self):
        if self.is_processing:
            return
        path = filedialog.askopenfilename(
            filetypes=[("Model packages", "*.pkl *.joblib")]
        )
        if not path:
            return
        if not messagebox.askyesno(
            "Trusted model required",
            "Model files can execute code. Only open a model you created or trust. Continue?",
        ):
            return

        def loaded(result):
            self.tab_predict.build_form()
            self.tab_evaluate.refresh_model_combo()
            messagebox.showinfo("Model loaded", self.core.model_name)

        self.run_threaded(lambda: self.core.load_pipeline(path), loaded)

    def export_report(self):
        if self.core.df is None:
            messagebox.showwarning("No Data", "Load a dataset first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".html", filetypes=[("HTML files", "*.html")]
        )
        if path:
            if Path(path).exists() and not messagebox.askyesno(
                "Overwrite", f"File exists. Overwrite {Path(path).name}?"
            ):
                return
            self.run_threaded(
                lambda: self.core.export_report(path),
                lambda r: (
                    messagebox.showinfo("Success", f"Report saved to {path}")
                    if r.get("success")
                    else messagebox.showerror(
                        "Error", r.get("message", "Unknown error")
                    )
                ),
            )

    def export_cleaned(self):
        if self.core.df is None:
            messagebox.showwarning("No Data", "Load a dataset first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
        )
        if path:
            if Path(path).exists() and not messagebox.askyesno(
                "Overwrite", f"File exists. Overwrite {Path(path).name}?"
            ):
                return
            self.run_threaded(
                lambda: self.core.export_cleaned_csv(path),
                lambda r: (
                    messagebox.showinfo("Success", f"Cleaned data saved to {path}")
                    if r.get("success")
                    else messagebox.showerror(
                        "Error", r.get("message", "Unknown error")
                    )
                ),
            )

    def on_closing(self):
        if messagebox.askokcancel("Exit", "Do you want to quit?"):
            self.root.destroy()

    def run(self):
        self.root.mainloop()


# ======================= TAB 1: LOAD DATA =======================
class LoadTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.create_widgets()

    def create_widgets(self):
        main = ttk.Frame(self, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(main, text="Load Dataset", padding=15)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))

        ttk.Button(left, text="📂 Load CSV", command=self.load_csv, width=25).pack(
            fill=tk.X, pady=5
        )
        ttk.Button(left, text="📊 Load Excel", command=self.load_excel, width=25).pack(
            fill=tk.X, pady=5
        )
        ttk.Button(left, text="📄 Load JSON", command=self.load_json, width=25).pack(
            fill=tk.X, pady=5
        )
        ttk.Button(
            left, text="🗄️ Load Parquet", command=self.load_parquet, width=25
        ).pack(fill=tk.X, pady=5)

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=15)

        ttk.Label(left, text="Dataset Info:", font=("Segoe UI", 10, "bold")).pack(
            anchor=tk.W
        )
        self.info_text = tk.Text(
            left,
            height=16,
            width=42,
            font=FONT_MONO,
            wrap=tk.WORD,
            bg=CARD_BG,
            relief=tk.FLAT,
            borderwidth=0,
        )
        self.info_text.pack(fill=tk.BOTH, expand=True, pady=8)

        ttk.Button(
            left, text="🔄 Reset to Original", command=self.reset_data, width=25
        ).pack(fill=tk.X, pady=5)

        right = ttk.LabelFrame(main, text="Data Preview", padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        control_frame = ttk.Frame(right)
        control_frame.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(control_frame, text="Rows:").pack(side=tk.LEFT, padx=5)
        self.rows_spin = tk.Spinbox(
            control_frame, from_=1, to=100, width=8, font=FONT_NORMAL
        )
        self.rows_spin.delete(0, tk.END)
        self.rows_spin.insert(0, "10")
        self.rows_spin.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Refresh", command=self.refresh_preview).pack(
            side=tk.LEFT, padx=10
        )

        tree_frame = ttk.Frame(right)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        v_scroll = ttk.Scrollbar(tree_frame)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        h_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL)
        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree = ttk.Treeview(
            tree_frame, yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set
        )
        self.tree.pack(fill=tk.BOTH, expand=True)
        v_scroll.config(command=self.tree.yview)
        h_scroll.config(command=self.tree.xview)

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if path:
            self.app.run_threaded(lambda: self.app.core.load_csv(path), self.on_load)

    def load_excel(self):
        path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx *.xls")])
        if path:
            self.app.run_threaded(lambda: self.app.core.load_excel(path), self.on_load)

    def load_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if path:
            self.app.run_threaded(lambda: self.app.core.load_json(path), self.on_load)

    def load_parquet(self):
        path = filedialog.askopenfilename(filetypes=[("Parquet files", "*.parquet")])
        if path:
            self.app.run_threaded(
                lambda: self.app.core.load_parquet(path), self.on_load
            )

    def on_load(self, result):
        if result["success"]:
            self.update_info()
            self.refresh_preview()
            self.app.tab_train.update_target_list()
            self.app.tab_eda.update_column_lists()
            self.app.tab_predict.build_form()
            self.app.tab_clean.refresh_report()
            messagebox.showinfo("Success", result["message"])
        else:
            messagebox.showerror("Error", result["message"])

    def update_info(self):
        self.info_text.delete(1.0, tk.END)
        info = self.app.core.get_info()
        self.info_text.insert(
            tk.END,
            f"Rows: {info['rows']:,}\n"
            f"Columns: {info['cols']}\n"
            f"Memory: {info['memory_mb']} MB\n\n"
            f"📊 Feature Types:\n"
            f"  Numeric: {len(info['numeric_cols'])}\n"
            f"  Categorical: {len(info['categorical_cols'])}\n\n"
            f"🔍 Data Quality:\n"
            f"  Missing: {info['total_missing']:,}\n"
            f"  Duplicates: {info['duplicates']:,}\n\n"
            f"📋 Columns:\n{', '.join(info['columns'][:15])}",
        )

        # Show feature summary if target is set
        if self.app.core.problem_type:
            self.info_text.insert(
                tk.END, f"\n\n✅ Problem Type: {self.app.core.problem_type.upper()}"
            )
        if self.app.core.target_col:
            self.info_text.insert(tk.END, f"\n🎯 Target: {self.app.core.target_col}")

    def refresh_preview(self):
        if self.app.core.df is None:
            return
        try:
            n = int(self.rows_spin.get())
        except ValueError:
            n = 10
        df = self.app.core.df.head(n)
        self.tree.delete(*self.tree.get_children())
        self.tree["columns"] = list(df.columns)
        self.tree["show"] = "headings"
        for col in df.columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=100, minwidth=50)
        for _, row in df.iterrows():
            self.tree.insert("", tk.END, values=list(row))

    def reset_data(self):
        self.app.core.reset_to_original()
        self.update_info()
        self.refresh_preview()
        self.app.set_status("Reset to original data")


# ======================= TAB 2: CLEAN DATA =======================
class CleanTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.create_widgets()

    def create_widgets(self):
        main = ttk.Frame(self, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(main, text="Cleaning Operations", padding=15)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))

        ttk.Label(left, text="Missing Values:", font=("Segoe UI", 10, "bold")).pack(
            anchor=tk.W
        )
        self.missing_strategy = ttk.Combobox(
            left,
            values=[
                "drop_rows",
                "mean",
                "median",
                "mode",
                "ffill",
                "bfill",
                "zero",
                "unknown",
                "knn",
                "iterative",
            ],
            state="readonly",
            width=25,
        )
        self.missing_strategy.set("median")
        self.missing_strategy.pack(fill=tk.X, pady=5)
        ttk.Button(
            left, text="Apply Missing Handling", command=self.handle_missing, width=25
        ).pack(fill=tk.X, pady=5)

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=10)

        ttk.Button(
            left, text="🗑️ Remove Duplicates", command=self.remove_duplicates, width=25
        ).pack(fill=tk.X, pady=5)
        ttk.Button(
            left, text="🏷️ Encode Categorical (Label)", command=self.encode_cat, width=25
        ).pack(fill=tk.X, pady=5)
        ttk.Button(
            left,
            text="🗑️ Remove Constant Columns",
            command=self.remove_constant,
            width=25,
        ).pack(fill=tk.X, pady=5)
        ttk.Button(
            left, text="🔄 Fix Data Types", command=self.fix_dtypes, width=25
        ).pack(fill=tk.X, pady=5)
        ttk.Button(
            left,
            text="⚠️ Remove Outliers (IsoForest)",
            command=self.remove_outliers_iso,
            width=25,
        ).pack(fill=tk.X, pady=5)

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=10)

        ttk.Button(
            left, text="✨ AUTO CLEAN (One Click)", command=self.auto_clean, width=25
        ).pack(fill=tk.X, pady=10)

        right = ttk.LabelFrame(main, text="Missing Values Report", padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tree_frame = ttk.Frame(right)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        v_scroll = ttk.Scrollbar(tree_frame)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.report_tree = ttk.Treeview(
            tree_frame,
            yscrollcommand=v_scroll.set,
            columns=("Column", "Missing", "Percent"),
            show="headings",
            height=18,
        )
        self.report_tree.heading("Column", text="Column")
        self.report_tree.heading("Missing", text="Missing Count")
        self.report_tree.heading("Percent", text="Percent (%)")
        self.report_tree.column("Column", width=220)
        self.report_tree.column("Missing", width=130)
        self.report_tree.column("Percent", width=110)
        self.report_tree.pack(fill=tk.BOTH, expand=True)
        v_scroll.config(command=self.report_tree.yview)

        ttk.Button(right, text="Refresh Report", command=self.refresh_report).pack(
            pady=8
        )
        self.refresh_report()

    def refresh_report(self):
        if self.app.core.df is None:
            return
        self.report_tree.delete(*self.report_tree.get_children())
        report = self.app.core.get_missing_report()
        for _, row in report.iterrows():
            pct = (
                row["Percent"]
                if "Percent" in row
                else row["Missing"] / len(self.app.core.df) * 100
            )
            self.report_tree.insert(
                "", tk.END, values=(row["Column"], row["Missing"], f"{pct:.1f}%")
            )

    def handle_missing(self):
        self.app.run_threaded(
            lambda strategy=self.missing_strategy.get(): self.app.core.handle_missing(
                strategy
            ),
            lambda r: (
                self.refresh_report(),
                self.app.set_status(
                    f"Missing values handled: {r.get('message', 'done')}"
                ),
            ),
        )

    def remove_duplicates(self):
        self.app.run_threaded(
            lambda: self.app.core.remove_duplicates(),
            lambda r: (
                self.refresh_report(),
                self.app.set_status(f"Duplicates removed: {r.get('removed', 0)} rows"),
            ),
        )

    def encode_cat(self):
        self.app.run_threaded(
            lambda: self.app.core.encode_categorical(),
            lambda r: (
                self.refresh_report(),
                self.app.set_status(
                    f"Categorical columns encoded: {r.get('encoded_columns', [])}"
                ),
            ),
        )

    def remove_constant(self):
        self.app.run_threaded(
            lambda: self.app.core.remove_constant_columns(),
            lambda r: (
                self.refresh_report(),
                self.app.set_status(
                    f"Constant columns removed: {r.get('removed', [])}"
                ),
            ),
        )

    def fix_dtypes(self):
        self.app.run_threaded(
            lambda: self.app.core.fix_dtypes(),
            lambda r: (
                self.refresh_report(),
                self.app.set_status(f"Data types fixed: {r.get('converted', [])}"),
            ),
        )

    def remove_outliers_iso(self):
        self.app.run_threaded(
            lambda: self.app.core.remove_outliers_isolation_forest(contamination=0.1),
            lambda r: (
                self.refresh_report(),
                self.app.set_status(
                    f"Outliers removed: {r.get('removed', 0)} rows using Isolation Forest"
                ),
            ),
        )

    def auto_clean(self):
        target = (
            self.app.tab_train.target_combo.get()
            if self.app.tab_train.target_combo.get()
            else None
        )
        self.app.run_threaded(
            lambda: self.app.core.auto_clean(target_col=target),
            lambda r: (
                self.refresh_report(),
                messagebox.showinfo("Auto Clean", "\n".join(r["steps"])),
                self.app.set_status("Auto‑clean completed"),
            ),
        )


# ======================= TAB 3: EDA =======================
class EDATab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.create_widgets()

    def create_widgets(self):
        main = ttk.Frame(self, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(main, text="Chart Controls", padding=15)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))

        self.chart_type = ttk.Combobox(
            left,
            values=[
                "Histogram",
                "Box Plot",
                "Correlation Heatmap",
                "Target Distribution",
                "Scatter Plot",
                "Missing Values Map",
                "Observations",
            ],
            state="readonly",
            width=28,
        )
        self.chart_type.set("Histogram")
        self.chart_type.pack(fill=tk.X, pady=5)

        ttk.Label(left, text="Column:").pack(anchor=tk.W, pady=(10, 2))
        self.col_combo = ttk.Combobox(left, width=28)
        self.col_combo.pack(fill=tk.X, pady=2)

        ttk.Label(left, text="Second Column (scatter):").pack(anchor=tk.W, pady=(10, 2))
        self.col2_combo = ttk.Combobox(left, width=28)
        self.col2_combo.pack(fill=tk.X, pady=2)

        ttk.Button(
            left, text="📈 Generate Chart", command=self.generate_chart, width=28
        ).pack(fill=tk.X, pady=15)
        ttk.Button(
            left, text="💾 Save Chart as PNG", command=self.save_chart, width=28
        ).pack(fill=tk.X, pady=5)

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=10)

        ttk.Label(left, text="📋 Observations", font=("Segoe UI", 10, "bold")).pack(
            anchor=tk.W
        )
        self.obs_text = scrolledtext.ScrolledText(
            left, height=12, width=45, wrap=tk.WORD, font=FONT_MONO
        )
        self.obs_text.pack(fill=tk.BOTH, expand=True, pady=5)
        ttk.Button(
            left, text="Generate Observations", command=self.generate_obs, width=28
        ).pack(fill=tk.X)

        right = ttk.LabelFrame(main, text="Chart", padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.chart_label = tk.Label(right, bg="white", relief=tk.SUNKEN, bd=1)
        self.chart_label.pack(fill=tk.BOTH, expand=True)

        self.current_chart_base64 = None
        self.update_column_lists()

    def update_column_lists(self):
        if self.app.core.df is not None:
            cols = list(self.app.core.df.columns)
            self.col_combo["values"] = cols
            self.col2_combo["values"] = cols

    def save_chart(self):
        if self.current_chart_base64:
            path = filedialog.asksaveasfilename(
                defaultextension=".png", filetypes=[("PNG files", "*.png")]
            )
            if path:
                img_data = base64.b64decode(self.current_chart_base64)
                with open(path, "wb") as f:
                    f.write(img_data)
                self.app.set_status(f"Chart saved to {path}")
        else:
            messagebox.showwarning("No Chart", "Generate a chart first.")

    def generate_chart(self):
        if self.app.core.df is None:
            messagebox.showwarning("No Data", "Load a dataset first.")
            return

        chart = self.chart_type.get()
        col = self.col_combo.get()
        col2 = self.col2_combo.get()

        if chart == "Histogram" and col:
            self.app.run_threaded(
                lambda: self.app.core.plot_histogram(col),
                lambda img: self.display_chart(img),
            )
        elif chart == "Box Plot" and col:
            self.app.run_threaded(
                lambda: self.app.core.plot_boxplot(col),
                lambda img: self.display_chart(img),
            )
        elif chart == "Correlation Heatmap":
            self.app.run_threaded(
                self.app.core.plot_correlation_heatmap,
                lambda img: self.display_chart(img),
            )
        elif chart == "Target Distribution":
            target_col = col if col else self.app.core.target_col
            if not target_col:
                messagebox.showwarning(
                    "No Target", "Select a target column or train a model first."
                )
                return
            self.app.run_threaded(
                lambda: self.app.core.plot_target_distribution(target_col),
                lambda img: self.display_chart(img),
            )
        elif chart == "Scatter Plot" and col and col2:
            self.app.run_threaded(
                lambda: self.app.core.plot_scatter(col, col2),
                lambda img: self.display_chart(img),
            )
        elif chart == "Missing Values Map":
            self.app.run_threaded(
                self.app.core.plot_missing_heatmap, lambda img: self.display_chart(img)
            )
        elif chart == "Observations":
            self.generate_obs()

    def display_chart(self, img_base64):
        self.current_chart_base64 = img_base64
        self.app.display_image(img_base64, self.chart_label)

    def generate_obs(self):
        if self.app.core.df is None:
            messagebox.showwarning("No Data", "Load a dataset first.")
            return
        self.app.run_threaded(
            lambda: self.app.core.generate_observations(15), self.display_observations
        )

    def display_observations(self, obs):
        self.obs_text.delete(1.0, tk.END)
        for o in obs:
            self.obs_text.insert(tk.END, f"• {o}\n")


# ======================= TAB 4: TRAIN MODEL (Enhanced) =======================
class TrainTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.create_widgets()

    def create_widgets(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        # ----- LEFT PANEL (fixed width, scrollable) -----
        left_container = ttk.LabelFrame(main, text="Training Configuration", padding=10)
        left_container.pack(
            side=tk.LEFT, fill=tk.Y, padx=(0, 10)
        )  # fill only Y, not BOTH
        left_container.pack_propagate(False)
        left_container.config(width=420)

        # Canvas + Scrollbar
        left_canvas = tk.Canvas(left_container, highlightthickness=0, borderwidth=0)
        left_scrollbar = ttk.Scrollbar(
            left_container, orient="vertical", command=left_canvas.yview
        )
        left_scrollable_frame = ttk.Frame(left_canvas)

        left_scrollable_frame.bind(
            "<Configure>",
            lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")),
        )
        left_window = left_canvas.create_window(
            (0, 0), window=left_scrollable_frame, anchor="nw"
        )
        left_canvas.bind(
            "<Configure>",
            lambda e: left_canvas.itemconfigure(left_window, width=e.width),
        )
        left_canvas.bind(
            "<MouseWheel>",
            lambda e: left_canvas.yview_scroll(int(-e.delta / 120), "units"),
        )
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=0, pady=0)
        left_scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=0, pady=0)

        # All controls go inside `left` frame
        left = left_scrollable_frame

        # ---- Reduced vertical spacing ----
        # Target column
        ttk.Label(left, text="Target Column:", font=("Segoe UI", 10, "bold")).pack(
            anchor=tk.W, pady=(5, 0)
        )
        self.target_combo = ttk.Combobox(left, width=40)
        self.target_combo.pack(fill=tk.X, pady=(2, 5))

        # Test size
        ttk.Label(left, text="Test Size:").pack(anchor=tk.W, pady=(5, 0))
        self.test_size = ttk.Scale(
            left, from_=0.1, to=0.4, orient=tk.HORIZONTAL, value=0.2
        )
        self.test_size.pack(fill=tk.X, pady=2)
        self.test_label = ttk.Label(left, text="20%")
        self.test_label.pack(anchor=tk.W, pady=(0, 5))

        # Model selection
        ttk.Label(left, text="Model:").pack(anchor=tk.W, pady=(5, 0))
        self.model_combo = ttk.Combobox(
            left,
            values=[
                "Random Forest",
                "Logistic Regression",
                "Decision Tree",
                "SVM",
                "KNN",
                "Gradient Boosting",
                "Naive Bayes",
            ],
            state="readonly",
            width=40,
        )
        self.model_combo.set("Random Forest")
        self.model_combo.pack(fill=tk.X, pady=2)
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_change)

        # Hyperparameters
        self.hyper_frame = ttk.LabelFrame(left, text="Hyperparameters", padding=5)
        self.hyper_frame.pack(fill=tk.X, pady=5)
        self.hyper_widgets = {}
        self.on_model_change()

        # Advanced options
        self.adv_frame = ttk.LabelFrame(left, text="Advanced Options", padding=5)
        self.adv_frame.pack(fill=tk.X, pady=5)

        self.smote_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            self.adv_frame,
            text="Use SMOTE (oversample minority class)",
            variable=self.smote_var,
        ).pack(anchor=tk.W, pady=2)

        ttk.Label(self.adv_frame, text="Class Weight:").pack(anchor=tk.W)
        self.class_weight_combo = ttk.Combobox(
            self.adv_frame, values=["None", "balanced"], state="readonly", width=30
        )
        self.class_weight_combo.set("None")
        self.class_weight_combo.pack(anchor=tk.W, pady=2)

        # ---------- FEATURE SELECTION SECTION (ADD THIS) ----------
        self.fs_frame = ttk.LabelFrame(left, text="🎯 Feature Selection", padding=5)
        self.fs_frame.pack(fill=tk.X, pady=8)

        # Row 1: RF Importance + Mutual Info
        btn_frame1 = ttk.Frame(self.fs_frame)
        btn_frame1.pack(fill=tk.X, pady=2)
        ttk.Button(
            btn_frame1,
            text="🌲 RF Importance",
            command=self.show_rf_importance,
            width=16,
        ).pack(side=tk.LEFT, padx=2)
        ttk.Button(
            btn_frame1, text="📊 Mutual Info", command=self.show_mutual_info, width=16
        ).pack(side=tk.LEFT, padx=2)

        # Row 2: Manual selection
        manual_frame = ttk.Frame(self.fs_frame)
        manual_frame.pack(fill=tk.X, pady=4)
        ttk.Label(manual_frame, text="Manual:", width=7).pack(side=tk.LEFT)
        self.manual_features_entry = ttk.Entry(manual_frame, width=35)
        ttk.Button(
            manual_frame, text="Apply", command=self.apply_manual_selection, width=8
        ).pack(side=tk.RIGHT)
        self.manual_features_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Row 3: PCA
        pca_frame = ttk.Frame(self.fs_frame)
        pca_frame.pack(fill=tk.X, pady=4)
        ttk.Label(pca_frame, text="PCA:", width=10).pack(side=tk.LEFT)
        self.pca_components = tk.Spinbox(pca_frame, from_=2, to=20, width=8)
        self.pca_components.delete(0, tk.END)
        self.pca_components.insert(0, "2")
        self.pca_components.pack(side=tk.LEFT, padx=5)
        ttk.Button(pca_frame, text="Apply PCA", command=self.apply_pca, width=10).pack(
            side=tk.LEFT, padx=2
        )
        ttk.Button(
            pca_frame, text="Restore", command=self.restore_from_pca, width=10
        ).pack(side=tk.LEFT)

        # Feature status label
        self.feature_status_label = ttk.Label(
            self.fs_frame, text="", font=("Segoe UI", 9, "italic")
        )
        self.feature_status_label.pack(pady=2)

        # Buttons
        self.train_btn = ttk.Button(
            left, text="🚀 Split & Train", command=self.train_model, width=40
        )
        self.train_btn.pack(fill=tk.X, pady=8)
        ttk.Button(
            left, text="🏆 Compare All Models", command=self.compare_all, width=40
        ).pack(fill=tk.X, pady=5)
        ttk.Button(
            left,
            text="🔗 Train Stacked Model",
            command=self.show_stacking_dialog,
            width=40,
        ).pack(fill=tk.X, pady=5)

        ttk.Separator(left, orient="horizontal").pack(fill=tk.X, pady=8)

        # Feature Info
        ttk.Label(left, text="Feature Info:", font=("Segoe UI", 9, "bold")).pack(
            anchor=tk.W, pady=(5, 0)
        )
        self.feature_info = tk.Text(
            left, height=8, width=40, font=FONT_MONO, wrap=tk.WORD, bg=CARD_BG
        )
        self.feature_info.pack(fill=tk.BOTH, expand=True, pady=5)

        # ----- RIGHT PANEL (takes all remaining space) -----
        right = ttk.LabelFrame(main, text="Training Results", padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.results_text = scrolledtext.ScrolledText(
            right, font=FONT_MONO, wrap=tk.WORD
        )
        self.results_text.pack(fill=tk.BOTH, expand=True)

        self.update_target_list()

    def on_model_change(self, event=None):
        for frame, _, _ in self.hyper_widgets.values():
            frame.destroy()
        self.hyper_widgets.clear()
        model = self.model_combo.get()
        if model == "Random Forest":
            self.add_hyper_param("n_estimators", "int", 100, 10, 500)
            self.add_hyper_param("max_depth", "int", 10, 1, 50)
        elif model == "Gradient Boosting":
            self.add_hyper_param("n_estimators", "int", 100, 10, 500)
            self.add_hyper_param("max_depth", "int", 5, 1, 20)
            self.add_hyper_param("learning_rate", "float", 0.1, 0.01, 1.0)
        elif model == "Logistic Regression":
            self.add_hyper_param("C", "float", 1.0, 0.01, 10.0)
        elif model == "SVM":
            self.add_hyper_param("C", "float", 1.0, 0.01, 10.0)
        elif model == "KNN":
            self.add_hyper_param("n_neighbors", "int", 5, 1, 20)

    def add_hyper_param(self, name, typ, default, minv, maxv):
        frame = ttk.Frame(self.hyper_frame)
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=f"{name}:").pack(side=tk.LEFT, padx=5)
        if typ == "int":
            var = tk.IntVar(value=default)
            spin = tk.Spinbox(frame, from_=minv, to=maxv, textvariable=var, width=10)
            spin.pack(side=tk.LEFT, padx=5)
            self.hyper_widgets[name] = (frame, var, typ)  # store frame, var, typ
        else:
            var = tk.DoubleVar(value=default)
            scale = ttk.Scale(
                frame,
                from_=minv,
                to=maxv,
                orient=tk.HORIZONTAL,
                variable=var,
                length=150,
            )
            scale.pack(side=tk.LEFT, padx=5)
            label = ttk.Label(frame, text=f"{default}")
            label.pack(side=tk.LEFT)

            def update_label(var_obj, lbl=label):
                lbl.config(text=f"{float(var_obj.get()):.2f}")

            var.trace(
                "w", lambda *a, var_obj=var, lbl=label: update_label(var_obj, lbl)
            )
            self.hyper_widgets[name] = (frame, var, typ)

    def get_hyperparams(self):
        params = {}
        for name, (_, var, _) in self.hyper_widgets.items():
            params[name] = var.get()
        return params

    def update_target_list(self):
        if self.app.core.df is not None:
            self.target_combo["values"] = list(self.app.core.df.columns)

    def train_model(self):
        if self.app.core.df is None:
            messagebox.showwarning("No Data", "Load a dataset first.")
            return
        target = self.target_combo.get()
        if not target:
            messagebox.showwarning("No Target", "Select a target column.")
            return
        self.app.core.target_col = target

        # Show model recommendation
        rec = self.app.core.get_model_recommendation(target=target)
        rec_msg = f"Detected: {rec['problem_type'].upper()}\n"
        if rec["recommended_strategies"]:
            rec_msg += "Strategies:\n" + "\n".join(
                f"  • {s}" for s in rec["recommended_strategies"][:3]
            )
        messagebox.showinfo("Recommendation", rec_msg)

        ts = float(self.test_size.get())
        use_smote = self.smote_var.get()
        class_weight = (
            self.class_weight_combo.get()
            if self.class_weight_combo.get() != "None"
            else None
        )
        hyper = self.get_hyperparams()

        model_name = self.model_combo.get()

        def train():
            split = self.app.core.split_data(target, test_size=ts)
            if not split.get("success", False):
                return {
                    "success": False,
                    "message": split.get("message", "Split failed"),
                }
            train_res = self.app.core.train(
                model_name=model_name,
                use_smote=use_smote,
                class_weight=class_weight,
                **hyper,
            )
            metrics = self.app.core.get_metrics()
            return {
                "success": True,
                "split": split,
                "train": train_res,
                "metrics": metrics,
            }

        self.app.run_threaded(train, self.display_training_results)

    def display_training_results(self, results):
        self.results_text.delete(1.0, tk.END)
        if not results.get("success", False):
            self.results_text.insert(
                tk.END, f"❌ Error: {results.get('message', 'Unknown error')}\n"
            )
            return
        if "split" in results:
            s = results["split"]
            self.results_text.insert(tk.END, f"📊 SPLIT RESULTS\n{'=' * 40}\n")
            self.results_text.insert(tk.END, f"Problem Type: {s['problem_type']}\n")
            self.results_text.insert(tk.END, f"Train size: {s['train_size']}\n")
            self.results_text.insert(tk.END, f"Test size: {s['test_size']}\n")
            self.results_text.insert(tk.END, f"Features: {s['features']}\n\n")
        if "metrics" in results:
            self.results_text.insert(tk.END, f"📈 METRICS\n{'=' * 40}\n")
            for k, v in results["metrics"].items():
                if not isinstance(v, str):
                    self.results_text.insert(tk.END, f"{k}: {v:.4f}\n")
        self.feature_info.delete(1.0, tk.END)
        if self.app.core.feature_names:
            self.feature_info.insert(
                tk.END, f"Features ({len(self.app.core.feature_names)}):\n"
            )
            for f in self.app.core.feature_names[:15]:
                self.feature_info.insert(tk.END, f"• {f}\n")
        self.app.tab_predict.build_form()
        self.app.tab_evaluate.update_metrics_display()

    def save_comparison_csv(self, df):
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
        )
        if not path:
            return
        try:
            df.to_csv(path, index=False)
            messagebox.showinfo("Saved", f"Comparison results saved to {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV: {e}")

    def compare_all(self):
        if self.app.core.df is None:
            messagebox.showwarning("No Data", "Load a dataset first.")
            return
        target = self.target_combo.get()
        if not target:
            messagebox.showwarning("No Target", "Select a target column.")
            return
        ts = float(self.test_size.get())
        use_smote = self.smote_var.get()
        class_weight = (
            self.class_weight_combo.get()
            if self.class_weight_combo.get() != "None"
            else None
        )
        hyper = self.get_hyperparams()

        def compare():
            self.app.core.split_data(target, test_size=ts)
            return self.app.core.train_all_models(
                use_smote=use_smote, class_weight=class_weight, **hyper
            )

        self.app.run_threaded(compare, self.display_comparison)

    def update_feature_status(self):
        """Update the feature status label"""
        if self.app.core._selected_feature_names:
            total = len(self.app.core.feature_names)
            selected = len(self.app.core._selected_feature_names)
            self.feature_status_label.config(
                text=f"📌 Using {selected}/{total} features"
            )
        else:
            self.feature_status_label.config(text="")

    def show_rf_importance(self):
        """Show Random Forest feature importance dialog"""
        if self.app.core.X_train is None:
            messagebox.showwarning("No Data", "Split data first (click Split & Train).")
            return

        def get_importance():
            return self.app.core.get_fast_feature_importance(top_n=20)

        def display(importance_series):
            win = tk.Toplevel(self)
            win.title("Random Forest Feature Importance")
            win.geometry("650x500")

            # Main frame
            main = ttk.Frame(win, padding=10)
            main.pack(fill=tk.BOTH, expand=True)

            # Text area
            text = scrolledtext.ScrolledText(
                main, font=("Consolas", 10), wrap=tk.WORD, height=20
            )
            text.pack(fill=tk.BOTH, expand=True)

            text.insert(tk.END, "🌲 RANDOM FOREST FEATURE IMPORTANCE\n")
            text.insert(tk.END, "=" * 50 + "\n\n")

            for feat, imp in importance_series.items():
                bar_len = int(imp * 50)
                bar = "█" * bar_len + "░" * (50 - bar_len)
                text.insert(tk.END, f"{feat:35s} {imp:.4f} {bar}\n")

            # Button frame
            btn_frame = ttk.Frame(main)
            btn_frame.pack(fill=tk.X, pady=10)

            def select_top():
                k = simpledialog.askinteger(
                    "Select Features",
                    "Number of top features to keep:",
                    initialvalue=10,
                    minvalue=1,
                    maxvalue=len(importance_series),
                )
                if k:
                    top_features = list(importance_series.head(k).index)
                    self.app.core.select_features_manual(top_features)
                    self.update_feature_status()
                    messagebox.showinfo(
                        "Success",
                        f"Selected {k} features:\n{', '.join(top_features[:5])}",
                    )
                    win.destroy()

            ttk.Button(btn_frame, text="✓ Keep Top Features", command=select_top).pack(
                side=tk.LEFT, padx=5
            )
            ttk.Button(btn_frame, text="Close", command=win.destroy).pack(
                side=tk.LEFT, padx=5
            )

        self.app.run_threaded(get_importance, display)

    def show_mutual_info(self):
        """Show Mutual Information scores dialog"""
        if self.app.core.X_train is None:
            messagebox.showwarning("No Data", "Split data first (click Split & Train).")
            return

        def get_mi():
            return self.app.core.get_mutual_information_top(k=20)

        def display(mi_series):
            win = tk.Toplevel(self)
            win.title("Mutual Information Scores")
            win.geometry("650x500")

            main = ttk.Frame(win, padding=10)
            main.pack(fill=tk.BOTH, expand=True)

            text = scrolledtext.ScrolledText(
                main, font=("Consolas", 10), wrap=tk.WORD, height=20
            )
            text.pack(fill=tk.BOTH, expand=True)

            text.insert(tk.END, "📊 MUTUAL INFORMATION SCORES\n")
            text.insert(tk.END, "=" * 50 + "\n")
            text.insert(tk.END, "(Higher = stronger relationship with target)\n\n")

            for feat, score in mi_series.items():
                bar_len = (
                    int(score / mi_series.max() * 50) if mi_series.max() > 0 else 0
                )
                bar = "█" * bar_len + "░" * (50 - bar_len)
                text.insert(tk.END, f"{feat:35s} {score:.6f} {bar}\n")

            btn_frame = ttk.Frame(main)
            btn_frame.pack(fill=tk.X, pady=10)

            def select_top():
                k = simpledialog.askinteger(
                    "Select Features",
                    "Number of top features to keep:",
                    initialvalue=10,
                    minvalue=1,
                    maxvalue=len(mi_series),
                )
                if k:
                    top_features = list(mi_series.head(k).index)
                    self.app.core.select_features_manual(top_features)
                    self.update_feature_status()
                    messagebox.showinfo(
                        "Success",
                        f"Selected {k} features:\n{', '.join(top_features[:5])}",
                    )
                    win.destroy()

            ttk.Button(btn_frame, text="✓ Keep Top Features", command=select_top).pack(
                side=tk.LEFT, padx=5
            )
            ttk.Button(btn_frame, text="Close", command=win.destroy).pack(
                side=tk.LEFT, padx=5
            )

        self.app.run_threaded(get_mi, display)

    def apply_manual_selection(self):
        """Apply manual feature selection from entry field"""
        text = self.manual_features_entry.get().strip()
        if not text:
            messagebox.showwarning("Empty", "Enter feature names separated by commas")
            return

        features = [f.strip() for f in text.split(",") if f.strip()]

        # Validate features exist
        missing = [f for f in features if f not in self.app.core.feature_names]
        if missing:
            messagebox.showerror("Not Found", f"Features not found: {missing}")
            return

        self.app.core.select_features_manual(features)
        self.update_feature_status()
        messagebox.showinfo("Success", f"Selected {len(features)} features")

        # Update feature info display
        self.feature_info.delete(1.0, tk.END)
        self.feature_info.insert(tk.END, f"Selected Features ({len(features)}):\n")
        for f in features[:15]:
            self.feature_info.insert(tk.END, f"• {f}\n")

    def apply_pca(self):
        """Apply PCA dimensionality reduction"""
        if self.app.core.X_train is None:
            messagebox.showwarning("No Data", "Split data first (click Split & Train).")
            return

        try:
            n_components = int(self.pca_components.get())
        except ValueError:
            messagebox.showerror("Invalid", "Enter a valid number of components")
            return

        if n_components > len(self.app.core.feature_names):
            messagebox.showerror(
                "Too Many",
                f"Can't have more than {len(self.app.core.feature_names)} components",
            )
            return

        def do_pca():
            return self.app.core.apply_pca(n_components)

        def display(result):
            explained = result["explained_variance_ratio"]

            msg = f"PCA Applied: {n_components} components\n"
            msg += "Explained variance per component:\n"
            for i, var in enumerate(explained):
                msg += f"  PC{i + 1}: {var:.2%}\n"
            msg += f"\nTotal explained variance: {sum(explained):.2%}"

            messagebox.showinfo("PCA Complete", msg)
            self.update_feature_status()

            # Update feature info display
            self.feature_info.delete(1.0, tk.END)
            self.feature_info.insert(
                tk.END, f"PCA Features ({n_components} components):\n"
            )
            for i in range(n_components):
                self.feature_info.insert(tk.END, f"• PC{i + 1}\n")

        self.app.run_threaded(do_pca, display)

    def restore_from_pca(self):
        """Restore original features after PCA"""
        result = self.app.core.restore_from_pca()
        if result:
            self.update_feature_status()
            # Update feature info display
            self.feature_info.delete(1.0, tk.END)
            self.feature_info.insert(
                tk.END, f"Features ({len(self.app.core.feature_names)}):\n"
            )
            for f in self.app.core.feature_names[:15]:
                self.feature_info.insert(tk.END, f"• {f}\n")
            messagebox.showinfo("Restored", "Original features restored")
        else:
            messagebox.showinfo("Info", "No PCA applied or already restored")

    def display_comparison(self, df):
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, "🏆 MODEL LEADERBOARD\n")
        self.results_text.insert(tk.END, "=" * 70 + "\n\n")

        # Add ranking
        if self.app.core.problem_type == "classification":
            df_ranked = df.copy()
            df_ranked.insert(0, "Rank", range(1, len(df_ranked) + 1))
            table_str = df_ranked.to_string(index=False)
        else:
            df_ranked = df.copy()
            df_ranked.insert(0, "Rank", range(1, len(df_ranked) + 1))
            table_str = df_ranked.to_string(index=False)

        self.results_text.insert(tk.END, table_str)
        self.results_text.insert(
            tk.END, "\n\n✅ Best Model: " + self.app.core.model_name
        )

        # Remove any existing save button
        if hasattr(self, "_save_btn_frame"):
            self._save_btn_frame.destroy()

        # Create a new frame and button
        self._save_btn_frame = ttk.Frame(self.results_text.master)
        self._save_btn_frame.pack(pady=5)
        ttk.Button(
            self._save_btn_frame,
            text="💾 Save Comparison as CSV",
            command=lambda: self.save_comparison_csv(df),
        ).pack()

        messagebox.showinfo("Complete", f"🥇 Best model: {self.app.core.model_name}")
        self.app.tab_predict.build_form()
        self.app.tab_evaluate.update_metrics_display()

    def show_stacking_dialog(self):
        """Show dialog to configure and train a stacked model."""
        trained_models = self.app.core.get_trained_models()

        if len(trained_models) < 2:
            messagebox.showwarning(
                "Insufficient Models",
                "Train at least 2 different models before creating a stacked model.",
            )
            return

        # Create dialog
        dialog = tk.Toplevel(self)
        dialog.title("Train Stacked Model")
        dialog.geometry("400x300")
        dialog.transient(self)
        dialog.grab_set()

        main_frame = ttk.Frame(dialog, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            main_frame,
            text="Combine two trained models:",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor=tk.W, pady=(0, 10))

        # Base model 1
        ttk.Label(main_frame, text="Base Model 1:").pack(anchor=tk.W)
        base1_combo = ttk.Combobox(
            main_frame, values=trained_models, state="readonly", width=45
        )
        base1_combo.pack(fill=tk.X, pady=(0, 10))
        base1_combo.set(trained_models[0])

        # Base model 2
        ttk.Label(main_frame, text="Base Model 2:").pack(anchor=tk.W)
        base2_combo = ttk.Combobox(
            main_frame, values=trained_models, state="readonly", width=45
        )
        base2_combo.pack(fill=tk.X, pady=(0, 10))
        base2_combo.set(
            trained_models[1] if len(trained_models) > 1 else trained_models[0]
        )

        # Meta-learner
        ttk.Label(main_frame, text="Meta-Learner (Final Model):").pack(anchor=tk.W)
        meta_models = [
            "Logistic Regression",
            "Random Forest",
            "Gradient Boosting",
            "SVM",
        ]
        meta_combo = ttk.Combobox(
            main_frame, values=meta_models, state="readonly", width=45
        )
        meta_combo.pack(fill=tk.X, pady=(0, 15))
        meta_combo.set("Logistic Regression")

        def train_stacked():
            base1 = base1_combo.get()
            base2 = base2_combo.get()
            meta = meta_combo.get()

            if base1 == base2:
                messagebox.showwarning("Invalid", "Base models must be different!")
                return

            def do_stacking():
                return self.app.core.train_stacked_model(base1, base2, meta)

            def on_complete(result):
                if result.get("success"):
                    messagebox.showinfo(
                        "Success",
                        f"Stacked model trained!\n\n"
                        f"Base Models: {result.get('base_models')}\n"
                        f"Meta-Learner: {result.get('meta_model')}\n"
                        f"Time: {result.get('elapsed')}s",
                    )
                    dialog.destroy()
                    self.app.tab_predict.build_form()
                    self.app.tab_evaluate.update_metrics_display()
                else:
                    messagebox.showerror("Error", result.get("message"))

            self.app.run_threaded(do_stacking, on_complete)

        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        ttk.Button(
            button_frame, text="🔗 Train Stacked Model", command=train_stacked, width=20
        ).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy, width=10).pack(
            side=tk.RIGHT
        )


# ======================= TAB 5: EVALUATE (Enhanced) =======================
class EvaluateTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.create_widgets()

    def create_widgets(self):
        main = ttk.Frame(self, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        sidebar = ttk.LabelFrame(main, text="Evaluation", padding=10)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))
        sidebar.configure(width=290)
        sidebar.pack_propagate(False)
        canvas = tk.Canvas(sidebar, highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar, orient=tk.VERTICAL, command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        canvas.configure(yscrollcommand=scrollbar.set)
        left = ttk.Frame(canvas)
        window = canvas.create_window((0, 0), window=left, anchor="nw")
        left.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        canvas.bind(
            "<Configure>", lambda e: canvas.itemconfigure(window, width=e.width)
        )
        canvas.bind(
            "<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units")
        )

        # Model selection
        ttk.Label(left, text="Select Model:").pack(fill=tk.X, pady=(0, 5))
        self.model_combo = ttk.Combobox(left, state="readonly", width=25)
        self.model_combo.pack(fill=tk.X, pady=(0, 10))
        self.model_combo.bind("<<ComboboxSelected>>", self.on_model_selected)
        ttk.Button(
            left, text="🔄 Load Model", command=self.load_selected_model, width=28
        ).pack(fill=tk.X, pady=5)

        ttk.Button(
            left, text="📊 Show Metrics", command=self.show_metrics, width=28
        ).pack(fill=tk.X, pady=5)
        ttk.Button(
            left, text="🔢 Confusion Matrix", command=self.show_confusion, width=28
        ).pack(fill=tk.X, pady=5)
        ttk.Button(left, text="📈 ROC Curve", command=self.show_roc, width=28).pack(
            fill=tk.X, pady=5
        )
        ttk.Button(
            left, text="📉 Precision-Recall Curve", command=self.show_pr, width=28
        ).pack(fill=tk.X, pady=5)
        ttk.Button(
            left, text="⭐ Feature Importance", command=self.show_importance, width=28
        ).pack(fill=tk.X, pady=5)
        ttk.Button(
            left,
            text="🔍 Permutation Importance",
            command=self.show_perm_importance,
            width=28,
        ).pack(fill=tk.X, pady=5)
        ttk.Button(
            left, text="📊 Residual Plot", command=self.show_residuals, width=28
        ).pack(fill=tk.X, pady=5)
        ttk.Button(
            left, text="🔍 Cross-Validation Score", command=self.show_cv, width=28
        ).pack(fill=tk.X, pady=5)
        ttk.Button(
            left,
            text="⚠️ Misclassified Samples",
            command=self.show_misclassified,
            width=28,
        ).pack(fill=tk.X, pady=5)

        right = ttk.LabelFrame(main, text="Visualization", padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.metrics_text = scrolledtext.ScrolledText(right, height=7, font=FONT_MONO)
        self.metrics_text.pack(fill=tk.X, pady=(0, 10))
        self.chart_label = tk.Label(right, bg="white", relief=tk.SUNKEN, bd=1)
        self.chart_label.pack(fill=tk.BOTH, expand=True)

    def update_metrics_display(self):
        if self.app.core.model:
            self.show_metrics()
        self.refresh_model_combo()

    def refresh_model_combo(self):
        """Refresh the model selection dropdown."""
        models = self.app.core.get_trained_models()
        self.model_combo["values"] = models
        if models and self.app.core.model_name in models:
            self.model_combo.set(self.app.core.model_name)
        elif models:
            self.model_combo.set(models[0])

    def on_model_selected(self, event=None):
        """Handle model selection from dropdown."""
        pass  # Selection will be loaded when button is clicked

    def load_selected_model(self):
        model_name = self.model_combo.get()
        if not model_name:
            messagebox.showwarning("No Selection", "Please select a model first.")
            return

        # Use set_evaluation_model instead of select_model
        result = self.app.core.set_evaluation_model(model_name)
        if result.get("success"):
            messagebox.showinfo(
                "Model Loaded",
                f"Evaluation model set to: {model_name}\nNow Predict tab will use this model.",
            )
            self.show_metrics()
            # Also update Predict tab form to use selected features
            self.app.tab_predict.build_form()
        else:
            messagebox.showerror("Error", result.get("message"))
        self.app.tab_predict.update_active_model_label()

    def show_metrics(self):
        if self.app.core.model is None:
            messagebox.showwarning("No Model", "Train a model first.")
            return
        self.app.run_threaded(self.app.core.get_metrics, self.display_metrics)

    def display_metrics(self, metrics):
        self.metrics_text.delete(1.0, tk.END)
        for k, v in metrics.items():
            if not isinstance(v, str):
                self.metrics_text.insert(tk.END, f"{k.upper()}: {v}\n")
            else:
                self.metrics_text.insert(tk.END, f"{k}:\n{v}\n")

    def show_confusion(self):
        if self.app.core.model is None:
            messagebox.showwarning("No Model", "Train a model first.")
            return
        if self.app.core.problem_type != "classification":
            messagebox.showinfo("Info", "Confusion matrix only for classification.")
            return
        self.app.run_threaded(
            self.app.core.plot_confusion_matrix,
            lambda img: self.app.display_image(img, self.chart_label),
        )

    def show_roc(self):
        if self.app.core.model is None:
            messagebox.showwarning("No Model", "Train a model first.")
            return
        if self.app.core.problem_type != "classification":
            messagebox.showinfo("Info", "ROC curve only for classification.")
            return
        self.app.run_threaded(
            self.app.core.plot_roc_curve,
            lambda img: self.app.display_image(img, self.chart_label),
        )

    def show_pr(self):
        if self.app.core.model is None:
            messagebox.showwarning("No Model", "Train a model first.")
            return
        if self.app.core.problem_type != "classification":
            messagebox.showinfo(
                "Info", "Precision-Recall curve only for classification."
            )
            return
        self.app.run_threaded(
            self.app.core.plot_precision_recall_curve,
            lambda img: self.app.display_image(img, self.chart_label),
        )

    def show_importance(self):
        if self.app.core.model is None:
            messagebox.showwarning("No Model", "Train a model first.")
            return
        self.app.run_threaded(
            self.app.core.plot_feature_importance,
            lambda img: self.app.display_image(img, self.chart_label),
        )

    def show_perm_importance(self):
        if self.app.core.model is None:
            messagebox.showwarning("No Model", "Train a model first.")
            return

        def get_perm_imp():
            perm_df = self.app.core.get_permutation_importance(top_n=15)
            return perm_df

        def display_perm(df):
            self.metrics_text.delete(1.0, tk.END)
            self.metrics_text.insert(tk.END, "🔀 PERMUTATION IMPORTANCE\n")
            self.metrics_text.insert(tk.END, "=" * 50 + "\n\n")
            if df.empty:
                self.metrics_text.insert(
                    tk.END, "Could not calculate permutation importance.\n"
                )
            else:
                self.metrics_text.insert(tk.END, df.to_string(index=False))

        self.app.run_threaded(get_perm_imp, display_perm)

    def show_residuals(self):
        if self.app.core.model is None:
            messagebox.showwarning("No Model", "Train a model first.")
            return
        if self.app.core.problem_type != "regression":
            messagebox.showinfo(
                "Info", "Residual plots are for regression models only."
            )
            return
        self.app.run_threaded(
            self.app.core.plot_residuals,
            lambda img: self.app.display_image(img, self.chart_label),
        )

    def show_cv(self):
        if self.app.core.model is None:
            messagebox.showwarning("No Model", "Train a model first.")
            return
        self.app.run_threaded(
            lambda: self.app.core.get_cross_val_score(cv=5), self.display_cv
        )

    def display_cv(self, cv_res):
        self.metrics_text.insert(tk.END, f"\nCROSS-VALIDATION (5-fold)\n{'=' * 30}\n")
        self.metrics_text.insert(
            tk.END, f"Mean {cv_res['metric'].upper()}: {cv_res['mean']}\n"
        )
        self.metrics_text.insert(tk.END, f"Std: {cv_res['std']}\n")
        self.metrics_text.insert(tk.END, f"Scores: {cv_res['scores']}\n")

    def show_misclassified(self):
        if self.app.core.model is None:
            messagebox.showwarning("No Model", "Train a model first.")
            return
        if self.app.core.problem_type != "classification":
            messagebox.showinfo(
                "Info", "Misclassified samples only for classification."
            )
            return

        # Get misclassified samples with limited features
        df_mis = self.app.core.get_misclassified_samples(max_samples=30)

        if df_mis.empty:
            messagebox.showinfo("Info", "No misclassified samples found.")
            return

        # Create a new window with scrollbars
        win = tk.Toplevel(self)
        win.title(f"Misclassified Samples ({len(df_mis)} samples)")
        win.geometry("800x500")

        # Create frame with scrollbars
        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True)

        # Canvas for scrolling
        canvas = tk.Canvas(frame)
        h_scroll = ttk.Scrollbar(frame, orient=tk.HORIZONTAL, command=canvas.xview)
        v_scroll = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        h_scroll.pack(side=tk.BOTTOM, fill=tk.X)
        v_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Treeview inside canvas
        tree_frame = ttk.Frame(canvas)
        canvas.create_window((0, 0), window=tree_frame, anchor="nw")

        # Create treeview
        tree = ttk.Treeview(
            tree_frame, columns=list(df_mis.columns), show="headings", height=20
        )
        for col in df_mis.columns:
            tree.heading(col, text=col)
            tree.column(col, width=120 if col != "Correct?" else 80)

        # Add data
        for _, row in df_mis.iterrows():
            values = []
            for col in df_mis.columns:
                val = row[col]
                if col == "Correct?":
                    values.append(val)
                elif isinstance(val, float):
                    values.append(f"{val:.2f}")
                else:
                    values.append(str(val))
            tree.insert("", tk.END, values=values)

        tree.pack()

        # Add a label showing count
        wrong_count = len(df_mis[df_mis["Correct?"] == "✗"])
        ttk.Label(
            win,
            text=f"Total misclassified: {wrong_count} out of {len(self.app.core.y_test)} test samples",
            font=("Segoe UI", 10),
        ).pack(pady=5)

        # Update scroll region
        def configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        tree_frame.bind("<Configure>", configure)


# ======================= TAB 6: INSIGHTS (Enhanced) =======================
class InsightsTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.create_widgets()

    def create_widgets(self):
        main = ttk.Frame(self, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(main, text="Insight Controls", padding=15)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 15))

        ttk.Button(
            left,
            text="✨ GENERATE INSIGHTS",
            command=self.generate_insights,
            width=30,
            style="Accent.TButton",
        ).pack(fill=tk.X, pady=10)
        ttk.Button(
            left, text="📋 Copy to Clipboard", command=self.copy_insights, width=30
        ).pack(fill=tk.X, pady=5)
        ttk.Button(
            left, text="💾 Save to File", command=self.save_insights, width=30
        ).pack(fill=tk.X, pady=5)
        ttk.Button(
            left, text="📊 Show Health Score", command=self.show_health, width=30
        ).pack(fill=tk.X, pady=5)

        right = ttk.LabelFrame(main, text="📊 Data Insights", padding=10)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.insights_text = scrolledtext.ScrolledText(
            right, font=("Consolas", 10), wrap=tk.WORD
        )
        self.insights_text.pack(fill=tk.BOTH, expand=True)

    def generate_insights(self):
        if self.app.core.df is None:
            messagebox.showwarning("No Data", "Load a dataset first.")
            return
        self.app.run_threaded(self.app.core.generate_insights, self.display_insights)

    def display_insights(self, insights):
        self.insights_text.delete(1.0, tk.END)
        for i in insights:
            self.insights_text.insert(tk.END, f"{i}\n")

    def show_health(self):
        health = self.app.core.get_dataset_health_score()
        score = health["score"]
        grade = health["grade"]

        # Color code by grade
        color_map = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴", "F": "🔴"}
        color = color_map.get(grade, "⚪")

        msg = f"{color} Dataset Health Score: {score}/100 ({grade})\n\n"

        # Show detailed metrics if available
        info = self.app.core.get_info()
        total_cells = info["rows"] * info["cols"]
        missing_pct = (
            (info["total_missing"] / total_cells * 100) if total_cells > 0 else 0
        )
        dup_pct = (info["duplicates"] / info["rows"] * 100) if info["rows"] > 0 else 0
        complete_pct = max(0, 100 - missing_pct)
        unique_pct = max(0, 100 - dup_pct)

        msg += (
            "Completeness: "
            + "█" * int(complete_pct / 5)
            + "░" * (20 - int(complete_pct / 5))
            + f" {complete_pct:.1f}%\n"
        )
        msg += (
            "Uniqueness:   "
            + "█" * int(unique_pct / 5)
            + "░" * (20 - int(unique_pct / 5))
            + f" {unique_pct:.1f}%\n"
        )

        if health["issues"]:
            msg += "\n⚠️ Issues detected:\n" + "\n".join(
                f"  • {issue}" for issue in health["issues"][:5]
            )
        else:
            msg += "\n✅ No issues detected!"

        messagebox.showinfo("Dataset Health Score", msg)

    def copy_insights(self):
        text = self.insights_text.get(1.0, tk.END)
        self.app.root.clipboard_clear()
        self.app.root.clipboard_append(text)
        self.app.set_status("Insights copied to clipboard")

    def save_insights(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt", filetypes=[("Text files", "*.txt")]
        )
        if path:
            with open(path, "w") as f:
                f.write(self.insights_text.get(1.0, tk.END))
            messagebox.showinfo("Success", f"Saved to {path}")


# ======================= TAB 7: PREDICT =======================
class PredictTab(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app
        self.input_entries = {}
        self.create_widgets()

    def create_widgets(self):
        main = ttk.Frame(self, padding=15)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(main, text="Manual Input", padding=15)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 15))

        canvas_frame = ttk.Frame(left)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(
            canvas_frame, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_frame = ttk.Frame(self.canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self.predict_btn = ttk.Button(
            left, text="🔮 Predict", command=self.predict_manual, width=25
        )
        self.predict_btn.pack(pady=10)

        right = ttk.LabelFrame(main, text="Prediction Results", padding=15)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self.result_text = scrolledtext.ScrolledText(right, font=FONT_MONO, height=20)
        self.result_text.pack(fill=tk.BOTH, expand=True)
        ttk.Button(
            right,
            text="📁 Batch Predict from CSV",
            command=self.batch_predict,
            width=25,
        ).pack(pady=10)

        # After creating result_text, add:
        self.active_model_label = ttk.Label(
            right, text="", font=("Segoe UI", 9, "italic")
        )
        self.active_model_label.pack(pady=5)

        self.build_form()

    def update_active_model_label(self):
        if self.app.core.eval_model_name:
            self.active_model_label.config(
                text=f"🔍 Using evaluation model: {self.app.core.eval_model_name}"
            )
        else:
            self.active_model_label.config(text="📌 Using main trained model")

    def build_form(self):
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        self.input_entries.clear()
        for item in (
            self.app.core.get_input_form_schema()
            if self.app.core.model is not None
            else []
        ):
            frame = ttk.Frame(self.scrollable_frame)
            frame.pack(fill=tk.X, pady=5)
            ttk.Label(frame, text=item["name"], width=28, anchor=tk.W).pack(
                side=tk.LEFT
            )
            if item["type"] == "categorical":
                entry = ttk.Combobox(frame, values=item.get("options", []), width=35)
            else:
                entry = ttk.Entry(frame, width=38)
                entry.insert(0, str(item.get("mean", "")))
            entry.pack(side=tk.LEFT, padx=5)
            self.input_entries[item["name"]] = entry
        if not self.input_entries:
            ttk.Label(
                self.scrollable_frame,
                text="Train or import a model to enable predictions.",
            ).pack()
        self.update_active_model_label()

    def predict_manual(self):
        if self.app.core.model is None:
            messagebox.showwarning("No Model", "Train a model first.")
            return
        input_dict = {}
        for feat, entry in self.input_entries.items():
            val = entry.get().strip()
            input_dict[feat] = val
        self.app.run_threaded(
            lambda: self.app.core.predict(input_dict), self.display_prediction
        )

    def display_prediction(self, result):
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, f"🔮 PREDICTION RESULT\n{'=' * 40}\n\n")
        self.result_text.insert(tk.END, f"Prediction: {result['prediction']}\n")
        if "confidence" in result:
            self.result_text.insert(tk.END, f"Model probability (uncalibrated): {result['confidence']:.2%}\n")
        if "probabilities" in result:
            self.result_text.insert(tk.END, "\nProbabilities:\n")
            for cls, prob in result["probabilities"].items():
                self.result_text.insert(tk.END, f"  {cls}: {prob:.2%}\n")

    def batch_predict(self):
        if self.app.core.model is None:
            messagebox.showwarning("No Model", "Train a model first.")
            return
        path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not path:
            return
        out_path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV files", "*.csv")]
        )
        if not out_path:
            return
        self.app.run_threaded(
            lambda: self.app.core.predict_from_csv(path, out_path),
            lambda r: messagebox.showinfo(
                "Success", f"Predictions saved to {out_path}"
            ),
        )


# ======================= MAIN =======================
if __name__ == "__main__":
    app = DataYabApp()
    app.run()
