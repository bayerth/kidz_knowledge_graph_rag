import traceback

import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import src.rag.ontology as ontology
import src.rag.retriever as retriever


class Logger:
    def info(self, msg, **kwargs):
        print(msg)
        print("--------------------------")

    def debug(self, msg, **kwargs):
        print(msg)
        print("--------------------------")

    def error(self, msg, **kwargs):
        print("==========================")
        print(msg)
        print("==========================")


class GUILogger:
    def __init__(self, window, logger=None):
        self.window = window
        self.logger = logger

    def info(self, msg, **kwargs):
        self.window.log(msg)
        self.window.update_idletasks()

    def debug(self, msg, **kwargs):
        self.window.log(msg)
        self.window.update_idletasks()

    def error(self, msg, **kwargs):
        self.window.log(msg)
        self.window.update_idletasks()


class App(tk.Tk):
    def __init__(self, ontology, model_dict, question_list=None, logger=Logger(), root=None):
        super().__init__()
        # self.results = asyncio.Queue()
        self.ontology = ontology
        self.model_dict = model_dict
        self.selected_model_id_var = None
        self.selected_model_id = None
        self.selected_model = None
        self.selected_index = None
        self.llm_client = None
        if question_list is None:
            question_list = []
        self.question_list = ["Enter Text"] + question_list
        self.logger = logger
        self.gui_logger = None
        self.root = root
        self.title("KIDZ: RAG Prototype")

        # Configure grid weights for resizing behavior
        self.columnconfigure(0, weight=1)
        # rowconfigure for growing areas will be set after layout rows are defined

        # Styles
        self.style = ttk.Style(self)
        try:
            self.style.theme_use("clam")
        except Exception:
            pass

        # Menubar with Model selection
        first_model_id = list(self.model_dict.keys())[0]
        self.model_id = first_model_id
        self.selected_model_id_var = tk.StringVar(value=self.model_id)
        menubar = tk.Menu(self)
        model_menu = tk.Menu(menubar, tearoff=0)
        for model_id in self.model_dict.keys():
            model_menu.add_radiobutton(
                    label=model_id,
                    value=model_id,
                    variable=self.selected_model_id_var,
                    command=lambda mm=model_id: self.set_model_info(mm)
                    )
        menubar.add_cascade(label="Model", menu=model_menu)
        # Help menu with simple proxy messages
        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="Documentation", command=self.show_documentation)
        help_menu.add_command(label="About", command=self.show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        self.config(menu=menubar)
        # self.selected_model_id_var.set(first_model_id)

        # Input label, dropdown and text
        ttk.Label(self, text="Input").grid(row=0, column=0, sticky="w", padx=8, pady=(8, 2))
        # Dropdown above the first text field
        self.input_dropdown = ttk.Combobox(self, state="readonly")
        self.input_dropdown['values'] = self.question_list
        self.input_dropdown.current(0)
        self.input_dropdown.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 4))
        self.input_dropdown.bind('<<ComboboxSelected>>', lambda event: self.on_dropdown_select(event))

        # 4. Define the handler function

        self.input_text = tk.Text(self, width=80, height=3, wrap="word")
        self.input_text.grid(row=2, column=0, sticky="nsew", padx=8)
        self.input_text.insert("1.0", "List all models for dataset niro")

        # One-line model_info field between input and output
        self.model_info = tk.Text(self, width=80, height=1, wrap="word")
        self.model_info.grid(row=3, column=0, sticky="ew", padx=8, pady=(6, 6))
        self.set_model_info_text()
        # self.model_info_var = tk.StringVar(value=f"Model: {self.selected_model_id}")
        # self.model_info = ttk.Entry(self, textvariable=self.model_info_var, state="readonly")
        # self.model_info.grid(row=3, column=0, sticky="ew", padx=8, pady=(6, 6))

        # Controls (Start + Clear buttons)
        controls_frame = ttk.Frame(self)
        controls_frame.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))
        controls_frame.columnconfigure(0, weight=1)

        # Place buttons aligned to the right
        self.clear_logs_btn = ttk.Button(controls_frame, text="Clear log_text", command=self.clear_logs)
        self.clear_logs_btn.pack(side="right", padx=(4, 0))

        self.clear_output_btn = ttk.Button(controls_frame, text="Clear output", command=self.clear_output)
        self.clear_output_btn.pack(side="right", padx=(4, 0))

        self.start_btn = ttk.Button(controls_frame, text="Start", command=self.on_start_button_click)
        self.start_btn.pack(side="right")

        # Output label and text with scrollbar
        ttk.Label(self, text="Output").grid(row=5, column=0, sticky="w", padx=8)
        self.output_frame = ttk.Frame(self)
        self.output_frame.grid(row=6, column=0, sticky="nsew", padx=8)
        self.output_frame.columnconfigure(0, weight=1)
        self.output_frame.rowconfigure(0, weight=1)
        self.output_text = tk.Text(self.output_frame, width=80, height=10, wrap="word")
        self.output_text.grid(row=0, column=0, sticky="nsew")
        out_scroll = ttk.Scrollbar(self.output_frame, orient="vertical", command=self.output_text.yview)
        out_scroll.grid(row=0, column=1, sticky="ns")
        self.output_text.configure(yscrollcommand=out_scroll.set)

        # Logs label and text with scrollbar
        ttk.Label(self, text="Logs").grid(row=7, column=0, sticky="w", padx=8, pady=(8, 0))
        self.logs_frame = ttk.Frame(self)
        self.logs_frame.grid(row=8, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.logs_frame.columnconfigure(0, weight=1)
        self.logs_frame.rowconfigure(0, weight=1)
        self.logs_text = tk.Text(self.logs_frame, width=80, height=10, wrap="word", state="disabled")
        self.logs_text.grid(row=0, column=0, sticky="nsew")
        logs_scroll = ttk.Scrollbar(self.logs_frame, orient="vertical", command=self.logs_text.yview)
        logs_scroll.grid(row=0, column=1, sticky="ns")
        self.logs_text.configure(yscrollcommand=logs_scroll.set)
        if logger:
            self.logger = GUILogger(self, logger)
        else:
            self.logger = GUILogger(self)
        self.logger.info("GUI logger initialized")

        # Configure grid weights for resizing after rows are finalized
        self.rowconfigure(6, weight=1)  # Output grows
        self.rowconfigure(8, weight=1)  # Logs grow

        # init model
        self.set_model_info(first_model_id)

    def on_dropdown_select(self, event):
        # 'event.widget' is the dropdown that triggered the event
        selected_value = self.input_dropdown.get()
        self.selected_index = self.input_dropdown.current()
        print(f"User selected: {selected_value}")
        if self.selected_index == 0:
            self.input_text.delete("1.0", "end")
        else:
            self.input_text.delete("1.0", "end")
            self.input_text.insert("1.0", self.question_list[self.selected_index])

    def set_model_info_text(self):
        self.model_info.delete("1.0", "end")
        self.model_info.insert("1.0", f"Model: {self.selected_model_id_var.get()}")

    def set_model_info(self, model_name: str):
        """Callback when a model is selected from the menu; updates model_info."""
        try:
            print(f"Setting model: {model_name}")
            self.selected_model_id_var.set(model_name)
            print(f"model_id_var={self.selected_model_id_var.get()}")
            self.selected_model_id = model_name
            self.selected_model_id = model_name
            self.selected_model = self.model_dict[model_name]
            # self.model_info_var.set(f"Model: {model_name}")
            self.set_model_info_text()
        except Exception:
            self.logger.error(f"Error setting model: {model_name}")
            traceback_string = traceback.format_exc()
            self.logger.error(traceback_string)
            messagebox.showerror(title="Error", message=traceback_string)

    def on_start_button_click(self):
        user_query = self.input_text.get("1.0", "end-1c")
        try:
            response_msg, retrieved_node_dict = retriever.rag(
                    user_query,
                    ontology=self.ontology,
                    llm_client=self.selected_model,
                    logger=self.logger
                    )
            text = str(response_msg)
            self.output_text.delete("1.0", "end")
            self.output_text.insert("1.0", text)
        except Exception as e:
            self.logger.error(f"Error in rag: {e}")
            traceback_string = traceback.format_exc()
            self.logger.error(traceback_string)
            messagebox.showerror(title="Error", message=traceback_string)

    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_msg = f"[{timestamp}] {message}\n"
        self.logs_text.configure(state="normal")
        self.logs_text.insert("end", full_msg)
        self.logs_text.see("end")
        self.logs_text.configure(state="disabled")

    # --- Clear helpers ---
    def clear_output(self):
        """Clear the Output text widget."""
        try:
            self.output_text.delete("1.0", "end")
        except Exception:
            pass

    def clear_logs(self):
        """Clear the Logs text widget (temporarily enable, then disable again)."""
        try:
            self.logs_text.configure(state="normal")
            self.logs_text.delete("1.0", "end")
            self.logs_text.configure(state="disabled")
        except Exception:
            pass

    # --- Help menu callbacks ---
    def show_documentation(self):
        """Display a simple proxy message for Documentation."""
        messagebox.showinfo(
                title="Documentation",
                message=(
                        "Please see the project README or wiki for usage instructions."
                )
                )

    def show_about(self):
        """Display a simple About dialog."""
        messagebox.showinfo(
                title="About",
                message=(
                        "KIDZ: RAG Prototype\n"
                        "A simple GUI to experiment with Retrieval-Augmented Generation.\n\n"
                        "© 2025 KIDZ Project\n"
                        "Contact: wolfram.hoepken@ruw.de"
                )
                )


def main_gui(ontology, model_dict, question_list, logger=Logger()):
    root = tk.Tk()
    app = App(root=root, ontology=ontology, model_dict=model_dict, question_list=question_list, logger=logger)
    app.mainloop()
