from __future__ import annotations

import io
import re
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from quantum_simulator_global import (
        QuantumSimulator,
        _CUDA_BACKEND_NAME,
        _CUDA_STATUS,
        _HAVE_CUDA,
        execute_qasm,
    )
except Exception as exc:
    messagebox.showerror("Import Error", f"Failed to import the simulator.\n\n{exc}")
    raise


APP_TITLE = "QASM Studio"
DEFAULT_QUBITS = 3
DEFAULT_SAMPLE = """OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
measure q -> c;
print_creg;
"""

QASM_KEYWORDS = [
    "OPENQASM",
    "include",
    "gate",
    "qreg",
    "creg",
    "qubit",
    "bit",
    "if",
    "else",
    "for",
    "while",
    "shots",
    "i",
    "id",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "sx",
    "sxdg",
    "p",
    "u",
    "u1",
    "u2",
    "u3",
    "rx",
    "ry",
    "rz",
    "swap",
    "iswap",
    "iswap_theta",
    "iswap_pow",
    "iswapdg",
    "fsim",
    "syc",
    "phased_iswap",
    "phasedfsim",
    "cz_wave",
    "rxx",
    "ryy",
    "rzz",
    "cx",
    "cy",
    "cz",
    "ch",
    "cs",
    "ct",
    "cp",
    "crx",
    "cry",
    "crz",
    "cu1",
    "cu3",
    "toffoli",
    "ccx",
    "cswap",
    "measure",
    "reset",
    "seed",
    "print_state",
    "print_probs",
    "print_creg",
    "noise_bitflip",
    "nbf",
    "noise_phaseflip",
    "npf",
    "noise_depolarizing",
    "ndp",
    "noise_amp",
    "nad",
    "noise_phase",
    "nph",
    "barrier",
    "delay",
]


class QasmStudio(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1200x720")
        self.minsize(960, 600)

        self.current_file: Path | None = None
        self.num_qubits = tk.IntVar(value=DEFAULT_QUBITS)
        self.sim_seed = tk.IntVar(value=42)

        self._build_ui()
        self._new_document(default_sample=True)

    def _build_ui(self):
        self._build_menu()

        top = ttk.Frame(self)
        top.pack(fill="x", padx=8, pady=6)

        ttk.Label(top, text="Qubits").pack(side="left")
        self.spin_qubits = tk.Spinbox(top, from_=1, to=30, textvariable=self.num_qubits, width=5)
        self.spin_qubits.pack(side="left", padx=(4, 16))

        ttk.Label(top, text="Seed").pack(side="left")
        ttk.Entry(top, width=10, textvariable=self.sim_seed).pack(side="left", padx=(4, 16))

        backend_text = _CUDA_BACKEND_NAME if _HAVE_CUDA else "cpu"
        ttk.Label(top, text=f"Backend: {backend_text}").pack(side="left")
        ttk.Label(top, text=_CUDA_STATUS).pack(side="left", padx=(8, 16))

        ttk.Button(top, text="Run (F5)", command=self.run_qasm).pack(side="left", padx=4)
        ttk.Button(top, text="Clear Output", command=self.clear_output).pack(side="left", padx=4)

        panes = ttk.PanedWindow(self, orient="horizontal")
        panes.pack(fill="both", expand=True, padx=8, pady=8)

        left = ttk.Frame(panes)
        self.txt = tk.Text(left, wrap="none", undo=True, font=("Consolas", 12))
        self._attach_scrollbars(left, self.txt)
        self._setup_highlight_tags(self.txt)
        self.txt.bind("<KeyRelease>", self._on_key_release)
        self.txt.bind("<Control-s>", self._save_shortcut)
        self.txt.bind("<F5>", lambda _event: self.run_qasm())
        panes.add(left, weight=3)

        right = ttk.Frame(panes)
        self.out = tk.Text(right, wrap="word", font=("Consolas", 11))
        self._attach_scrollbars(right, self.out)
        panes.add(right, weight=2)

        self.status = ttk.Label(self, anchor="w", relief="sunken")
        self.status.pack(fill="x", side="bottom")
        self._set_status("Ready")

    def _build_menu(self):
        menu = tk.Menu(self)
        self.config(menu=menu)

        file_menu = tk.Menu(menu, tearoff=0)
        file_menu.add_command(label="New", command=self._new_document, accelerator="Ctrl+N")
        file_menu.add_command(label="Open...", command=self._open_file, accelerator="Ctrl+O")
        file_menu.add_command(label="Save", command=self._save_file, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self._save_as)
        file_menu.add_separator()
        file_menu.add_command(label="Load Bell Sample", command=lambda: self._load_sample("bell"))
        file_menu.add_command(label="Load Control-Flow Sample", command=lambda: self._load_sample("flow"))
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.destroy)
        menu.add_cascade(label="File", menu=file_menu)

        run_menu = tk.Menu(menu, tearoff=0)
        run_menu.add_command(label="Run", command=self.run_qasm, accelerator="F5")
        run_menu.add_command(label="Clear Output", command=self.clear_output)
        menu.add_cascade(label="Run", menu=run_menu)

        help_menu = tk.Menu(menu, tearoff=0)
        help_menu.add_command(label="About", command=self._about)
        menu.add_cascade(label="Help", menu=help_menu)

        self.bind("<Control-n>", lambda _event: self._new_document())
        self.bind("<Control-o>", lambda _event: self._open_file())
        self.bind("<Control-s>", self._save_shortcut)

    def _about(self):
        messagebox.showinfo(
            "About",
            "QASM Studio\n\n"
            "Local editor for the SoftQuantum simulator.\n"
            "Supports OpenQASM-style headers, includes, control flow, and shots.",
        )

    def _attach_scrollbars(self, parent: ttk.Frame, widget: tk.Text):
        xscroll = ttk.Scrollbar(parent, orient="horizontal", command=widget.xview)
        yscroll = ttk.Scrollbar(parent, orient="vertical", command=widget.yview)
        widget.configure(xscrollcommand=xscroll.set, yscrollcommand=yscroll.set)
        widget.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)

    def _setup_highlight_tags(self, txt: tk.Text):
        txt.tag_configure("kw", foreground="#0057b8")
        txt.tag_configure("num", foreground="#9b870c")
        txt.tag_configure("com", foreground="#888888")

    def _on_key_release(self, _event=None):
        self._highlight_all()

    def _highlight_all(self):
        content = self.txt.get("1.0", "end-1c")
        for tag in ("kw", "num", "com"):
            self.txt.tag_remove(tag, "1.0", "end")

        for match in re.finditer(r"(//.*?$|#.*?$)", content, flags=re.M):
            self._tag_range(*match.span(), "com")
        for match in re.finditer(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", content):
            self._tag_range(*match.span(), "num")

        pattern = "|".join(sorted(map(re.escape, QASM_KEYWORDS), key=len, reverse=True))
        for match in re.finditer(rf"\b(?:{pattern})\b", content, flags=re.I):
            self._tag_range(*match.span(), "kw")

    def _tag_range(self, start_idx: int, end_idx: int, tag: str):
        content = self.txt.get("1.0", "end-1c")
        self.txt.tag_add(tag, self._to_tk_index(content, start_idx), self._to_tk_index(content, end_idx))

    def _to_tk_index(self, text: str, idx: int) -> str:
        lines = text.splitlines(keepends=True)
        row = 1
        remaining = idx
        for line in lines:
            if remaining <= len(line) - 1:
                return f"{row}.{remaining}"
            remaining -= len(line)
            row += 1
        return f"{row}.0"

    def _new_document(self, default_sample: bool = False):
        self.txt.delete("1.0", "end")
        self.current_file = None
        if default_sample:
            self.txt.insert("1.0", DEFAULT_SAMPLE)
        self._highlight_all()
        self._set_status("New document")

    def _open_file(self):
        path = filedialog.askopenfilename(
            title="Open QASM file",
            filetypes=[("QASM files", "*.qasm"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            text = Path(path).read_text(encoding="utf-8")
            self.txt.delete("1.0", "end")
            self.txt.insert("1.0", text)
            self.current_file = Path(path)
            self._highlight_all()
            self._set_status(f"Opened {self.current_file.name}")
        except Exception as exc:
            messagebox.showerror("Open Error", str(exc))

    def _save_shortcut(self, _event=None):
        self._save_file()
        return "break"

    def _save_file(self):
        if self.current_file is None:
            self._save_as()
            return
        try:
            self.current_file.write_text(self.txt.get("1.0", "end-1c"), encoding="utf-8")
            self._set_status(f"Saved {self.current_file.name}")
        except Exception as exc:
            messagebox.showerror("Save Error", str(exc))

    def _save_as(self):
        path = filedialog.asksaveasfilename(
            title="Save QASM file",
            defaultextension=".qasm",
            filetypes=[("QASM files", "*.qasm"), ("Text files", "*.txt"), ("All files", "*.*")],
        )
        if not path:
            return
        self.current_file = Path(path)
        self._save_file()

    def _load_sample(self, which: str):
        if which == "bell":
            sample = """OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
measure q -> c;
print_creg;
"""
        else:
            sample = """OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
shots 8;
for int i in [0:1] {
    h q[i];
}
measure q -> c;
if (c == 0) {
    x q[0];
    measure q[0] -> c[0];
}
"""
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", sample)
        self._highlight_all()
        self._set_status("Sample loaded")

    def clear_output(self):
        self.out.delete("1.0", "end")

    def _append_output(self, text: str):
        self.out.insert("end", text)
        self.out.see("end")

    def run_qasm(self):
        code = self.txt.get("1.0", "end-1c")
        if not code.strip():
            return

        try:
            sim = QuantumSimulator(max(int(self.num_qubits.get()), 1), seed=int(self.sim_seed.get()))
        except Exception as exc:
            messagebox.showerror("Simulator Error", str(exc))
            return

        buf = io.StringIO()
        base_path = self.current_file.parent if self.current_file else SCRIPT_DIR
        t0 = time.time()
        try:
            with redirect_stdout(buf):
                execute_qasm(sim, lines=code.splitlines(), base_path=base_path)
        except Exception as exc:
            self._append_output(f"\n[error] {exc}\n")
            self._set_status("Execution failed")
            return

        elapsed_ms = (time.time() - t0) * 1000.0
        self._append_output(
            f"\n===== Run complete ({elapsed_ms:.1f} ms, backend={sim.backend_name}) =====\n"
        )
        self._append_output(buf.getvalue() + "\n")
        self._set_status("Execution complete")

    def _set_status(self, message: str):
        self.status.config(text=f" {message}")


def main():
    app = QasmStudio()
    app.mainloop()


if __name__ == "__main__":
    main()
