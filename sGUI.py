"""
sGUI.py — GUI Pengolahan Citra Digital
Fitur:
  - Buka file / folder → navigasi gambar satu per satu
  - 5 kategori analisis → tiap kategori buka window hasil LENGKAP:
      • Grid gambar (original + semua hasil)
      • Histogram intensitas per gambar (seperti histogram.py)
      • Panel statistik ringkasan
  - Tema terang & bersih (biru-teal)
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from datetime import datetime

from core import (
    to_rgb, to_grayscale, to_binary,
    equalize_histogram,
    filter_mean, filter_median, filter_gaussian,
    edge_canny, edge_prewitt, edge_sobel,
    segment_otsu, segment_adaptive, segment_kmeans, segment_watershed,
    contrast_stretching, brightness_adjustment, sharpening,
    ImageNavigator,
    brightness, dominant_channel, segmentation_stats,
)

DEFAULT_FOLDER = "dataset"

C = {
    "bg":        "#F0F4F8",
    "surface":   "#FFFFFF",
    "border":    "#D1DCE8",
    "topbar":    "#1B4F72",
    "sidebar":   "#EBF5FB",

    "text":      "#17202A",
    "subtext":   "#5D6D7E",
    "muted":     "#AEB6BF",

    "blue":      "#2980B9",
    "teal":      "#148F77",
    "teal_lt":   "#A9DFBF",
    "green":     "#1E8449",
    "green_lt":  "#D5F5E3",
    "amber":     "#D68910",
    "amber_lt":  "#FDEBD0",
    "purple":    "#7D3C98",
    "purple_lt": "#E8DAEF",
    "red":       "#C0392B",
    "red_lt":    "#FADBD8",

    "cat1":      ("#2980B9", "#FFFFFF"),
    "cat2":      ("#148F77", "#FFFFFF"),
    "cat3":      ("#7D3C98", "#FFFFFF"),
    "cat4":      ("#D68910", "#FFFFFF"),
    "cat5":      ("#C0392B", "#FFFFFF"),
}

MPL_BG    = "#FFFFFF"
MPL_FIG   = "#F7FAFC"
MPL_TEXT  = "#17202A"
MPL_GRID  = "#D1DCE8"
MPL_HIST1 = "#2980B9"
MPL_HIST2 = "#E74C3C"

def _r(img):   return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
def _gray(img): return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def _bri(gray): return float(gray.mean())
def _shp(img):
    g = _gray(img) if len(img.shape) == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())
def _edgepct(e): return np.count_nonzero(e) / e.size * 100
def _std(img):
    g = _gray(img) if len(img.shape) == 3 else img
    return float(g.std())

CATEGORIES = [
    {
        "label":    "1. Konversi Citra",
        "color":    C["cat1"],
        "desc":     "RGB → Grayscale → Biner",
        "analysis": "rgb_hist",
        "items": [
            ("Original (RGB)",
             lambda i: (_r(i),
                        f"Ukuran: {i.shape[1]}×{i.shape[0]} px\n"
                        f"Mean: {_bri(_gray(i)):.1f} | Std: {_std(i):.1f}")),

            ("Grayscale",
             lambda i: (_gray(i),
                        f"Mean: {_bri(_gray(i)):.1f}/255\n"
                        f"{'⬆ Terang' if _bri(_gray(i))>127 else '⬇ Gelap'} | "
                        f"Weighted 0.299R+0.587G+0.114B")),

            ("Biner (t=127)",
             lambda i: (to_binary(i, 127),
                        f"Putih: {segmentation_stats(to_binary(i,127))['pct_fg']:.1f}%  "
                        f"Hitam: {100-segmentation_stats(to_binary(i,127))['pct_fg']:.1f}%\n"
                        f"Aturan: px > 127 → putih")),
        ],
    },
    {
        "label":    "2. Perbaikan Kualitas",
        "color":    C["cat2"],
        "desc":     "Hist.Eq · Contrast Stretch · Brightness · Sharpening",
        "analysis": "quality_hist",
        "items": [
            ("Original",
             lambda i: (_r(i),
                        f"Kontras (std): {_gray(i).std():.1f}\n"
                        f"Mean: {_bri(_gray(i)):.1f}/255")),

            ("Hist. Equalization",
             lambda i: (equalize_histogram(i),
                        f"Std: {_gray(i).std():.1f} → {equalize_histogram(i).std():.1f}  "
                        f"(Δ{equalize_histogram(i).std()-_gray(i).std():+.1f})\n"
                        f"Redistribusi CDF histogram → 0–255")),

            ("Contrast Stretching",
             lambda i: (_r(contrast_stretching(i)),
                        f"Std: {_gray(i).std():.1f} → {_gray(contrast_stretching(i)).std():.1f}  "
                        f"(Δ{_gray(contrast_stretching(i)).std()-_gray(i).std():+.1f})\n"
                        f"Min-max normalisasi per channel")),

            ("Brightness +60",
             lambda i: (_r(brightness_adjustment(i, 60)),
                        f"Mean: {_bri(_gray(i)):.1f} → "
                        f"{_bri(_gray(brightness_adjustment(i,60))):.1f}  (beta=+60)\n"
                        f"convertScaleAbs(alpha=1, beta=+60)")),

            ("Sharpening ×1.5",
             lambda i: (_r(sharpening(i, 1.5)),
                        f"Sharpness: {_shp(i):.1f} → {_shp(sharpening(i,1.5)):.1f}\n"
                        f"Unsharp mask, strength=1.5")),
        ],
    },
    {
        "label":    "3. Filtering",
        "color":    C["cat3"],
        "desc":     "Mean · Median · Gaussian",
        "analysis": "sharpness_bar",
        "items": [
            ("Original",
             lambda i: (_r(i),
                        f"Sharpness: {_shp(i):.1f}  (baseline)")),

            ("Mean 5×5",
             lambda i: (_r(filter_mean(i, 5)),
                        f"Sharpness: {_shp(filter_mean(i,5)):.1f}  "
                        f"(Δ{_shp(filter_mean(i,5))-_shp(i):+.1f})\n"
                        f"Rata-rata kernel 5×5 (box blur)")),

            ("Median 5×5",
             lambda i: (_r(filter_median(i, 5)),
                        f"Sharpness: {_shp(filter_median(i,5)):.1f}  "
                        f"(Δ{_shp(filter_median(i,5))-_shp(i):+.1f})\n"
                        f"Nilai tengah kernel 5×5 (non-linear)")),

            ("Gaussian 7×7",
             lambda i: (_r(filter_gaussian(i, 7)),
                        f"Sharpness: {_shp(filter_gaussian(i,7)):.1f}  "
                        f"(Δ{_shp(filter_gaussian(i,7))-_shp(i):+.1f})\n"
                        f"Kernel Gaussian 7×7, σ=auto")),
        ],
    },
    {
        "label":    "4. Deteksi Tepi",
        "color":    C["cat4"],
        "desc":     "Sobel · Canny · Prewitt",
        "analysis": "edge_bar",
        "items": [
            ("Grayscale (referensi)",
             lambda i: (_gray(i),
                        f"Sharpness: {_shp(i):.1f}  (input ke detector)")),

            ("Sobel",
             lambda i: (edge_sobel(i),
                        f"Piksel tepi: {_edgepct(edge_sobel(i)):.2f}%\n"
                        f"Sobel 3×3 Gx+Gy → magnitude")),

            ("Canny (100/200)",
             lambda i: (edge_canny(i, 100, 200),
                        f"Piksel tepi: {_edgepct(edge_canny(i,100,200)):.2f}%\n"
                        f"Gaussian→Sobel→NMS→Hysteresis")),

            ("Prewitt",
             lambda i: (edge_prewitt(i),
                        f"Piksel tepi: {_edgepct(edge_prewitt(i)):.2f}%\n"
                        f"Kernel Prewitt 3×3 Gx+Gy → magnitude")),
        ],
    },
    {
        "label":    "5. Segmentasi",
        "color":    C["cat5"],
        "desc":     "Otsu · Adaptive · K-Means · Watershed",
        "analysis": "segment_bar",
        "items": [
            ("Original",
             lambda i: (_r(i),
                        f"Mean: {_bri(_gray(i)):.1f}/255")),

            ("Otsu",
             lambda i: (segment_otsu(i)[0],
                        f"Threshold otomatis: t={segment_otsu(i)[1]:.0f}\n"
                        f"FG: {segmentation_stats(segment_otsu(i)[0])['pct_fg']:.1f}%  "
                        f"BG: {100-segmentation_stats(segment_otsu(i)[0])['pct_fg']:.1f}%")),

            ("Adaptive",
             lambda i: (segment_adaptive(i),
                        f"FG: {segmentation_stats(segment_adaptive(i))['pct_fg']:.1f}%\n"
                        f"Gaussian-C block=11, C=2")),

            ("K-Means (k=3)",
             lambda i: (_r(segment_kmeans(i, 3)),
                        f"3 cluster warna\n"
                        f"K-Means iter=20, ε=0.5")),

            ("Watershed",
             lambda i: (_r(segment_watershed(i)),
                        f"Batas segmen = merah\n"
                        f"Otsu→morph open→dist→watershed")),
        ],
    },
]

class SimpleApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Pengolahan Citra Digital")
        self.root.geometry("920x640")
        self.root.configure(bg=C["bg"])
        self.root.resizable(True, True)

        self.img_bgr: np.ndarray | None = None
        self._navigator: ImageNavigator | None = None
        self._tk_preview = None

        self._build_ui()
        self._auto_load_default()

    def _build_ui(self):
        topbar = tk.Frame(self.root, bg=C["topbar"])
        topbar.pack(fill=tk.X)

        tk.Label(topbar, text="PENGOLAHAN CITRA DIGITAL",
                 font=("Arial", 13, "bold"),
                 bg=C["topbar"], fg="#FFFFFF").pack(side=tk.LEFT, padx=14, pady=10)

        nav_fr = tk.Frame(topbar, bg=C["topbar"])
        nav_fr.pack(side=tk.RIGHT, padx=10)

        self._btn(nav_fr, "◀ Prev", self._nav_prev, C["blue"], "#FFFFFF").pack(
            side=tk.LEFT, padx=3, pady=7)
        self._nav_lbl = tk.Label(nav_fr, text="–/–",
                                  font=("Consolas", 9, "bold"),
                                  bg=C["topbar"], fg="#90CAF9")
        self._nav_lbl.pack(side=tk.LEFT, padx=6)
        self._btn(nav_fr, "Next ▶", self._nav_next, C["blue"], "#FFFFFF").pack(
            side=tk.LEFT, padx=3, pady=7)
        self._btn(nav_fr, "🔀", self._nav_shuffle, "#455A64", "#FFFFFF").pack(
            side=tk.LEFT, padx=6, pady=7)

        btn_fr = tk.Frame(topbar, bg=C["topbar"])
        btn_fr.pack(side=tk.LEFT, padx=6)
        self._btn(btn_fr, "📁 Buka File",   self._open_file,   "#27AE60", "#FFFFFF").pack(
            side=tk.LEFT, padx=3, pady=7)
        self._btn(btn_fr, "📂 Buka Folder", self._open_folder, "#2ECC71", "#FFFFFF").pack(
            side=tk.LEFT, padx=3, pady=7)

        body = tk.Frame(self.root, bg=C["bg"])
        body.pack(fill=tk.BOTH, expand=True)

        self._build_sidebar(body)

        right = tk.Frame(body, bg=C["bg"])
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=12, pady=10)

        tk.Label(right, text="Pilih kategori analisis:",
                 font=("Arial", 10, "bold"),
                 bg=C["bg"], fg=C["subtext"]).pack(anchor="w", pady=(0, 8))

        for cat in CATEGORIES:
            self._build_cat_row(right, cat)

        self._status = tk.StringVar(value="Buka gambar atau folder untuk memulai.")
        status_bar = tk.Label(self.root, textvariable=self._status,
                               font=("Consolas", 8),
                               bg=C["topbar"], fg="#A8D8EA",
                               anchor="w", padx=10)
        status_bar.pack(fill=tk.X, side=tk.BOTTOM)

    def _build_sidebar(self, parent):
        sidebar = tk.Frame(parent, bg=C["sidebar"],
                            width=240, relief=tk.FLAT,
                            highlightbackground=C["border"],
                            highlightthickness=1)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(10, 0), pady=10)
        sidebar.pack_propagate(False)

        tk.Label(sidebar, text="P R E V I E W",
                 font=("Arial", 8, "bold"),
                 bg=C["sidebar"], fg=C["subtext"]).pack(pady=(10, 4))

        tk.Frame(sidebar, bg=C["border"], height=1).pack(fill=tk.X, padx=10)

        self._preview_lbl = tk.Label(sidebar, bg=C["sidebar"],
                                      text="Tidak ada gambar",
                                      fg=C["muted"], font=("Arial", 9))
        self._preview_lbl.pack(fill=tk.BOTH, expand=True, padx=6, pady=(6, 4))

        tk.Frame(sidebar, bg=C["border"], height=1).pack(fill=tk.X, padx=10)
        self._info_lbl = tk.Label(sidebar, text="",
                                   font=("Consolas", 8),
                                   bg=C["sidebar"], fg=C["subtext"],
                                   justify="left", anchor="w")
        self._info_lbl.pack(fill=tk.X, padx=10, pady=6)

    def _build_cat_row(self, parent, cat):
        bg_btn, fg_btn = cat["color"]

        frame = tk.Frame(parent, bg=C["surface"],
                          relief=tk.FLAT,
                          highlightbackground=C["border"],
                          highlightthickness=1)
        frame.pack(fill=tk.X, pady=5)

        btn = tk.Button(
            frame, text=cat["label"],
            command=lambda c=cat: self._open_category(c),
            bg=bg_btn, fg=fg_btn,
            font=("Arial", 10, "bold"),
            relief=tk.FLAT, cursor="hand2",
            activebackground="#17202A", activeforeground="#FFFFFF",
            width=22, padx=6, pady=10,
        )
        btn.pack(side=tk.LEFT, padx=10, pady=6)

        desc_fr = tk.Frame(frame, bg=C["surface"])
        desc_fr.pack(side=tk.LEFT, fill=tk.Y, pady=6)

        tk.Label(desc_fr, text=cat["desc"],
                 font=("Arial", 9, "italic"),
                 bg=C["surface"], fg=C["subtext"],
                 anchor="w").pack(anchor="w")

    def _btn(self, parent, text, cmd, bg, fg="#FFFFFF"):
        return tk.Button(parent, text=text, command=cmd,
                          bg=bg, fg=fg,
                          font=("Arial", 9, "bold"),
                          relief=tk.FLAT, cursor="hand2",
                          activebackground="#17202A", activeforeground="#FFFFFF",
                          padx=8, pady=3)

    def _show_preview(self, img_bgr: np.ndarray):
        if len(img_bgr.shape) == 2:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        else:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        scale = min(210 / w, 280 / h, 1.0)
        disp  = cv2.resize(rgb, (int(w * scale), int(h * scale)),
                            interpolation=cv2.INTER_AREA)
        img_tk = ImageTk.PhotoImage(Image.fromarray(disp))
        self._preview_lbl.config(image=img_tk, text="")
        self._tk_preview = img_tk

    def _load(self, path: str):
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", f"Gagal membaca:\n{path}")
            return
        self.img_bgr = img
        self._show_preview(img)
        h, w = img.shape[:2]
        gray = _gray(img)
        dom, means = dominant_channel(img)
        b = brightness(gray)
        self._info_lbl.config(
            text=(f"  {os.path.basename(path)}\n"
                  f"  {w} × {h} px\n"
                  f"  Mean: {b:.0f}/255  Std: {gray.std():.1f}\n"
                  f"  Dominan: {dom}  "
                  f"(R={means['R']:.0f} G={means['G']:.0f} B={means['B']:.0f})\n"
                  f"  {'⬆ Terang' if b>127 else '⬇ Gelap'}")
        )
        self._status.set(
            f"📂 {os.path.basename(path)}  ({w}×{h}px)  |  "
            f"Kecerahan: {b:.0f}/255  |  Dominan channel: {dom}"
        )
        self._update_nav()

    def _update_nav(self):
        if self._navigator and self._navigator.has_images():
            self._nav_lbl.config(
                text=f"{self._navigator.index+1}/{self._navigator.total}")
        else:
            self._nav_lbl.config(text="–/–")

    def _auto_load_default(self):
        if os.path.isdir(DEFAULT_FOLDER):
            try:
                self._navigator = ImageNavigator(DEFAULT_FOLDER)
                self._load(self._navigator.current())
            except Exception:
                pass

    def _open_file(self):
        path = filedialog.askopenfilename(
            filetypes=[("Gambar", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        self._navigator = None
        self._update_nav()
        self._load(path)

    def _open_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return
        try:
            self._navigator = ImageNavigator(folder)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return
        self._load(self._navigator.current())
        self._status.set(
            f"📂 Folder: {folder}  ({self._navigator.total} gambar)")

    def _nav_next(self):
        if not self._navigator:
            messagebox.showinfo("Info", "Buka folder dulu."); return
        self._load(self._navigator.next())

    def _nav_prev(self):
        if not self._navigator:
            messagebox.showinfo("Info", "Buka folder dulu."); return
        self._load(self._navigator.prev())

    def _nav_shuffle(self):
        if not self._navigator:
            return
        self._navigator.reshuffle()
        self._load(self._navigator.current())

    def _open_category(self, cat: dict):
        if self.img_bgr is None:
            messagebox.showwarning("Peringatan", "Buka gambar terlebih dahulu!")
            return

        img   = self.img_bgr
        items = cat["items"]
        n     = len(items)

        results = []
        for title, fn in items:
            try:
                disp, info = fn(img)
            except Exception as e:
                disp = np.zeros((80, 80), np.uint8)
                info = f"Error: {e}"
            results.append((title, disp, info))

        cols     = 3 if n in [5, 6] else min(n, 4)
        img_rows = (n + cols - 1) // cols

        analysis_id = cat.get("analysis", "none")
        extra_rows = 0 if analysis_id == "none" else 1
        total_rows = img_rows + extra_rows

        height_ratios = [2.8] * img_rows
        if extra_rows:
            height_ratios += [1.8]

        # FIXED: Removed fig.suptitle(...) to get rid of the top text banner inside the figure
        fig = plt.figure(figsize=(4.0 * cols, 3.2 * img_rows + (1.8 if extra_rows else 0.2)),
                          facecolor=MPL_FIG, layout="constrained")

        gs = gridspec.GridSpec(
            total_rows, cols,
            figure=fig,
            hspace=0.25, wspace=0.20,
            height_ratios=height_ratios,
        )

        for idx, (title, disp, info) in enumerate(results):
            row = idx // cols
            col = idx % cols
            ax  = fig.add_subplot(gs[row, col])
            if len(disp.shape) == 2:
                ax.imshow(disp, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(disp)
            ax.set_title(f"{title}\n{info}",
                          fontsize=8.5, color=MPL_TEXT, pad=5, linespacing=1.3)
            ax.axis("off")
            ax.set_facecolor(MPL_BG)
            for sp in ax.spines.values():
                sp.set_edgecolor(C["border"])

        for idx in range(n, img_rows * cols):
            ax_d = fig.add_subplot(gs[idx // cols, idx % cols])
            ax_d.set_visible(False)

        if extra_rows:
            self._plot_analysis(fig, gs, img_rows, cols, img, results, analysis_id)

        bg_btn, fg_btn = cat["color"]
        win = tk.Toplevel(self.root)
        win.title(cat["label"])
        win.configure(bg=C["bg"])

        hdr = tk.Frame(win, bg=bg_btn)
        hdr.pack(fill=tk.X, side=tk.TOP)
        tk.Label(hdr, text=cat["label"],
                  font=("Arial", 12, "bold"),
                  bg=bg_btn, fg=fg_btn).pack(side=tk.LEFT, padx=14, pady=8)
        tk.Label(hdr, text=cat["desc"],
                  font=("Arial", 9, "italic"),
                  bg=bg_btn, fg="#FFFFFF").pack(side=tk.LEFT, padx=4)

        # FIXED: Brought the footer layout directly underneath the header / graph using normal sizing
        footer = tk.Frame(win, bg=C["surface"],
                           highlightbackground=C["border"],
                           highlightthickness=1)
        footer.pack(fill=tk.X, side=tk.BOTTOM)

        canvas_mpl = FigureCanvasTkAgg(fig, master=win)
        canvas_mpl.draw()
        canvas_mpl.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

        def _save_figure():
            os.makedirs(out_dir, exist_ok=True)
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = cat["label"].replace(" ", "_").replace(".", "")
            path = os.path.join(out_dir, f"{slug}_{ts}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            messagebox.showinfo("Tersimpan", f"Figure disimpan ke:\n{path}", parent=win)

        def _save_images():
            os.makedirs(out_dir, exist_ok=True)
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            slug = cat["label"].replace(" ", "_").replace(".", "")
            saved = []
            for title, disp, _ in results:
                name = title.split("\n")[0].replace(" ", "_").replace("/", "-")
                fname = os.path.join(out_dir, f"{slug}_{name}_{ts}.png")
                if len(disp.shape) == 2:
                    cv2.imwrite(fname, disp)
                else:
                    cv2.imwrite(fname, cv2.cvtColor(disp, cv2.COLOR_RGB2BGR))
                saved.append(os.path.basename(fname))
            messagebox.showinfo(
                "Tersimpan",
                f"{len(saved)} gambar disimpan ke:\n{out_dir}\n\n" + "\n".join(saved),
                parent=win)

        tk.Button(footer, text="💾  Simpan Figure",
                   command=_save_figure,
                   bg=C["teal"], fg="#FFFFFF",
                   font=("Arial", 9, "bold"), relief=tk.FLAT,
                   cursor="hand2", padx=12, pady=6).pack(side=tk.LEFT, padx=10, pady=6)

        tk.Button(footer, text="🖼  Simpan Gambar",
                   command=_save_images,
                   bg=C["blue"], fg="#FFFFFF",
                   font=("Arial", 9, "bold"), relief=tk.FLAT,
                   cursor="hand2", padx=12, pady=6).pack(side=tk.LEFT, padx=4, pady=6)

        tk.Button(footer, text="✕  Tutup",
                   command=lambda: (plt.close(fig), win.destroy()),
                   bg=C["red"], fg="#FFFFFF",
                   font=("Arial", 9, "bold"), relief=tk.FLAT,
                   cursor="hand2", padx=12, pady=6).pack(side=tk.LEFT, padx=4, pady=6)

        gray_img = _gray(img)
        last_disp = results[-1][1]
        if len(last_disp.shape) == 3:
            last_disp = cv2.cvtColor(cv2.cvtColor(last_disp, cv2.COLOR_RGB2BGR),
                                      cv2.COLOR_BGR2GRAY)
        delta_std = last_disp.std() - gray_img.std()
        info_text = (f"Original std={gray_img.std():.1f}  →  "
                     f"Hasil terakhir std={last_disp.std():.1f}  "
                     f"(Δ{delta_std:+.1f})")
        tk.Label(footer, text=info_text,
                  font=("Consolas", 8),
                  bg=C["surface"], fg=C["subtext"]).pack(side=tk.LEFT, padx=16)

        win.protocol("WM_DELETE_WINDOW", lambda: (plt.close(fig), win.destroy()))

        # FIXED: Removed the fixed math geometry string. Let Tkinter calculate the dimensions naturally 
        # based on the widget constraints, preventing the buttons from falling off the display.
        win.update_idletasks()
        fig_w, fig_h = fig.get_size_inches()
        w_win = max(int(fig_w * 85), 750)
        h_win = max(int(fig_h * 85) + 80, 500)
        win.geometry(f"{w_win}x{h_win}")

    def _plot_analysis(self, fig, gs, img_row, cols, img, results, analysis_id):
        def _ax_style(ax, title, xlabel, ylabel):
            ax.set_title(title, fontsize=9, color=MPL_TEXT, pad=5, fontweight="bold")
            ax.set_xlabel(xlabel, fontsize=8, color=C["subtext"])
            ax.set_ylabel(ylabel, fontsize=8, color=C["subtext"])
            ax.set_facecolor(MPL_BG)
            ax.tick_params(labelsize=7.5, colors=C["subtext"])
            ax.grid(True, color=MPL_GRID, linewidth=0.5, alpha=0.7, axis="y")
            for sp in ax.spines.values():
                sp.set_edgecolor(C["border"])
                sp.set_linewidth(0.8)

        palette = [MPL_HIST1, MPL_HIST2, "#27AE60", "#8E44AD", "#E67E22", "#16A085"]

        if analysis_id == "rgb_hist":
            half = max(1, cols // 2)
            ax1 = fig.add_subplot(gs[img_row, :half])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            for ch_idx, (ch_name, color) in enumerate(
                    [("R", "#E74C3C"), ("G", "#27AE60"), ("B", "#2980B9")]):
                h = cv2.calcHist([img_rgb], [ch_idx], None, [256], [0, 256]).flatten()
                ax1.plot(h, color=color, alpha=0.75, linewidth=1.4, label=ch_name)
            ax1.set_xlim([0, 255])
            ax1.legend(fontsize=8, loc="upper right")
            _ax_style(ax1, "Distribusi Channel RGB (Original)", "Intensitas", "Frekuensi")

            ax2 = fig.add_subplot(gs[img_row, half:])
            gray = _gray(img)
            h_gray = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            ax2.plot(h_gray, color="#5D6D7E", alpha=0.8, linewidth=1.4, label="Grayscale")
            ax2.set_xlim([0, 255])
            ax2.legend(fontsize=8)
            _ax_style(ax2, "Distribusi Grayscale", "Intensitas", "Frekuensi")

        elif analysis_id == "quality_hist":
            hists = []
            for title, disp, _ in results:
                g = cv2.cvtColor(cv2.cvtColor(disp, cv2.COLOR_RGB2BGR),
                                 cv2.COLOR_BGR2GRAY) if len(disp.shape) == 3 else disp
                h = cv2.calcHist([g], [0], None, [256], [0, 256]).flatten()
                hists.append((title.split("\n")[0], h))

            ax1 = fig.add_subplot(gs[img_row, :max(1, cols - 1)])
            for i, (lbl, h) in enumerate(hists):
                ax1.plot(h, color=palette[i % len(palette)], alpha=0.72,
                         linewidth=1.3, label=lbl)
            ax1.set_xlim([0, 255])
            ax1.legend(fontsize=6.5, loc="upper right", ncol=2)
            _ax_style(ax1, "Histogram Intensitas – Semua Metode", "Intensitas (0–255)", "Frekuensi")

            ax2 = fig.add_subplot(gs[img_row, -1])
            labels = [t for t, _ in hists]
            stds = [h.std() for _, h in hists]
            bars = ax2.bar(range(len(stds)), stds,
                           color=[palette[i % len(palette)] for i in range(len(stds))],
                           width=0.6, edgecolor="white", linewidth=0.8)
            ax2.set_xticks(range(len(labels)))
            ax2.set_xticklabels([l[:10] for l in labels], rotation=30, ha="right", fontsize=6.5)
            for bar, val in zip(bars, stds):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         f"{val:.1f}", ha="center", va="bottom", fontsize=7, color=MPL_TEXT)
            _ax_style(ax2, "Std Deviasi (Kontras)", "", "Std Dev")

        elif analysis_id == "sharpness_bar":
            ax = fig.add_subplot(gs[img_row, :])
            labels = [r[0].split("\n")[0] for r in results]
            scores = []
            for _, disp, _ in results:
                g = cv2.cvtColor(cv2.cvtColor(disp, cv2.COLOR_RGB2BGR),
                                 cv2.COLOR_BGR2GRAY) if len(disp.shape) == 3 else disp
                scores.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))

            colors_bar = [palette[i % len(palette)] for i in range(len(scores))]
            colors_bar[0] = "#AEB6BF"
            bars = ax.bar(range(len(scores)), scores, color=colors_bar,
                          width=0.55, edgecolor="white", linewidth=0.9)
            for bar, val in zip(bars, scores):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(scores) * 0.01,
                        f"{val:.1f}", ha="center", va="bottom", fontsize=8,
                        color=MPL_TEXT, fontweight="bold")
            ax.axhline(scores[0], color="#AEB6BF", linewidth=1.2,
                       linestyle="--", label=f"Original ({scores[0]:.1f})")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=9)
            ax.legend(fontsize=8)
            _ax_style(ax, "Perbandingan Sharpness (Laplacian Variance) — lebih tinggi = lebih tajam",
                      "Metode Filter", "Sharpness Score")

        elif analysis_id == "edge_bar":
            ax = fig.add_subplot(gs[img_row, :])
            edge_results = results[1:]
            labels = [r[0].split("\n")[0] for r in edge_results]
            densities = []
            for _, disp, _ in edge_results:
                g = disp if len(disp.shape) == 2 else cv2.cvtColor(
                    cv2.cvtColor(disp, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
                densities.append(_edgepct(g))

            colors_edge = ["#E74C3C", "#E67E22", "#8E44AD"]
            bars = ax.bar(range(len(densities)), densities,
                          color=colors_edge[:len(densities)],
                          width=0.45, edgecolor="white", linewidth=0.9)
            for bar, val in zip(bars, densities):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                        f"{val:.2f}%", ha="center", va="bottom", fontsize=9,
                        color=MPL_TEXT, fontweight="bold")
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=10)
            max_density = max(densities) if densities else 100
            ax.axhspan(0,  5,  alpha=0.08, color="#27AE60", label="Rendah (<5%)")
            ax.axhspan(5,  15, alpha=0.08, color="#F39C12", label="Sedang (5–15%)")
            ax.axhspan(15, max(100, max_density * 1.2),
                       alpha=0.08, color="#E74C3C", label="Tinggi (>15%)")
            ax.set_ylim([0, max(max_density * 1.25, 20)])
            ax.legend(fontsize=7.5, loc="upper right")
            _ax_style(ax, "Kepadatan Tepi (Edge Density) per Metode",
                      "Metode Deteksi Tepi", "Edge Pixels (%)")

        elif analysis_id == "segment_bar":
            ax = fig.add_subplot(gs[img_row, :])
            seg_results = results[1:]
            labels = [r[0].split("\n")[0] for r in seg_results]
            fg_pcts, bg_pcts = [], []
            for _, disp, _ in seg_results:
                g = disp if len(disp.shape) == 2 else cv2.cvtColor(
                    cv2.cvtColor(disp, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
                stats = segmentation_stats(binary)
                fg_pcts.append(stats["pct_fg"])
                bg_pcts.append(100 - stats["pct_fg"])

            x = np.arange(len(labels))
            w = 0.38
            b1 = ax.bar(x - w / 2, fg_pcts, width=w, label="Foreground (putih)",
                        color=MPL_HIST1, edgecolor="white", linewidth=0.8)
            b2 = ax.bar(x + w / 2, bg_pcts, width=w, label="Background (hitam)",
                        color="#AEB6BF", edgecolor="white", linewidth=0.8)
            for bar, val in zip(list(b1) + list(b2), fg_pcts + bg_pcts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5, color=MPL_TEXT)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=9)
            ax.set_ylim([0, 115])
            ax.legend(fontsize=8)
            _ax_style(ax, "Proporsi Foreground vs Background per Metode Segmentasi",
                      "Metode", "Persentase Piksel (%)")

    def _on_close(self):
        plt.close("all")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app  = SimpleApp(root)
    root.protocol("WM_DELETE_WINDOW", app._on_close)
    root.mainloop()