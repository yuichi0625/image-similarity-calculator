import platform
import re
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.filedialog as tkfd
import tkinter.messagebox as tkmb
from itertools import islice
from threading import Thread
from typing import List

from PIL import ImageTk

import config as c
from utils.image_processor import calc_sim
from utils.image_generator import generate_display_images
from utils.output_generator import output_excel, output_csv

# フレーム間で共有する変数
img_paths: List[str] = []  # 入力した画像のパス
sim_img_paths: List[List[str]] = []  # 類似度を計算する画像のパスのペア
sim_scores: List[float] = []  # 上記に対応した類似度の値
ov_coords: List[List[List[int]]] = []  # 上記に対応した重なり合う部分の座標


def reset_variables():
    """フレーム間で共有する変数を初期化する
    """
    global img_paths, sim_img_paths, sim_scores, ov_coords
    img_paths = []
    sim_img_paths = []
    sim_scores = []
    ov_coords = []


class InputFrame(ttk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.label = tk.Label(self, text=c.WELCOME_TEXT, justify='center', font=('', '14'))
        self.activate()

    def activate(self):
        self.grid(row=0, column=0, sticky='ewsn')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.label.grid(row=0, column=0, sticky='ewsn')
        self.label.bind('<Button-1>', self.select)

    def select(self, event):
        global img_paths
        regex = re.compile('.(jpg|jpeg|png)')
        img_paths = tkfd.askopenfilenames()
        img_paths = [img_path for img_path in img_paths
                     if regex.search(img_path) and img_path.isascii()]
        if (num := len(img_paths)) > 1:
            tkmb.showinfo(
                title='検出を開始します',
                message=f'入力画像は{num}枚です。')
            self.event_generate(c.BIND_DETECT)
        else:
            tkmb.showwarning(
                title='再度選択してください',
                message='画像を2枚以上選んでください。')


class DetectFrame(ttk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.label_frame = ttk.LabelFrame(self, text='')
        self.label = ttk.Label(self.label_frame, text='検出中です...', font=('', '14'))
        self.progressbar = ttk.Progressbar(self.label_frame)
        self.button = ttk.Button(
            self.label_frame,
            text='結果を確認',
            command=lambda: self.event_generate(c.BIND_RESULT))
        self.activate()

    def activate(self):
        self.grid(row=0, column=0, sticky='ewsn')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.label_frame.grid(row=0, column=0)
        self.label.grid(row=0, column=0)
        self.progressbar.configure(
            value=0,
            mode='determinate',
            maximum=100,
            length=c.SCREEN_WIDTH-100)
        self.progressbar.grid(row=1, column=0)

    def detect(self):
        thread = Thread(
            target=calc_sim,
            args=(img_paths, sim_img_paths, sim_scores, ov_coords),
            daemon=True)
        thread.start()

        num_comb = len(img_paths) * (len(img_paths) - 1) / 2
        while True:
            prog_pct = len(sim_img_paths) / num_comb * 100
            self.progressbar.config(value=prog_pct)
            self.progressbar.update()
            if prog_pct == 100:
                break

        self.button.grid(row=2, column=0)


class ResultFrame(ttk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.canvas = tk.Canvas(self)
        self.label_frame = ttk.LabelFrame(self, text='')
        self.button_return = ttk.Button(
            self.label_frame,
            text='最初に戻る',
            command=lambda: self.event_generate(c.BIND_INPUT))
        self.button_csv = ttk.Button(
            self.label_frame,
            text='CSVに出力',
            command=self.output_csv)
        self.button_excel = ttk.Button(
            self.label_frame,
            text='Excelに出力',
            command=self.output_excel)
        self.scroll_x = tk.Scrollbar(self, orient=tk.HORIZONTAL)
        self.scroll_y = tk.Scrollbar(self, orient=tk.VERTICAL)
        self.activate()

    def activate(self):
        self.grid(row=0, column=0, sticky='ewsn')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.label_frame.grid(row=1, column=0)
        self.button_return.grid(row=0, column=0)
        self.button_csv.grid(row=0, column=1)
        self.button_excel.grid(row=0, column=2)
        self.scroll_x.config(command=self.canvas.xview)
        self.scroll_x.grid(row=2, column=0, sticky='ews')
        self.scroll_y.config(command=self.canvas.yview)
        self.scroll_y.grid(row=0, column=1, rowspan=3, sticky='esn')
        self.canvas.config(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)
        self.canvas.grid(row=0, column=0, sticky='ewsn')
        if platform.system() == 'Linux':
            self.canvas.bind('<4>', self._on_mouse_wheel)
            self.canvas.bind('<5>', self._on_mouse_wheel)
        else:  # 'Darwin' or 'Windows'
            self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)

    def _on_mouse_wheel(self, event):
        if platform.system() == 'Darwin':
            self.canvas.yview_scroll(-1 * event.delta, 'units')
        else:  # 'Linux' or 'Windows'
            self.canvas.yview_scroll(-1 * (event.delta // 120), 'units')

    def display_result(self):
        row_imgs_gen = generate_display_images(sim_img_paths, sim_scores, ov_coords)

        num = 1
        pos_y = c.DISPLAY_INTERVAL
        while True:
            row_imgs = list(islice(row_imgs_gen, 0, 3))
            if not row_imgs:
                break

            pos_x = c.DISPLAY_INTERVAL
            height = max(row_img.size[1] for row_img in row_imgs)
            for row_img in row_imgs:
                tkimg = ImageTk.PhotoImage(image=row_img)
                exec(f'self.canvas.img{num} = tkimg')
                self.canvas.create_image(pos_x, pos_y, image=eval(f'self.canvas.img{num}'), anchor='nw')
                num += 1
                pos_x += (c.DISPLAY_IMAGE_WIDTH + c.DISPLAY_INTERVAL)
            pos_y += (height + c.DISPLAY_INTERVAL)

        self.canvas.config(scrollregion=self.canvas.bbox('all'))

    @staticmethod
    def output_csv():
        output_csv(sim_img_paths, sim_scores)
        tkmb.showinfo(
            title='保存完了',
            message='CSVファイルの保存が完了しました。')

    @staticmethod
    def output_excel():
        output_excel(sim_img_paths, sim_scores)
        tkmb.showinfo(
            title='保存完了',
            message='Excelファイルの保存が完了しました。')
