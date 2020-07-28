import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as tkfd
from tkinter import messagebox as tkmb
import platform
from collections import deque
from threading import Thread

from PIL import ImageTk

from calc_sim_utils import extract_image_paths, calc_similarities_combinatorially
from output_utils import output_csv, output_excel

WIDTH = 1000  # default window width
HEIGHT = 500  # default window height
DETECT_MIN_LENGTH = 500  # minimum length of images while calculating similarities
NUM_DISPLAY = 15  # number of images displayed on the result screen
DISPLAY_WIDTH = 300  # width of images displayed on the result screen
DISPLAY_INTERVAL = 20  # interval length of images displayed on the result screen
OS = platform.system()  # some tkinter actions differ among different OSes

progress_q = deque()
sim_scores = []
sim_src_paths = []
display_imgs = []
frames = {}


class InputFrame(ttk.Frame):
    def __init__(self, root):
        super().__init__(root)
        text = '''
            クリックしてディレクトリを選択してください。\n
            ディレクトリ内の全画像に対して、同じ画像らしさを計算します。'''
        self.label = tk.Label(self, text=text, justify='center', font=('', '14'))
        self.activate()

    def activate(self):
        self.grid(row=0, column=0, sticky='ewsn')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.label.grid(row=0, column=0, sticky='ewsn')
        self.label.bind('<Button-1>', self.select_dir)

    @staticmethod
    def select_dir(event):
        dir_path = tkfd.askdirectory()
        if dir_path:
            img_paths = extract_image_paths(dir_path)
            nums = len(img_paths)
            if nums:
                tkmb.showinfo(
                    '検出を開始します。',
                    f'{nums}枚の画像が見つかりました。')
                frames['DetectFrame'].detect(dir_path)
            else:
                tkmb.showwarning(
                    '再度選択してください。',
                    '画像が見つかりませんでした。')


class DetectFrame(ttk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.label_frame = ttk.LabelFrame(self, text='')
        self.label = ttk.Label(self.label_frame, text='検出中です...', font=('', '14'))
        self.progressbar = ttk.Progressbar(self.label_frame)
        self.button = ttk.Button(self.label_frame, text='結果を確認', command=frames['ResultFrame'].show_results)
        self.activate()

    def activate(self):
        self.grid(row=0, column=0, sticky='ewsn')
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.label_frame.grid(row=0, column=0)
        self.label.grid(row=0, column=0)
        self.progressbar.configure(value=0, mode='determinate', maximum=100, length=WIDTH-100)
        self.progressbar.grid(row=1, column=0)

    def reset(self):
        global sim_scores, display_imgs
        sim_scores = []
        display_imgs = []
        self.button.grid_forget()

    def detect(self, dir_path):
        self.reset()
        self.tkraise()

        thread = Thread(
            target=calc_similarities_combinatorially,
            args=(dir_path, progress_q, sim_scores, sim_src_paths, display_imgs, DETECT_MIN_LENGTH, DISPLAY_WIDTH),
            daemon=True)
        thread.start()

        while True:
            if len(progress_q) > 0:
                progress = progress_q.pop()
                self.progressbar.configure(value=progress)
                self.progressbar.update()
                if progress == 100:
                    break
        progress_q.clear()

        self.button.grid(row=2, column=0)


class ResultFrame(ttk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.canvas = tk.Canvas(self)
        self.label_frame = ttk.LabelFrame(self, text='')
        self.button_return = ttk.Button(self.label_frame, text='最初に戻る', command=frames['InputFrame'].tkraise)
        self.button_csv = ttk.Button(self.label_frame, text='CSVに出力', command=self.output_csv)
        self.button_excel = ttk.Button(self.label_frame, text='Excelに出力', command=self.output_excel)
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
        if OS == 'Linux':
            self.canvas.bind('<4>', self._on_mouse_wheel)
            self.canvas.bind('<5>', self._on_mouse_wheel)
        else:
            self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)

    def _on_mouse_wheel(self, event):
        if OS == 'Darwin':
            self.canvas.yview_scroll(-1 * event.delta, 'units')
        else:
            self.canvas.yview_scroll(-1 * (event.delta // 120), 'units')

    def show_results(self):
        # sort images by scores
        _, sorted_display_imgs = zip(*sorted(zip(sim_scores, display_imgs), key=lambda x: x[0], reverse=True))
        sorted_display_imgs = sorted_display_imgs[:NUM_DISPLAY]

        # display images
        num = 1
        pos_y = DISPLAY_INTERVAL
        for idx in range(0, len(sorted_display_imgs), 3):
            pos_x = DISPLAY_INTERVAL
            row_imgs = sorted_display_imgs[idx:idx+3]
            max_height = max(row_img.size[1] for row_img in row_imgs)
            for row_img in row_imgs:
                tkimg = ImageTk.PhotoImage(image=row_img)
                exec(f'self.canvas.img{num} = tkimg')
                self.canvas.create_image(pos_x, pos_y, image=eval(f'self.canvas.img{num}'), anchor='nw')
                num += 1
                pos_x += (DISPLAY_WIDTH + DISPLAY_INTERVAL)
            pos_y += (max_height + DISPLAY_INTERVAL)

        self.canvas.config(scrollregion=self.canvas.bbox('all'))
        self.tkraise()

    @staticmethod
    def output_csv():
        output_csv()

    @staticmethod
    def output_excel():
        output_excel(sim_scores, sim_src_paths, 'test.xlsx')


def main():
    root = tk.Tk()
    root.title('同一画像判定アプリ')
    root.geometry(f'{WIDTH}x{HEIGHT}')
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)

    frames['InputFrame'] = InputFrame(root)
    frames['ResultFrame'] = ResultFrame(root)
    frames['DetectFrame'] = DetectFrame(root)

    frames['InputFrame'].tkraise()
    root.mainloop()


if __name__ == '__main__':
    main()
