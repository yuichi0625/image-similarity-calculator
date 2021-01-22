import tkinter as tk

import config as c
from .frames import reset_variables, InputFrame, DetectFrame, ResultFrame


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title('画像類似度算出アプリ')
        self.geometry(f'{c.SCREEN_WIDTH}x{c.SCREEN_HEIGHT}')
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self.input_frame = InputFrame(self)
        self.detect_frame = DetectFrame(self)
        self.result_frame = ResultFrame(self)

        # https://stackoverflow.com/questions/35029188/how-would-i-make-a-method-which-is-run-every-time-a-frame-is-shown-in-tkinter
        self.input_frame.bind(c.BIND_DETECT, self.show_detect_frame)
        self.detect_frame.bind(c.BIND_INPUT, self.show_input_frame)
        self.detect_frame.bind(c.BIND_RESULT, self.show_result_frame)
        self.result_frame.bind(c.BIND_INPUT, self.show_input_frame)

        self.show_input_frame()

    def show_input_frame(self, *args) -> None:
        reset_variables()
        self.input_frame.tkraise()

    def show_detect_frame(self, *args) -> None:
        self.detect_frame.tkraise()
        self.detect_frame.detect()

    def show_result_frame(self, *args) -> None:
        self.result_frame.tkraise()
        self.result_frame.display_result()
