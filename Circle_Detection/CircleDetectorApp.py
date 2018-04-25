import threading
import tkinter as tk
from tkinter import ttk

from Circle_Detection.hough_circle import find_circle


class CircleFinderProgressWindow(tk.Tk):

    def __init__(self, screenName=None, baseName=None, className='Tk', useTk=1, sync=0, use=None):
        super().__init__(screenName, baseName, className, useTk, sync, use)
        self.__initialize_UI()
        self.__circle = None

    def __initialize_UI(self):
        self.grid()
        self.title('Shrimp-Tracker')
        self.label = tk.Label(self, text="Computing Circle...")
        self.label.grid(column=0, row=0, sticky='EW')
        self.progress_bar = ttk.Progressbar(self, mode='determinate')
        self.progress_bar.grid(column=0, row=1, sticky='EW')

        self.grid_columnconfigure(0, weight=1)
        self.resizable(False, False)

    @property
    def circle(self):
        return self.__circle

    def launch(self, filename, resize):
        self.progress = 0
        def progress_update(current, total):
            self.progress = round(current / float(total) * 100)
        def thread_work():
            _, self.__circle = find_circle(filename, resize=resize, progress_func=progress_update)

        t = threading.Thread(target=thread_work)
        t.start()
        previous_progress = self.progress
        while t.is_alive():
            if previous_progress != self.progress:
                self.progress_bar["value"] = self.progress
                self.update()
            previous_progress = self.progress
        self.destroy()


def detectCircle(filename, resize=None):
    app = CircleFinderProgressWindow()
    app.launch(filename=filename, resize=resize)
    app.mainloop()
    return app.circle