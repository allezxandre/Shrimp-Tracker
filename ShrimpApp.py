import tkinter as Tkinter
from tkinter import filedialog

from main import main


class FilePrompter(Tkinter.Tk):

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.initialize()


    def initialize(self):
        self.grid()
        self.filename_entry_var = Tkinter.StringVar()
        self.filename_entry = Tkinter.Entry(self, textvariable=self.filename_entry_var)
        self.filename_entry.grid(column=0, row=0, sticky='EW')
        self.filename_entry.bind("<Return>", self.onOKPressed)
        self.filename_entry_var.set("")

        button_open = Tkinter.Button(self, text="Open...", command=self.onOpen)
        button_open.grid(column=1, row=0)

        buttonOK = Tkinter.Button(self, text="OK", command=self.onOKPressed)
        buttonOK.grid(column=2, row=0)

        self.circle_entry_var1 = Tkinter.IntVar()
        self.circle_entry_var2 = Tkinter.IntVar()
        self.circle_entry_var3 = Tkinter.IntVar()

        self.circle_entry1 = Tkinter.Entry(self, textvariable=self.circle_entry_var1)
        self.circle_entry1.grid(column=0, row=1, sticky='EW')
        self.circle_entry2 = Tkinter.Entry(self, textvariable=self.circle_entry_var2)
        self.circle_entry2.grid(column=1, row=1, sticky='EW')
        self.circle_entry3 = Tkinter.Entry(self, textvariable=self.circle_entry_var3)
        self.circle_entry3.grid(column=2, row=1, sticky='EW')


        self.grid_columnconfigure(0, weight=1)
        self.resizable(True, False)

    @property
    def filename(self):
        return self.filename_entry_var.get()

    @property
    def circle(self):
        x, y, r = self.circle_entry_var1.get(), self.circle_entry_var2.get(), self.circle_entry_var3.get()
        if r < x and r < y and r > 0:
            return (x, y, r)
        else:
            return None

    def onOpen(self):
        ftypes = [('AVI files', '*.avi'), ('MP4 files', '*.mp4'), ('All files', '*')]
        dlg = filedialog.Open(self, filetypes=ftypes,
                              initialfile=self.filename_entry_var.get() if len(self.filename_entry_var.get()) > 0 else None)
        fl = dlg.show()
        self.filename_entry_var.set(fl)

    def onOKPressed(self):
        self.destroy()

    def validate_filepath(self):
        filename = self.filename_entry_var.get()
        return len(filename) > 0


if __name__ == "__main__":
    app = FilePrompter(None)
    app.title('Shrimp Tracker')
    app.mainloop()
    filename = app.filename
    circle = app.circle
    main(filename, resize=0.7, circle=circle)