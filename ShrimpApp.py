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
        label1 = Tkinter.Label(self, text="File Name")
        label2 = Tkinter.Label(self, text="Circle (cx, cy, r)")
        label3 = Tkinter.Label(self, text="Kalman coefficients")
        label4 = Tkinter.Label(self, text="Cx")
        label5 = Tkinter.Label(self, text="Cy")
        label6 = Tkinter.Label(self, text="Area")
        label7 = Tkinter.Label(self, text="Angle")
        label8 = Tkinter.Label(self, text="Lambda 1")
        label9 = Tkinter.Label(self, text="Lambda 2")
        label10 = Tkinter.Label(self, text="Velocity")

        label1.grid(column=0, row=0, sticky='EW')

        self.filename_entry_var = Tkinter.StringVar()
        self.filename_entry = Tkinter.Entry(self, textvariable=self.filename_entry_var)
        self.filename_entry.grid(column=1, row=0, sticky='EW')
        self.filename_entry.bind("<Return>", self.onOKPressed)
        self.filename_entry_var.set("")

        button_open = Tkinter.Button(self, text="Open...", command=self.onOpen)
        button_open.grid(column=2, row=0)

        buttonOK = Tkinter.Button(self, text="OK", command=self.onOKPressed)
        buttonOK.grid(column=3, row=6)

        self.circle_entry_var1 = Tkinter.IntVar()
        self.circle_entry_var2 = Tkinter.IntVar()
        self.circle_entry_var3 = Tkinter.IntVar()
        self.kalman_filter_cx = Tkinter.IntVar()
        self.kalman_filter_cy = Tkinter.IntVar()
        self.kalman_filter_area = Tkinter.IntVar()
        self.kalman_filter_angle = Tkinter.IntVar()
        self.kalman_filter_lambda1 = Tkinter.IntVar()
        self.kalman_filter_lambda2 = Tkinter.IntVar()
        self.kalman_filter_vel = Tkinter.IntVar()

        label2.grid(column=0, row=1, sticky='EW')
        self.circle_entry1 = Tkinter.Entry(self, textvariable=self.circle_entry_var1)
        self.circle_entry1.grid(column=1, row=1, sticky='EW')
        self.circle_entry2 = Tkinter.Entry(self, textvariable=self.circle_entry_var2)
        self.circle_entry2.grid(column=2, row=1, sticky='EW')
        self.circle_entry3 = Tkinter.Entry(self, textvariable=self.circle_entry_var3)
        self.circle_entry3.grid(column=3, row=1, sticky='EW')

        label3.grid(column=0, row=2, sticky='EW')
        label4.grid(column=0, row=3, sticky='EW')
        self.kalman_filter_cx = Tkinter.Entry(self, textvariable=self.kalman_filter_cx)
        self.kalman_filter_cx.grid(column=1, row=3, sticky='EW')
        label5.grid(column=2, row=3, sticky='EW')
        self.kalman_filter_cy = Tkinter.Entry(self, textvariable=self.kalman_filter_cy)
        self.kalman_filter_cy.grid(column=3, row=3, sticky='EW')
        label6.grid(column=0, row=4, sticky='EW')
        self.kalman_filter_angle= Tkinter.Entry(self, textvariable=self.kalman_filter_angle)
        self.kalman_filter_angle.grid(column=1, row=4, sticky='EW')
        label7.grid(column=2, row=4, sticky='EW')
        self.kalman_filter_area = Tkinter.Entry(self, textvariable=self.kalman_filter_area)
        self.kalman_filter_area.grid(column=3, row=4, sticky='EW')
        label8.grid(column=0, row=5, sticky='EW')
        self.kalman_filter_lambda1 = Tkinter.Entry(self, textvariable=self.kalman_filter_lambda1)
        self.kalman_filter_lambda1.grid(column=1, row=5, sticky='EW')
        label9.grid(column=2, row=5, sticky='EW')
        self.kalman_filter_lambda2 = Tkinter.Entry(self, textvariable=self.kalman_filter_lambda2)
        self.kalman_filter_lambda2.grid(column=3, row=5, sticky='EW')
        label10.grid(column=0, row=6, sticky='EW')
        self.kalman_filter_vel = Tkinter.Entry(self, textvariable=self.kalman_filter_vel)
        self.kalman_filter_vel.grid(column=1, row=6, sticky='EW')

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

        @property
        def kalman_filter(self):
            cx, cy, area, angle, lambda1, lambda2, vel = self.kalman_filter_cx.get(), self.kalman_filter_cy.get(), \
                                                         self.kalman_filter_area.get(), self.kalman_filter_angle.get(), \
                                                         self.kalman_filter_lambda1.get(), self.kalman_filter_lambda2.get(),\
                                                         self.kalman_filter_vel.get()
            return (cx, cy, area, angle, lambda1, lambda2, vel)

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