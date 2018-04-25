import tkinter as Tkinter
from tkinter import filedialog

from Circle_Detection.CircleDetectorApp import detectCircle
from Circle_Detection.circle_crop import CircleCrop
from SettingsSaver import SettingsSaver
from main import main

EXIT_CODE_NO_FILENAME_PROVIDED = 5

class FilePrompter(Tkinter.Tk):

    def __init__(self, parent, settings_saver: SettingsSaver):
        super().__init__(parent)
        self.settings_saver = settings_saver
        self.parent = parent
        self.initialize()

    def initialize(self):

        self.grid()
        label1 = Tkinter.Label(self, text="File Name")
        label2 = Tkinter.Label(self, text="Circle (cx, cy, r)")
        label3 = Tkinter.Label(self, text="Observation noise coefficients")
        label4 = Tkinter.Label(self, text="Cx")
        label5 = Tkinter.Label(self, text="Cy")
        label6 = Tkinter.Label(self, text="Angle")
        label7 = Tkinter.Label(self, text="Area")
        label8 = Tkinter.Label(self, text="Lambda 1")
        label9 = Tkinter.Label(self, text="Lambda 2")

        label1.grid(column=0, row=0, sticky='EW')

        self.filename_entry_var = Tkinter.StringVar()
        self.filename_entry = Tkinter.Entry(self, textvariable=self.filename_entry_var)
        self.filename_entry.grid(column=1, columnspan=2, row=0, sticky='EW')
        self.filename_entry.bind("<Return>", self.onOKPressed)
        self.filename_entry_var.set("")

        button_open = Tkinter.Button(self, text="Open...", command=self.onOpen)
        button_open.grid(column=3, row=0)

        buttonOK = Tkinter.Button(self, text="OK", command=self.onOKPressed)
        buttonOK.grid(column=3, row=6)

        self.circle_entry_var1 = Tkinter.IntVar()
        self.circle_entry_var2 = Tkinter.IntVar()
        self.circle_entry_var3 = Tkinter.IntVar()
        self.kalman_filter_cx_var = Tkinter.DoubleVar()
        self.kalman_filter_cy_var = Tkinter.DoubleVar()
        self.kalman_filter_area_var = Tkinter.DoubleVar()
        self.kalman_filter_angle_var = Tkinter.DoubleVar()
        self.kalman_filter_lambda1_var = Tkinter.DoubleVar()
        self.kalman_filter_lambda2_var = Tkinter.DoubleVar()
        self.kalman_filter_vel_var = Tkinter.DoubleVar()

        # Circle
        label2.grid(column=0, row=1, sticky='EW')
        self.circle_entry1 = Tkinter.Entry(self, textvariable=self.circle_entry_var1)
        self.circle_entry1.grid(column=1, row=1, sticky='EW')
        self.circle_entry2 = Tkinter.Entry(self, textvariable=self.circle_entry_var2)
        self.circle_entry2.grid(column=2, row=1, sticky='EW')
        self.circle_entry3 = Tkinter.Entry(self, textvariable=self.circle_entry_var3)
        self.circle_entry3.grid(column=3, row=1, sticky='EW')

        # Kalman
        label3.grid(column=0, row=2, sticky='EW')
        label4.grid(column=0, row=3, sticky='EW')
        self.kalman_filter_cx = Tkinter.Entry(self, textvariable=self.kalman_filter_cx_var)
        self.kalman_filter_cx.grid(column=1, row=3, sticky='EW')
        label5.grid(column=2, row=3, sticky='EW')
        self.kalman_filter_cy = Tkinter.Entry(self, textvariable=self.kalman_filter_cy_var)
        self.kalman_filter_cy.grid(column=3, row=3, sticky='EW')
        label6.grid(column=0, row=4, sticky='EW')
        self.kalman_filter_angle = Tkinter.Entry(self, textvariable=self.kalman_filter_angle_var)
        self.kalman_filter_angle.grid(column=1, row=4, sticky='EW')
        label7.grid(column=2, row=4, sticky='EW')
        self.kalman_filter_area = Tkinter.Entry(self, textvariable=self.kalman_filter_area_var)
        self.kalman_filter_area.grid(column=3, row=4, sticky='EW')
        label8.grid(column=0, row=5, sticky='EW')
        self.kalman_filter_lambda1 = Tkinter.Entry(self, textvariable=self.kalman_filter_lambda1_var)
        self.kalman_filter_lambda1.grid(column=1, row=5, sticky='EW')
        label9.grid(column=2, row=5, sticky='EW')
        self.kalman_filter_lambda2 = Tkinter.Entry(self, textvariable=self.kalman_filter_lambda2_var)
        self.kalman_filter_lambda2.grid(column=3, row=5, sticky='EW')

        self.grid_columnconfigure(0, weight=1)
        self.resizable(True, False)

        self.input_defaults()

    def input_defaults(self):
        self.set_circle(None)
        from numpy import diag, pi
        self.set_kalman(diag([3.0, 3.0, pi / 6, 10., 10., 5.]))

    @property
    def filename(self):
        return self.filename_entry_var.get()

    def set_circle(self, circle):
        if circle is not None:
            print(circle)
            x, y, r = circle
            self.circle_entry_var1.set(x)
            self.circle_entry_var2.set(y)
            self.circle_entry_var3.set(r)
        else:
            self.circle_entry_var1.set('')
            self.circle_entry_var2.set('')
            self.circle_entry_var3.set('')

    def set_kalman(self, kalman):
        if kalman is not None:
            from numpy import diag, round
            cx, cy, angle, area, lambda1, lambda2 = diag(kalman)
            self.kalman_filter_cx_var.set(cx)
            self.kalman_filter_cy_var.set(cy)
            self.kalman_filter_area_var.set(area)
            from numpy import rad2deg
            if abs(angle) < 1:
                angle_deg = round(rad2deg(angle))
            else:
                angle_deg = angle  # The angle is probably already in deg unit
            self.kalman_filter_angle_var.set(angle_deg)
            self.kalman_filter_lambda1_var.set(lambda1)
            self.kalman_filter_lambda2_var.set(lambda2)

    @property
    def circle(self):
        try:
            x, y, r = self.circle_entry_var1.get(), self.circle_entry_var2.get(), self.circle_entry_var3.get()
        except:
            return None
        else:
            if r < x and r < y and r > 0:
                return (x, y, r)
            else:
                return None

    @property
    def kalman(self):
        from numpy import deg2rad, diag
        cx, cy, angle_deg, area, lambda1, lambda2 = self.kalman_filter_cx_var.get(), self.kalman_filter_cy_var.get(), \
                                                    self.kalman_filter_angle_var.get(), self.kalman_filter_area_var.get(), \
                                                    self.kalman_filter_lambda1_var.get(), self.kalman_filter_lambda2_var.get()
        angle = deg2rad(angle_deg)
        return diag([cx, cy, angle, area, lambda1, lambda2])

    def onOpen(self):
        ftypes = [('Video files', '*.avi;*.mp4;*.mpeg'), ('All files', '*')]
        dlg = filedialog.Open(self, filetypes=ftypes,
                              initialfile=self.filename_entry_var.get() if len(
                                  self.filename_entry_var.get()) > 0 else None)
        fl = dlg.show()
        self.filename_entry_var.set(fl)
        cache = self.settings_saver.read_from_cache(fl)
        self.set_circle(cache.circle)
        self.set_kalman(cache.kalman)

    def onOKPressed(self):
        self.destroy()

    def validate_filepath(self):
        filename = self.filename_entry_var.get()
        return len(filename) > 0


if __name__ == "__main__":
    resize = 0.7

    settings = SettingsSaver()

    app = FilePrompter(None, settings)
    app.title('Shrimp Tracker')
    app.mainloop()

    # User pressed OK at this point
    filename = app.filename
    if len(filename) == 0: exit(EXIT_CODE_NO_FILENAME_PROVIDED)
    circle = app.circle
    kalman = app.kalman

    if circle is None:
        # Detect circle
        circle = detectCircle(filename, resize=resize)

    # Save settings to cache
    settings.add_to_cache(filename, circle=circle)
    settings.add_to_cache(filename, kalman=kalman)

    main(filename, resize=resize, circle=circle, kalman=kalman)
