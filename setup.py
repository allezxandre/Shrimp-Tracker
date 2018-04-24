import os
import sys

from cx_Freeze import setup, Executable

PYTHON_PATH = r'C:\Users\Alexandre\AppData\Local\Programs\Python\Python36-32'

base = None
if sys.platform == "win32":
    base = "Win32GUI"
    os.environ['TCL_LIBRARY'] = PYTHON_PATH + r'\tcl\tcl8.6'
    os.environ['TK_LIBRARY'] = PYTHON_PATH + r'\tcl\tk8.6'

setup(
    name='Shrimp-Tracker',
    version='1.0',
    options={"build_exe":
        {
            "packages": ["tkinter", "appdirs", "matplotlib", "numpy", "cv2", "scipy"],
            "include_files": [
                PYTHON_PATH + r"\DLLs\tcl86t.dll",
                PYTHON_PATH + r"\DLLs\tk86t.dll",
            ]
        }
    },
    packages=[''],
    url='',
    license='LGPLv3',
    author='A. Jouandin and R. Ross',
    author_email='',
    description='Computer Vision algorithm that tracks Shrimps',
    executables=[Executable("ShrimpApp.py", base=base)],
)
