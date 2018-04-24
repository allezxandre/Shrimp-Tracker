import sys

from cx_Freeze import setup, Executable

base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name='Shrimp-Tracker',
    version='1.0',
    packages=[''],
    url='',
    license='LGPLv3',
    author='A. Jouandin and R. Ross',
    author_email='',
    description='Computer Vision algorithm that tracks Shrimps',
    executables=[Executable("ShrimpApp.py", base=base)],
)
